from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.backtest.engine import BacktestConfig, DailyBacktester
from ashare_quant.data.csv_adapter import CSVDataSource, ensure_price_views
from ashare_quant.labels.label_builder import ExecutionSpec, add_label_columns
from ashare_quant.models.offline_signal_model import OfflineSignalPredictor
from ashare_quant.pipeline import DEFAULT_BACKTEST, DEFAULT_RESEARCH, load_json, prepare_research_frame, score_research_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate candidate-pool rerank with ml_score inside original top pool.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--research-config", required=True)
    parser.add_argument("--backtest-config", required=True)
    parser.add_argument("--ml-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--train-start", default="2020-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2026-04-08")
    parser.add_argument("--candidate-top-n", type=int, default=20)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--compare-top-ns", default="5,10")
    return parser


def _next_trade_date_map(dates: list[pd.Timestamp], lag: int) -> dict[pd.Timestamp, pd.Timestamp]:
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for idx, date in enumerate(dates):
        target_idx = idx + lag
        if target_idx < len(dates):
            mapping[pd.Timestamp(date)] = pd.Timestamp(dates[target_idx])
    return mapping


def _manual_targets(
    selected_map: dict[pd.Timestamp, pd.DataFrame],
    *,
    dates: list[pd.Timestamp],
    execution_lag: int,
) -> pd.DataFrame:
    execution_map = _next_trade_date_map(dates, lag=execution_lag)
    rows: list[dict[str, Any]] = []
    for signal_date, sl in selected_map.items():
        if signal_date not in execution_map:
            continue
        execution_date = execution_map[signal_date]
        if sl.empty:
            continue
        weight = 1.0 / len(sl)
        ranked = sl.reset_index(drop=True).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        for row in ranked.itertuples(index=False):
            rows.append(
                {
                    "date": pd.Timestamp(signal_date),
                    "signal_date": pd.Timestamp(signal_date),
                    "execution_date": pd.Timestamp(execution_date),
                    "sleeve": 0,
                    "code": getattr(row, "code"),
                    "target_weight": float(weight),
                    "score": float(getattr(row, "score_for_portfolio")),
                    "rank": int(getattr(row, "rank")),
                    "industry": getattr(row, "industry", ""),
                    "trade_reason": getattr(row, "trade_reason", ""),
                    "risk_state": "full_risk",
                    "risk_multiplier": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _bucket_stats(picks_df: pd.DataFrame) -> dict[str, float]:
    if picks_df.empty:
        return {"rows": 0, "hit_rate": 0.0, "avg_max_up_5d": 0.0}
    return {
        "rows": int(len(picks_df)),
        "hit_rate": float(pd.to_numeric(picks_df["label_high_5d_up"], errors="coerce").mean()),
        "avg_max_up_5d": float(pd.to_numeric(picks_df["label_high_5d_up_max_up"], errors="coerce").mean()),
    }


def _run_portfolio(panel: pd.DataFrame, targets: pd.DataFrame, *, backtest_cfg: dict[str, Any], research_cfg: dict[str, Any]) -> dict[str, float]:
    if targets.empty:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
            "payoff_ratio": 0.0,
            "avg_turnover": 0.0,
            "gross_annual_return": 0.0,
            "after_cost_return_drag": 0.0,
        }
    _, metrics = DailyBacktester(
        config=BacktestConfig(
            **backtest_cfg,
            signal_time=str(research_cfg["signal_time"]),
            execution_price=str(research_cfg["execution_price"]),
            execution_lag=int(research_cfg["execution_lag"]),
            holding_window=5,
            sleeve_count=1,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            trailing_stop_pct=0.0,
        )
    ).run(panel, targets)
    return {
        "annual_return": float(metrics.get("annual_return", 0.0)),
        "annual_volatility": float(metrics.get("annual_volatility", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "hit_rate": float(metrics.get("hit_rate", 0.0)),
        "payoff_ratio": float(metrics.get("payoff_ratio", 0.0)),
        "avg_turnover": float(metrics.get("avg_turnover", 0.0)),
        "gross_annual_return": float(metrics.get("gross_annual_return", 0.0)),
        "after_cost_return_drag": float(metrics.get("after_cost_return_drag", 0.0)),
    }


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    research_cfg = {**DEFAULT_RESEARCH, **load_json(args.research_config)}
    research_cfg["use_ml_score"] = False
    backtest_cfg = {**DEFAULT_BACKTEST, **load_json(args.backtest_config)}
    predictor = OfflineSignalPredictor.load(args.ml_model_path)
    compare_top_ns = [int(x.strip()) for x in str(args.compare_top_ns).split(",") if x.strip()]

    raw_df = ensure_price_views(CSVDataSource(args.data_path, adjust=args.adjust).load_daily_bars())
    scored_df, metadata = prepare_research_frame(raw_df, research_cfg)
    scored_df = add_label_columns(
        scored_df,
        horizons=[],
        spec=ExecutionSpec(
            signal_time=str(research_cfg["signal_time"]),
            execution_price=str(research_cfg["execution_price"]),
            execution_lag=int(research_cfg["execution_lag"]),
            holding_window=5,
        ),
        label_types=["high_5d_up"],
        binary_up_threshold=0.05,
    )

    train_mask = (scored_df["date"] >= pd.Timestamp(args.train_start)) & (scored_df["date"] <= pd.Timestamp(args.train_end))
    train_dates = set(pd.to_datetime(scored_df.loc[train_mask, "date"]).drop_duplicates())
    scored_df, score_details = score_research_frame(
        df=scored_df,
        research_cfg=research_cfg,
        metadata=metadata,
        train_dates=train_dates,
    )

    test_mask = (scored_df["date"] >= pd.Timestamp(args.test_start)) & (scored_df["date"] <= pd.Timestamp(args.test_end))
    panel = scored_df.loc[test_mask].copy()
    all_dates = sorted(pd.to_datetime(panel["date"].drop_duplicates()))
    rebalance_dates = all_dates[:: max(int(args.rebalance_every), 1)]

    original_pick_frames: dict[int, list[pd.DataFrame]] = {n: [] for n in compare_top_ns}
    rerank_pick_frames: dict[int, list[pd.DataFrame]] = {n: [] for n in compare_top_ns}
    original_targets_map: dict[int, dict[pd.Timestamp, pd.DataFrame]] = {n: {} for n in compare_top_ns}
    rerank_targets_map: dict[int, dict[pd.Timestamp, pd.DataFrame]] = {n: {} for n in compare_top_ns}

    for signal_date in rebalance_dates:
        sl = panel.loc[panel["date"] == signal_date].copy()
        eligible = sl.loc[sl["strategy_tradeable"].fillna(False).astype(bool) & sl["score"].notna()].copy()
        if eligible.empty:
            continue
        candidate = eligible.sort_values("score", ascending=False).head(int(args.candidate_top_n)).copy()
        if candidate.empty:
            continue
        candidate["score_original"] = candidate["score"]
        candidate["ml_score"] = predictor.predict(candidate)
        reranked = candidate.sort_values(["ml_score", "score_original"], ascending=[False, False]).copy()

        for top_n in compare_top_ns:
            original_sel = candidate.head(top_n).copy()
            original_sel["score_for_portfolio"] = original_sel["score_original"]
            rerank_sel = reranked.head(top_n).copy()
            rerank_sel["score_for_portfolio"] = rerank_sel["ml_score"]
            original_pick_frames[top_n].append(original_sel)
            rerank_pick_frames[top_n].append(rerank_sel)
            original_targets_map[top_n][pd.Timestamp(signal_date)] = original_sel
            rerank_targets_map[top_n][pd.Timestamp(signal_date)] = rerank_sel

    rows = []
    details: dict[str, Any] = {
        "score_details": score_details,
        "compare_top_ns": compare_top_ns,
    }

    for top_n in compare_top_ns:
        original_picks = pd.concat(original_pick_frames[top_n], ignore_index=True) if original_pick_frames[top_n] else pd.DataFrame()
        rerank_picks = pd.concat(rerank_pick_frames[top_n], ignore_index=True) if rerank_pick_frames[top_n] else pd.DataFrame()
        original_stats = _bucket_stats(original_picks)
        rerank_stats = _bucket_stats(rerank_picks)
        original_targets = _manual_targets(original_targets_map[top_n], dates=all_dates, execution_lag=int(research_cfg["execution_lag"]))
        rerank_targets = _manual_targets(rerank_targets_map[top_n], dates=all_dates, execution_lag=int(research_cfg["execution_lag"]))
        original_portfolio = _run_portfolio(panel, original_targets, backtest_cfg=backtest_cfg, research_cfg=research_cfg)
        rerank_portfolio = _run_portfolio(panel, rerank_targets, backtest_cfg=backtest_cfg, research_cfg=research_cfg)

        rows.append(
            {
                "portfolio": f"original_top{top_n}",
                "candidate_top_n": int(args.candidate_top_n),
                "selected_rows": original_stats["rows"],
                "hit_rate": original_stats["hit_rate"],
                "avg_max_up_5d": original_stats["avg_max_up_5d"],
                **original_portfolio,
            }
        )
        rows.append(
            {
                "portfolio": f"rerank_top{top_n}",
                "candidate_top_n": int(args.candidate_top_n),
                "selected_rows": rerank_stats["rows"],
                "hit_rate": rerank_stats["hit_rate"],
                "avg_max_up_5d": rerank_stats["avg_max_up_5d"],
                **rerank_portfolio,
            }
        )

        details[f"top{top_n}"] = {
            "original": {
                "hit_rate": original_stats["hit_rate"],
                "avg_max_up_5d": original_stats["avg_max_up_5d"],
                "portfolio": original_portfolio,
            },
            "rerank": {
                "hit_rate": rerank_stats["hit_rate"],
                "avg_max_up_5d": rerank_stats["avg_max_up_5d"],
                "portfolio": rerank_portfolio,
            },
        }

    summary_df = pd.DataFrame(rows)
    summary_csv = output_dir / "candidate_rerank_summary.csv"
    summary_json = output_dir / "candidate_rerank_summary.json"
    summary_md = output_dir / "candidate_rerank_summary.md"
    summary_df.to_csv(summary_csv, index=False)
    payload = {
        "train_range": [args.train_start, args.train_end],
        "test_range": [args.test_start, args.test_end],
        "candidate_top_n": int(args.candidate_top_n),
        "rebalance_every": int(args.rebalance_every),
        "rows": summary_df.to_dict(orient="records"),
        "details": details,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary_md.write_text(summary_df.to_markdown(index=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
