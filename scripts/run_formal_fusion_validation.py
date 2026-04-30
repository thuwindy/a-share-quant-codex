from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.backtest.engine import BacktestConfig, DailyBacktester
from ashare_quant.data.csv_adapter import CSVDataSource, ensure_price_views
from ashare_quant.data.research_slice import build_research_slice
from ashare_quant.pipeline import (
    DEFAULT_BACKTEST,
    DEFAULT_RESEARCH,
    load_json,
    prepare_research_frame,
    score_research_frame,
)
from ashare_quant.portfolio.construction import build_portfolio_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run formal baseline/main/steady ML-fusion validation.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--slice-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_production_managed_15bps.json"))
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--main-config", required=True)
    parser.add_argument("--steady-config", required=True)
    parser.add_argument("--ml-model-path", required=True)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-04-08")
    parser.add_argument("--train-start", default="2020-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2026-04-08")
    parser.add_argument("--max-codes", type=int, default=500)
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    return parser


def _scenario_config(path: str | Path, *, ml_model_path: str | None) -> dict:
    cfg = {**DEFAULT_RESEARCH, **load_json(path)}
    if ml_model_path and bool(cfg.get("use_ml_score", False)):
        cfg["ml_model_path"] = str(ml_model_path)
    return cfg


def _run_one(df: pd.DataFrame, *, research_cfg: dict, backtest_cfg: dict, train_start: str, train_end: str, test_start: str, test_end: str) -> dict:
    prepared, metadata = prepare_research_frame(df.copy(), research_cfg)
    train_mask = (prepared["date"] >= pd.Timestamp(train_start)) & (prepared["date"] <= pd.Timestamp(train_end))
    train_dates = set(pd.to_datetime(prepared.loc[train_mask, "date"]).drop_duplicates())
    scored, score_details = score_research_frame(
        df=prepared,
        research_cfg=research_cfg,
        metadata=metadata,
        train_dates=train_dates,
    )
    test_mask = (scored["date"] >= pd.Timestamp(test_start)) & (scored["date"] <= pd.Timestamp(test_end))
    test_df = scored.loc[test_mask].copy()
    targets = build_portfolio_targets(
        test_df,
        score_col="score",
        top_n=int(research_cfg["top_n"]),
        rebalance_every=int(research_cfg["rebalance_every"]),
        sleeve_count=int(research_cfg["sleeve_count"]),
        execution_lag=int(research_cfg["execution_lag"]),
        weighting_method=str(research_cfg["weighting_method"]),
        max_weight=float(research_cfg["max_weight"]),
        industry_cap=float(research_cfg["industry_cap"]),
        min_holdings=int(research_cfg["min_holdings"]),
        score_threshold=float(research_cfg["score_threshold"]),
        softmax_temperature=float(research_cfg["softmax_temperature"]),
        weight_change_threshold=float(research_cfg["weight_change_threshold"]),
        rank_change_threshold=int(research_cfg["rank_change_threshold"]),
        entry_score_advantage_threshold=float(research_cfg.get("entry_score_advantage_threshold", 0.0)),
        entry_filter_overhead_enabled=bool(research_cfg.get("entry_filter_overhead_enabled", False)),
        entry_filter_min_overhead_resistance=float(research_cfg.get("entry_filter_min_overhead_resistance", float("-inf"))),
        entry_filter_score_floor=float(research_cfg.get("entry_filter_score_floor", -9999.0)),
    )
    result, metrics = DailyBacktester(
        config=BacktestConfig(
            **backtest_cfg,
            signal_time=str(research_cfg["signal_time"]),
            execution_price=str(research_cfg["execution_price"]),
            execution_lag=int(research_cfg["execution_lag"]),
            holding_window=int(research_cfg["holding_window"]),
            sleeve_count=int(research_cfg["sleeve_count"]),
            stop_loss_pct=float(research_cfg.get("stop_loss_pct", 0.0)),
            take_profit_pct=float(research_cfg.get("take_profit_pct", 0.0)),
            trailing_stop_pct=float(research_cfg.get("trailing_stop_pct", 0.0)),
        )
    ).run(test_df, targets)
    latest_signal_date = str(pd.Timestamp(targets["signal_date"].max()).date()) if not targets.empty else ""
    return {
        "rows": int(len(test_df)),
        "signal_dates": int(test_df["date"].nunique()),
        "target_rows": int(len(targets)),
        "latest_signal_date": latest_signal_date,
        "annual_return": float(metrics.get("annual_return", 0.0)),
        "annual_volatility": float(metrics.get("annual_volatility", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "hit_rate": float(metrics.get("hit_rate", 0.0)),
        "payoff_ratio": float(metrics.get("payoff_ratio", 0.0)),
        "avg_turnover": float(metrics.get("avg_turnover", 0.0)),
        "gross_annual_return": float(metrics.get("gross_annual_return", 0.0)),
        "after_cost_return_drag": float(metrics.get("after_cost_return_drag", 0.0)),
        "use_ml_score": bool(research_cfg.get("use_ml_score", False)),
        "original_score_weight": float(research_cfg.get("original_score_weight", 1.0)),
        "ml_score_weight": float(research_cfg.get("ml_score_weight", 0.0)),
        "ml_model_path": str(research_cfg.get("ml_model_path", "")),
        "score_details": score_details,
    }


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = build_research_slice(
        input_path=args.input_path,
        output_path=args.slice_path,
        start_date=args.start_date,
        end_date=args.end_date,
        include_suffixes=("SH", "SZ"),
        max_codes=int(args.max_codes),
    )
    df = ensure_price_views(CSVDataSource(args.slice_path, adjust=args.adjust).load_daily_bars())
    backtest_cfg = {**DEFAULT_BACKTEST, **load_json(args.backtest_config)}

    scenarios = [
        ("baseline", _scenario_config(args.baseline_config, ml_model_path=None)),
        ("main", _scenario_config(args.main_config, ml_model_path=args.ml_model_path)),
        ("steady", _scenario_config(args.steady_config, ml_model_path=args.ml_model_path)),
    ]

    rows = []
    details = {}
    for name, cfg in scenarios:
        print(f"[formal] run scenario={name}", flush=True)
        summary = _run_one(
            df,
            research_cfg=cfg,
            backtest_cfg=backtest_cfg,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        rows.append({"scenario": name, **{k: v for k, v in summary.items() if k != "score_details"}})
        details[name] = summary

    summary_df = pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).reset_index(drop=True)
    summary_csv = output_dir / "formal_fusion_validation_summary.csv"
    summary_json = output_dir / "formal_fusion_validation_summary.json"
    summary_md = output_dir / "formal_fusion_validation_summary.md"
    summary_df.to_csv(summary_csv, index=False)
    payload = {
        "slice_profile": profile.__dict__,
        "split": {
            "train": [args.train_start, args.train_end],
            "test": [args.test_start, args.test_end],
        },
        "rows": rows,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary_md.write_text(summary_df.to_markdown(index=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
