from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.backtest.engine import BacktestConfig, DailyBacktester
from ashare_quant.data.csv_adapter import CSVDataSource
from ashare_quant.pipeline import (
    DEFAULT_BACKTEST,
    DEFAULT_RESEARCH,
    load_json,
    prepare_research_frame,
    score_research_frame,
)
from ashare_quant.portfolio.construction import build_portfolio_targets
from ashare_quant.strategy_classifier import assessment_payload


def parse_int_grid(raw: str, *, fallback: list[int]) -> list[int]:
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    values: list[int] = []
    for token in tokens:
        value = int(token)
        if value not in values:
            values.append(value)
    return values or [int(v) for v in fallback]


def parse_float_grid(raw: str, *, fallback: list[float]) -> list[float]:
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    values: list[float] = []
    for token in tokens:
        value = float(token)
        if value not in values:
            values.append(value)
    return values or [float(v) for v in fallback]


def _train_test_split_dates(dates: list[pd.Timestamp], train_ratio: float = 0.6) -> tuple[set[pd.Timestamp], list[pd.Timestamp]]:
    split = max(int(len(dates) * train_ratio), 1)
    train_dates = {pd.Timestamp(date) for date in dates[:split]}
    test_dates = [pd.Timestamp(date) for date in dates[split:]]
    return train_dates, test_dates


def _latest_targets(targets: pd.DataFrame) -> pd.DataFrame:
    if targets.empty or "signal_date" not in targets.columns:
        return pd.DataFrame()
    latest_date = targets["signal_date"].max()
    if pd.isna(latest_date):
        return pd.DataFrame()
    return targets.loc[targets["signal_date"] == latest_date].copy()


def run_topn_notrade_quick_grid(
    *,
    data_path: str | Path,
    research_config_path: str | Path,
    backtest_config_path: str | Path,
    data_adjust: str = "qfq",
    top_n_grid: list[int] | None = None,
    weight_change_grid: list[float] | None = None,
    rank_change_grid: list[int] | None = None,
    sleeve_grid: list[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    research_cfg = {**DEFAULT_RESEARCH, **load_json(research_config_path)}
    backtest_cfg = {**DEFAULT_BACKTEST, **load_json(backtest_config_path)}

    top_n_grid = [int(v) for v in (top_n_grid or [int(research_cfg["top_n"])])]
    weight_change_grid = [float(v) for v in (weight_change_grid or [float(research_cfg["weight_change_threshold"])])]
    rank_change_grid = [int(v) for v in (rank_change_grid or [int(research_cfg["rank_change_threshold"])])]
    sleeve_grid = [int(v) for v in (sleeve_grid or [int(research_cfg["sleeve_count"])])]

    df = CSVDataSource(data_path, adjust=data_adjust).load()
    prepared_df, metadata = prepare_research_frame(df.copy(), research_cfg)
    dates = sorted(prepared_df["date"].drop_duplicates())
    full_train_dates, test_dates = _train_test_split_dates(dates)
    if not test_dates:
        raise ValueError("No test dates found after train/test split; cannot run quick grid.")

    scored_df, score_details = score_research_frame(
        df=prepared_df,
        research_cfg=research_cfg,
        metadata=metadata,
        train_dates=full_train_dates,
    )
    test_df = scored_df.loc[~scored_df["date"].isin(full_train_dates)].copy()

    rows: list[dict[str, Any]] = []
    total = len(top_n_grid) * len(weight_change_grid) * len(rank_change_grid) * len(sleeve_grid)
    idx = 0
    for top_n in top_n_grid:
        for weight_change_threshold in weight_change_grid:
            for rank_change_threshold in rank_change_grid:
                for sleeve_count in sleeve_grid:
                    idx += 1
                    print(
                        f"[quick_grid] run {idx}/{total}: top_n={top_n}, sleeve={sleeve_count}, weight_change={weight_change_threshold:.4f}, rank_change={rank_change_threshold}",
                        flush=True,
                    )
                    targets = build_portfolio_targets(
                        test_df,
                        score_col="score",
                        top_n=int(top_n),
                        rebalance_every=int(research_cfg["rebalance_every"]),
                        sleeve_count=int(sleeve_count),
                        execution_lag=int(research_cfg["execution_lag"]),
                        weighting_method=str(research_cfg["weighting_method"]),
                        max_weight=float(research_cfg["max_weight"]),
                        industry_cap=float(research_cfg["industry_cap"]),
                        min_holdings=int(research_cfg["min_holdings"]),
                        score_threshold=float(research_cfg["score_threshold"]),
                        softmax_temperature=float(research_cfg["softmax_temperature"]),
                        weight_change_threshold=float(weight_change_threshold),
                        rank_change_threshold=int(rank_change_threshold),
                        entry_score_advantage_threshold=float(research_cfg.get("entry_score_advantage_threshold", 0.0)),
                        entry_filter_overhead_enabled=bool(research_cfg.get("entry_filter_overhead_enabled", False)),
                        entry_filter_min_overhead_resistance=float(
                            research_cfg.get("entry_filter_min_overhead_resistance", float("-inf"))
                        ),
                        entry_filter_score_floor=float(research_cfg.get("entry_filter_score_floor", -9999.0)),
                    )
                    _, metrics = DailyBacktester(
                        config=BacktestConfig(
                            **backtest_cfg,
                            signal_time=str(research_cfg["signal_time"]),
                            execution_price=str(research_cfg["execution_price"]),
                            execution_lag=int(research_cfg["execution_lag"]),
                            holding_window=int(research_cfg["holding_window"]),
                            sleeve_count=int(sleeve_count),
                        )
                    ).run(test_df, targets)
                    assessment = assessment_payload(metrics=metrics, picks=_latest_targets(targets))
                    rows.append(
                        {
                            "top_n": int(top_n),
                            "sleeve_count": int(sleeve_count),
                            "weight_change_threshold": float(weight_change_threshold),
                            "rank_change_threshold": int(rank_change_threshold),
                            "annual_return": float(metrics["annual_return"]),
                            "gross_annual_return": float(metrics.get("gross_annual_return", 0.0)),
                            "sharpe": float(metrics["sharpe"]),
                            "max_drawdown": float(metrics["max_drawdown"]),
                            "avg_turnover": float(metrics["avg_turnover"]),
                            "after_cost_return_drag": float(metrics.get("after_cost_return_drag", 0.0)),
                            "strategy_classification": str(assessment["classification"]),
                        }
                    )

    summary_df = pd.DataFrame(rows).sort_values(
        ["sharpe", "after_cost_return_drag", "avg_turnover", "max_drawdown", "annual_return"],
        ascending=[False, True, True, False, False],
    ).reset_index(drop=True)
    if summary_df.empty:
        raise ValueError("Quick grid generated no rows.")

    best = summary_df.iloc[0].to_dict()
    payload = {
        "rows": int(len(summary_df)),
        "data_path": str(data_path),
        "research_config_path": str(research_config_path),
        "backtest_config_path": str(backtest_config_path),
        "data_adjust": str(data_adjust),
        "grid": {
            "top_n": [int(v) for v in top_n_grid],
            "weight_change_threshold": [float(v) for v in weight_change_grid],
            "rank_change_threshold": [int(v) for v in rank_change_grid],
            "sleeve_count": [int(v) for v in sleeve_grid],
        },
        "best": best,
        "recommended_overrides": {
            "top_n": int(best["top_n"]),
            "sleeve_count": int(best["sleeve_count"]),
            "weight_change_threshold": float(best["weight_change_threshold"]),
            "rank_change_threshold": int(best["rank_change_threshold"]),
        },
        "universe_filter_summary": metadata["universe_summary"].__dict__,
        "score_details": score_details,
    }
    return summary_df, payload
