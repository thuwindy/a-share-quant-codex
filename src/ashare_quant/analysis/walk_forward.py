from __future__ import annotations

import math
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.backtest.engine import BacktestConfig, DailyBacktester
from ashare_quant.labels.label_builder import label_column_name
from ashare_quant.pipeline import (
    DEFAULT_BACKTEST,
    DEFAULT_RESEARCH,
    load_json,
    prepare_research_frame,
    score_research_frame,
)
from ashare_quant.portfolio.construction import build_portfolio_targets
from ashare_quant.data.base import MarketDataSource
from ashare_quant.data.csv_adapter import CSVDataSource


def _spearman_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 5:
        return math.nan
    rank_a = a.rank(pct=True)
    rank_b = b.rank(pct=True)
    if rank_a.nunique(dropna=True) < 2 or rank_b.nunique(dropna=True) < 2:
        return math.nan
    return rank_a.corr(rank_b)


def build_walk_forward_splits(
    dates: pd.Series | list[pd.Timestamp],
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
) -> list[dict[str, pd.Timestamp]]:
    unique_dates = pd.Index(sorted(pd.to_datetime(pd.Series(dates).dropna().unique())))
    if unique_dates.empty:
        return []

    cursor = unique_dates.min()
    last_date = unique_dates.max()
    splits: list[dict[str, pd.Timestamp]] = []

    while cursor <= last_date:
        train_end_target = cursor + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_end_target = train_end_target + pd.DateOffset(years=test_years)

        train_dates = unique_dates[(unique_dates >= cursor) & (unique_dates <= train_end_target)]
        test_dates = unique_dates[(unique_dates > train_end_target) & (unique_dates <= test_end_target)]
        if len(train_dates) == 0 or len(test_dates) == 0:
            break

        splits.append(
            {
                "train_start": pd.Timestamp(train_dates.min()),
                "train_end": pd.Timestamp(train_dates.max()),
                "test_start": pd.Timestamp(test_dates.min()),
                "test_end": pd.Timestamp(test_dates.max()),
            }
        )
        cursor = cursor + pd.DateOffset(years=step_years)

    return splits


def _compute_factor_ic_frame(
    df: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, sl in df.groupby("date"):
        for factor in factor_cols:
            data = sl[[factor, label_col]].dropna()
            if len(data) < 5:
                continue
            ic = _spearman_corr(data[factor], data[label_col])
            if pd.notna(ic):
                rows.append({"date": pd.Timestamp(date), "factor": factor, "ic": float(ic)})
    return pd.DataFrame(rows)


def _aggregate_factor_stability(factor_fold_df: pd.DataFrame) -> pd.DataFrame:
    if factor_fold_df.empty:
        return pd.DataFrame(
            columns=[
                "factor",
                "folds",
                "mean_test_ic",
                "std_test_ic",
                "positive_fold_ratio",
                "mean_weight",
                "weight_std",
                "sign_consistency",
                "stability_score",
            ]
        )

    grouped = factor_fold_df.groupby("factor", as_index=False)
    rows = []
    for factor, sl in grouped:
        mean_ic = float(sl["mean_test_ic"].mean())
        std_ic = float(sl["mean_test_ic"].std(ddof=0)) if len(sl) > 1 else 0.0
        mean_weight = float(sl["weight"].mean())
        weight_std = float(sl["weight"].std(ddof=0)) if len(sl) > 1 else 0.0
        sign_reference = np.sign(mean_weight) if mean_weight != 0 else np.sign(mean_ic)
        sign_series = np.sign(sl["weight"].replace(0.0, np.nan).fillna(sl["mean_test_ic"]))
        sign_consistency = float((sign_series == sign_reference).mean()) if sign_reference != 0 else 0.0
        positive_fold_ratio = float((sl["mean_test_ic"] > 0).mean())
        stability_score = float(abs(mean_ic) * positive_fold_ratio / (std_ic + 1e-6))
        rows.append(
            {
                "factor": factor,
                "folds": int(len(sl)),
                "mean_test_ic": mean_ic,
                "std_test_ic": std_ic,
                "positive_fold_ratio": positive_fold_ratio,
                "mean_weight": mean_weight,
                "weight_std": weight_std,
                "sign_consistency": sign_consistency,
                "stability_score": stability_score,
            }
        )
    return pd.DataFrame(rows).sort_values(["stability_score", "mean_test_ic"], ascending=False).reset_index(drop=True)


def run_walk_forward_analysis(
    data_path: str | Path,
    research_config_path: str | Path,
    backtest_config_path: str | Path,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    data_source: MarketDataSource | None = None,
    data_adjust: str = "none",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    research_cfg = {**DEFAULT_RESEARCH, **load_json(research_config_path)}
    backtest_cfg = {**DEFAULT_BACKTEST, **load_json(backtest_config_path)}

    if data_source is not None:
        df = data_source.load_daily_bars()
    else:
        df = CSVDataSource(data_path, adjust=data_adjust).load()
    df, metadata = prepare_research_frame(df, research_cfg)
    primary_horizon = metadata["label_horizons"][0]
    label_col = label_column_name(primary_horizon, str(research_cfg["label_type"]))
    primary_feature_cols = metadata["horizon_feature_map"][primary_horizon]
    splits = build_walk_forward_splits(
        dates=df["date"].drop_duplicates(),
        train_years=train_years,
        test_years=test_years,
        step_years=step_years,
    )
    if not splits:
        raise ValueError("No walk-forward folds could be built from the requested date range.")

    fold_rows: list[dict[str, Any]] = []
    factor_fold_rows: list[dict[str, Any]] = []
    backtester = DailyBacktester(config=BacktestConfig(**backtest_cfg))

    for fold_idx, fold in enumerate(splits, start=1):
        fold_df = df.loc[(df["date"] >= fold["train_start"]) & (df["date"] <= fold["test_end"])].copy()
        train_mask = (fold_df["date"] >= fold["train_start"]) & (fold_df["date"] <= fold["train_end"]) & fold_df[label_col].notna()
        test_mask = (fold_df["date"] >= fold["test_start"]) & (fold_df["date"] <= fold["test_end"])
        train_df = fold_df.loc[train_mask].copy()
        test_df = fold_df.loc[test_mask].copy()
        if train_df.empty or test_df.empty:
            continue

        train_dates = set(pd.to_datetime(train_df["date"]).drop_duplicates().tolist())
        scored_fold, score_details = score_research_frame(
            df=fold_df,
            research_cfg=research_cfg,
            metadata=metadata,
            train_dates=train_dates,
        )
        test_df = scored_fold.loc[test_mask].copy()

        targets = build_portfolio_targets(
            test_df,
            score_col="score",
            top_n=research_cfg["top_n"],
            rebalance_every=research_cfg["rebalance_every"],
            sleeve_count=research_cfg["sleeve_count"],
            execution_lag=research_cfg["execution_lag"],
            weighting_method=research_cfg["weighting_method"],
            max_weight=research_cfg["max_weight"],
            industry_cap=research_cfg["industry_cap"],
            min_holdings=research_cfg["min_holdings"],
            score_threshold=research_cfg["score_threshold"],
            softmax_temperature=research_cfg["softmax_temperature"],
            weight_change_threshold=research_cfg["weight_change_threshold"],
            rank_change_threshold=research_cfg["rank_change_threshold"],
        )
        result, metrics = DailyBacktester(
            config=BacktestConfig(
                **backtest_cfg,
                signal_time=research_cfg["signal_time"],
                execution_price=research_cfg["execution_price"],
                execution_lag=research_cfg["execution_lag"],
                holding_window=research_cfg["holding_window"],
                sleeve_count=research_cfg["sleeve_count"],
            )
        ).run(test_df, targets)
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_start": fold["train_start"].strftime("%Y-%m-%d"),
                "train_end": fold["train_end"].strftime("%Y-%m-%d"),
                "test_start": fold["test_start"].strftime("%Y-%m-%d"),
                "test_end": fold["test_end"].strftime("%Y-%m-%d"),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "test_dates": int(test_df["date"].nunique()),
                "selected_dates": int(targets["date"].nunique()) if not targets.empty else 0,
                **metrics,
            }
        )

        factor_ic = _compute_factor_ic_frame(test_df, factor_cols=primary_feature_cols, label_col=label_col)
        feature_weights = score_details.get("horizon_models", {}).get(str(primary_horizon), {}).get("weights", {})
        for factor in primary_feature_cols:
            ic_series = factor_ic.loc[factor_ic["factor"] == factor, "ic"] if not factor_ic.empty else pd.Series(dtype=float)
            factor_fold_rows.append(
                {
                    "fold": fold_idx,
                    "factor": factor,
                    "mean_test_ic": float(ic_series.mean()) if len(ic_series) else 0.0,
                    "weight": float(feature_weights.get(factor, 0.0)),
                }
            )

    folds_df = pd.DataFrame(fold_rows)
    factor_folds_df = pd.DataFrame(factor_fold_rows)
    factor_stats_df = _aggregate_factor_stability(factor_folds_df)

    if folds_df.empty:
        raise ValueError("Walk-forward produced no valid folds.")

    summary = {
        "data_path": str(data_path),
        "adjust_mode": data_adjust,
        "label_horizon": research_cfg["label_horizon"],
        "top_n": research_cfg["top_n"],
        "rebalance_every": research_cfg["rebalance_every"],
        "train_years": train_years,
        "test_years": test_years,
        "step_years": step_years,
        "folds": int(len(folds_df)),
        "mean_annual_return": float(folds_df["annual_return"].mean()),
        "median_annual_return": float(folds_df["annual_return"].median()),
        "mean_sharpe": float(folds_df["sharpe"].mean()),
        "worst_fold_drawdown": float(folds_df["max_drawdown"].min()),
        "mean_turnover": float(folds_df["avg_turnover"].mean()),
        "best_factor": factor_stats_df.iloc[0]["factor"] if not factor_stats_df.empty else "",
        "best_factor_mean_ic": float(factor_stats_df.iloc[0]["mean_test_ic"]) if not factor_stats_df.empty else 0.0,
    }
    return summary, folds_df, factor_stats_df


def build_walk_forward_summary_markdown(
    case_summaries: pd.DataFrame,
    factor_stats: pd.DataFrame,
) -> str:
    if case_summaries.empty:
        cases_table = "_No case summaries._"
    else:
        top_cases = case_summaries.sort_values(["mean_sharpe", "mean_annual_return"], ascending=False).head(10)
        case_rows = [
            "| universe | train_years | folds | mean_sharpe | mean_return | worst_mdd | mean_turnover |"
        ]
        case_rows.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in top_cases.itertuples(index=False):
            case_rows.append(
                "| {universe} | {train_years} | {folds} | {mean_sharpe:.4f} | {mean_return:.4f} | {worst_mdd:.4f} | {mean_turnover:.4f} |".format(
                    universe=getattr(row, "universe_tag", ""),
                    train_years=getattr(row, "train_years", ""),
                    folds=int(getattr(row, "folds", 0)),
                    mean_sharpe=float(getattr(row, "mean_sharpe", 0.0)),
                    mean_return=float(getattr(row, "mean_annual_return", 0.0)),
                    worst_mdd=float(getattr(row, "worst_fold_drawdown", 0.0)),
                    mean_turnover=float(getattr(row, "mean_turnover", 0.0)),
                )
            )
        cases_table = "\n".join(case_rows)

    if factor_stats.empty:
        factor_table = "_No factor stability rows._"
    else:
        top_factors = factor_stats.sort_values(["stability_score", "mean_test_ic"], ascending=False).head(10)
        factor_rows = [
            "| universe | train_years | factor | mean_test_ic | positive_fold_ratio | sign_consistency | stability_score |"
        ]
        factor_rows.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in top_factors.itertuples(index=False):
            factor_rows.append(
                "| {universe} | {train_years} | {factor} | {mean_test_ic:.4f} | {positive_ratio:.2%} | {sign_consistency:.2%} | {stability_score:.4f} |".format(
                    universe=getattr(row, "universe_tag", ""),
                    train_years=getattr(row, "train_years", ""),
                    factor=getattr(row, "factor", ""),
                    mean_test_ic=float(getattr(row, "mean_test_ic", 0.0)),
                    positive_ratio=float(getattr(row, "positive_fold_ratio", 0.0)),
                    sign_consistency=float(getattr(row, "sign_consistency", 0.0)),
                    stability_score=float(getattr(row, "stability_score", 0.0)),
                )
            )
        factor_table = "\n".join(factor_rows)

    return (
        dedent(
            f"""
            # Walk-Forward Comparison

            ## Top Cases

            {cases_table}

            ## Stable Factors

            {factor_table}

            ## Notes

            - Higher `mean_sharpe` and less negative `worst_mdd` are better, but stability across folds matters more than one good year.
            - `positive_fold_ratio` shows how often a factor had positive mean test IC across folds.
            - `sign_consistency` tells you whether the factor kept roughly the same directional role across folds.
            """
        ).strip()
        + "\n"
    )
