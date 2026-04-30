from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.analysis.research_report import extract_display_weights
from ashare_quant.analysis.audit_report import build_leakage_bias_checklist, build_strategy_change_log
from ashare_quant.backtest.engine import BacktestConfig, DailyBacktester
from ashare_quant.data.base import MarketDataSource
from ashare_quant.data.csv_adapter import CSVDataSource
from ashare_quant.labels.label_builder import label_column_name
from ashare_quant.models.dynamic_weighting import spearman_corr
from ashare_quant.pipeline import (
    DEFAULT_BACKTEST,
    DEFAULT_RESEARCH,
    load_json,
    prepare_research_frame,
    score_research_frame,
)
from ashare_quant.portfolio.construction import build_portfolio_targets
from ashare_quant.strategy_classifier import assessment_payload


DEFAULT_TOP_N_GRID = [8, 12, 15, 20, 30]
DEFAULT_COST_SCENARIOS = {
    "fixed_base": {},
    "fixed_stress": {"use_liquidity_aware_cost": False, "slippage": 0.0010},
    "liquidity_aware": {"use_liquidity_aware_cost": True},
}


@dataclass(frozen=True)
class ExperimentScenario:
    name: str
    description: str
    research_overrides: dict[str, Any] = field(default_factory=dict)
    backtest_overrides: dict[str, Any] = field(default_factory=dict)
    search_top_n: bool = False
    top_n_grid: list[int] = field(default_factory=lambda: DEFAULT_TOP_N_GRID.copy())
    weighting_methods: list[str] = field(default_factory=list)
    sleeve_grid: list[int] = field(default_factory=list)
    cost_scenarios: dict[str, dict[str, Any]] = field(default_factory=dict)


def build_default_ablation_scenarios() -> list[ExperimentScenario]:
    return [
        ExperimentScenario(
            name="baseline",
            description="Old baseline: raw 5d label, static mean-IC weights, same-day close execution assumption, equal-weight Top 8.",
            research_overrides={
                "label_horizons": [5],
                "label_type": "raw",
                "ranker_type": "static",
                "factor_combination_mode": "direct",
                "signal_time": "close",
                "execution_price": "close",
                "execution_lag": 0,
                "holding_window": 5,
                "top_n": 8,
                "rebalance_every": 5,
                "sleeve_count": 1,
                "weighting_method": "equal",
                "weight_change_threshold": 0.0,
                "rank_change_threshold": 0,
            },
            backtest_overrides={"use_liquidity_aware_cost": False},
        ),
        ExperimentScenario(
            name="baseline_aligned",
            description="Aligned close-to-close execution with t+1 close entry and 5-day hold.",
            research_overrides={
                "label_horizons": [5],
                "label_type": "raw",
                "ranker_type": "static",
                "factor_combination_mode": "direct",
                "signal_time": "close",
                "execution_price": "close",
                "execution_lag": 1,
                "holding_window": 5,
                "top_n": 8,
                "rebalance_every": 5,
                "sleeve_count": 1,
                "weighting_method": "equal",
            },
            backtest_overrides={"use_liquidity_aware_cost": False},
        ),
        ExperimentScenario(
            name="alpha_label",
            description="Aligned baseline plus neutralized-residual alpha label.",
            research_overrides={
                "label_horizons": [5],
                "label_type": "neutralized_residual",
                "ranker_type": "static",
                "factor_combination_mode": "direct",
                "execution_lag": 1,
                "execution_price": "close",
                "holding_window": 5,
            },
            backtest_overrides={"use_liquidity_aware_cost": False},
        ),
        ExperimentScenario(
            name="dynamic_weight",
            description="Aligned alpha-label pipeline with rolling ICIR dynamic factor weights.",
            research_overrides={
                "label_horizons": [5],
                "label_type": "neutralized_residual",
                "ranker_type": "dynamic_icir",
                "factor_combination_mode": "direct",
                "execution_lag": 1,
                "execution_price": "close",
                "holding_window": 5,
            },
            backtest_overrides={"use_liquidity_aware_cost": False},
        ),
        ExperimentScenario(
            name="factor_group",
            description="Dynamic weighting on factor-group scores instead of direct per-factor weights.",
            research_overrides={
                "label_horizons": [5],
                "label_type": "neutralized_residual",
                "ranker_type": "dynamic_icir",
                "factor_combination_mode": "group",
                "execution_lag": 1,
                "execution_price": "close",
                "holding_window": 5,
            },
            backtest_overrides={"use_liquidity_aware_cost": False},
        ),
        ExperimentScenario(
            name="multi_horizon",
            description="Factor-group model with 3d/5d/10d labels and fixed horizon mixing.",
            research_overrides={
                "label_horizons": [3, 5, 10],
                "horizon_score_weights": {"3": 0.3, "5": 0.4, "10": 0.3},
                "label_type": "neutralized_residual",
                "ranker_type": "dynamic_icir",
                "factor_combination_mode": "group",
                "execution_lag": 1,
                "execution_price": "close",
                "holding_window": 5,
            },
            backtest_overrides={"use_liquidity_aware_cost": False},
        ),
        ExperimentScenario(
            name="new_construction",
            description="Multi-horizon model with 5-sleeve rotation, score-based weighting, no-trade band, and Top-N search.",
            research_overrides={
                "label_horizons": [3, 5, 10],
                "horizon_score_weights": {"3": 0.3, "5": 0.4, "10": 0.3},
                "label_type": "neutralized_residual",
                "ranker_type": "dynamic_icir",
                "factor_combination_mode": "group",
                "execution_lag": 1,
                "execution_price": "close",
                "holding_window": 5,
                "sleeve_count": 5,
                "weighting_method": "rank",
                "weight_change_threshold": 0.01,
                "rank_change_threshold": 2,
                "max_weight": 0.12,
                "industry_cap": 0.25,
                "min_holdings": 8,
            },
            backtest_overrides={"use_liquidity_aware_cost": False},
            search_top_n=True,
            top_n_grid=DEFAULT_TOP_N_GRID.copy(),
            weighting_methods=["equal", "rank", "softmax"],
            sleeve_grid=[1, 5],
        ),
        ExperimentScenario(
            name="new_cost_model",
            description="New construction pipeline plus liquidity-aware trading cost model.",
            research_overrides={
                "label_horizons": [3, 5, 10],
                "horizon_score_weights": {"3": 0.3, "5": 0.4, "10": 0.3},
                "label_type": "neutralized_residual",
                "ranker_type": "dynamic_icir",
                "factor_combination_mode": "group",
                "execution_lag": 1,
                "execution_price": "close",
                "holding_window": 5,
                "sleeve_count": 5,
                "weighting_method": "rank",
                "weight_change_threshold": 0.01,
                "rank_change_threshold": 2,
                "max_weight": 0.12,
                "industry_cap": 0.25,
                "min_holdings": 8,
            },
            backtest_overrides={"use_liquidity_aware_cost": True},
            search_top_n=True,
            top_n_grid=DEFAULT_TOP_N_GRID.copy(),
            weighting_methods=["equal", "rank", "softmax"],
            sleeve_grid=[1, 5],
            cost_scenarios=DEFAULT_COST_SCENARIOS.copy(),
        ),
    ]


def _primary_label_col(research_cfg: dict[str, Any], metadata: dict[str, Any]) -> str:
    return label_column_name(int(metadata["label_horizons"][0]), str(research_cfg["label_type"]))


def _train_test_split_dates(dates: list[pd.Timestamp], train_ratio: float = 0.6) -> tuple[set[pd.Timestamp], list[pd.Timestamp]]:
    split = max(int(len(dates) * train_ratio), 1)
    train_dates = set(dates[:split])
    test_dates = [pd.Timestamp(date) for date in dates[split:]]
    return train_dates, test_dates


def _assign_score_buckets(df: pd.DataFrame, score_col: str = "score", buckets: int = 10) -> pd.DataFrame:
    out = df.copy()
    out["score_bucket"] = np.nan
    for _, idx in out.groupby("date").groups.items():
        sl = out.loc[idx]
        valid = sl[score_col].notna()
        if valid.sum() == 0:
            continue
        rank_pct = sl.loc[valid, score_col].rank(method="first", pct=True)
        bucket = np.ceil(rank_pct * buckets).clip(1, buckets)
        out.loc[sl.index[valid], "score_bucket"] = bucket.astype(int)
    return out


def _compute_ic_series(df: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, sl in df.groupby("date"):
        data = sl[[score_col, label_col]].dropna()
        ic = spearman_corr(data[score_col], data[label_col]) if len(data) >= 5 else np.nan
        if pd.notna(ic):
            rows.append({"date": pd.Timestamp(date), "ic": float(ic)})
    return pd.DataFrame(rows)


def _summarize_ic(ic_df: pd.DataFrame) -> dict[str, float]:
    if ic_df.empty:
        return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0}
    ic_mean = float(ic_df["ic"].mean())
    ic_std = float(ic_df["ic"].std(ddof=0)) if len(ic_df) > 1 else 0.0
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": float(ic_mean / ic_std) if ic_std > 1e-12 else 0.0,
    }


def _compute_quantile_tables(df: pd.DataFrame, label_col: str, score_col: str = "score") -> tuple[pd.DataFrame, pd.DataFrame]:
    bucketed = _assign_score_buckets(df, score_col=score_col, buckets=10)
    daily = (
        bucketed.dropna(subset=["score_bucket", label_col])
        .groupby(["date", "score_bucket"], as_index=False)[label_col]
        .mean()
        .rename(columns={label_col: "forward_return"})
    )
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = (
        daily.groupby("score_bucket", as_index=False)["forward_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_forward_return", "std": "std_forward_return"})
    )
    pivot = daily.pivot(index="date", columns="score_bucket", values="forward_return").sort_index()
    long_short = pivot.get(10, pd.Series(0.0, index=pivot.index)) - pivot.get(1, pd.Series(0.0, index=pivot.index))
    curve = pd.DataFrame(
        {
            "date": pivot.index,
            "top_bucket_return": pivot.get(10, pd.Series(0.0, index=pivot.index)).fillna(0.0).values,
            "bottom_bucket_return": pivot.get(1, pd.Series(0.0, index=pivot.index)).fillna(0.0).values,
            "long_short_return": long_short.fillna(0.0).values,
        }
    )
    curve["top_bucket_equity"] = (1.0 + curve["top_bucket_return"]).cumprod()
    curve["bottom_bucket_equity"] = (1.0 + curve["bottom_bucket_return"]).cumprod()
    curve["long_short_equity"] = (1.0 + curve["long_short_return"]).cumprod()
    return summary, curve


def _merge_position_labels(test_df: pd.DataFrame, targets: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame()
    enriched = test_df.copy()
    enriched["size_bucket"] = pd.Series("Unknown", index=enriched.index, dtype="object")
    for _, idx in enriched.groupby("date").groups.items():
        sl = enriched.loc[idx]
        valid = sl["market_cap"].notna()
        if valid.sum() < 3:
            enriched.loc[sl.index[valid], "size_bucket"] = "Unknown"
            continue
        rank_pct = sl.loc[valid, "market_cap"].rank(method="first", pct=True)
        bucket = pd.cut(
            rank_pct,
            bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=["Small", "Mid", "Large"],
            include_lowest=True,
        )
        enriched.loc[sl.index[valid], "size_bucket"] = bucket.astype(str)
    merged = targets.merge(
        enriched[["date", "code", "industry", "market_cap", "size_bucket", label_col]],
        left_on=["signal_date", "code"],
        right_on=["date", "code"],
        how="left",
    )
    merged = merged.drop(columns=["date_y"], errors="ignore").rename(columns={"date_x": "date"})
    return merged


def _weighted_group_summary(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for group, sl in df.groupby(group_col):
        weights = pd.to_numeric(sl["target_weight"], errors="coerce").fillna(0.0)
        values = pd.to_numeric(sl[value_col], errors="coerce")
        mask = weights.gt(0) & values.notna()
        if not mask.any():
            weighted_value = np.nan
        else:
            weighted_value = float(np.average(values.loc[mask], weights=weights.loc[mask]))
        rows.append(
            {
                group_col: group,
                "positions": int(len(sl)),
                "selected_dates": int(sl["signal_date"].nunique()) if "signal_date" in sl.columns else 0,
                "avg_weight": float(weights.mean()),
                "weighted_label_return": weighted_value,
                "avg_score": float(pd.to_numeric(sl["score"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_label_return", ascending=False).reset_index(drop=True)


def _market_state_table(test_df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
    close_col = "research_close" if "research_close" in test_df.columns else "close"
    market = test_df.sort_values(["code", "date"]).copy()
    market["ret_1"] = market.groupby("code")[close_col].pct_change()
    daily_market = market.groupby("date", as_index=False)["ret_1"].mean().rename(columns={"ret_1": "market_return"})
    if daily_market.empty:
        return pd.DataFrame()
    daily_market["trend_20"] = daily_market["market_return"].rolling(20, min_periods=5).mean()
    daily_market["vol_20"] = daily_market["market_return"].rolling(20, min_periods=5).std()
    vol_threshold = float(daily_market["vol_20"].median()) if daily_market["vol_20"].notna().any() else 0.0
    daily_market["market_state"] = np.where(
        daily_market["trend_20"].fillna(0.0) >= 0.0,
        "up",
        "down",
    ) + "_" + np.where(daily_market["vol_20"].fillna(0.0) >= vol_threshold, "highvol", "lowvol")
    merged = result.merge(daily_market[["date", "market_state"]], on="date", how="left")
    if merged.empty:
        return pd.DataFrame()
    rows = []
    for state, sl in merged.groupby("market_state"):
        returns = pd.to_numeric(sl["net_return"], errors="coerce").fillna(0.0)
        ann_vol = float(returns.std(ddof=0) * np.sqrt(252)) if len(returns) > 1 else 0.0
        ann_ret = float((1.0 + returns).prod() ** (252 / max(len(returns), 1)) - 1.0) if len(returns) else 0.0
        rows.append(
            {
                "market_state": state,
                "days": int(len(sl)),
                "annual_return": ann_ret,
                "annual_volatility": ann_vol,
                "sharpe": float(ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0,
                "hit_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
                "avg_turnover": float(pd.to_numeric(sl["turnover"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def _cost_sensitivity_table(
    test_df: pd.DataFrame,
    targets: pd.DataFrame,
    research_cfg: dict[str, Any],
    backtest_cfg: dict[str, Any],
    scenarios: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for name, overrides in scenarios.items():
        cfg = {
            **backtest_cfg,
            **overrides,
            "signal_time": str(research_cfg["signal_time"]),
            "execution_price": str(research_cfg["execution_price"]),
            "execution_lag": int(research_cfg["execution_lag"]),
            "holding_window": int(research_cfg["holding_window"]),
            "sleeve_count": int(research_cfg["sleeve_count"]),
        }
        _, metrics = DailyBacktester(config=BacktestConfig(**cfg)).run(test_df, targets)
        rows.append({"cost_scenario": name, **metrics})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def _evaluate_construction_grid(
    test_df: pd.DataFrame,
    research_cfg: dict[str, Any],
    backtest_cfg: dict[str, Any],
    top_n_grid: list[int],
    weighting_methods: list[str],
    sleeve_grid: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    methods = weighting_methods or [str(research_cfg["weighting_method"])]
    sleeves = sleeve_grid or [int(research_cfg["sleeve_count"])]
    for top_n in top_n_grid:
        for weighting_method in methods:
            for sleeve_count in sleeves:
                targets = build_portfolio_targets(
                    test_df,
                    score_col="score",
                    top_n=int(top_n),
                    rebalance_every=int(research_cfg["rebalance_every"]),
                    sleeve_count=int(sleeve_count),
                    execution_lag=int(research_cfg["execution_lag"]),
                    weighting_method=str(weighting_method),
                    max_weight=float(research_cfg["max_weight"]),
                    industry_cap=float(research_cfg["industry_cap"]),
                    min_holdings=int(research_cfg["min_holdings"]),
                    score_threshold=float(research_cfg["score_threshold"]),
                    softmax_temperature=float(research_cfg["softmax_temperature"]),
                    weight_change_threshold=float(research_cfg["weight_change_threshold"]),
                    rank_change_threshold=int(research_cfg["rank_change_threshold"]),
                    entry_score_advantage_threshold=float(research_cfg.get("entry_score_advantage_threshold", 0.0)),
                )
                metrics = DailyBacktester(
                    config=BacktestConfig(
                        **backtest_cfg,
                        signal_time=str(research_cfg["signal_time"]),
                        execution_price=str(research_cfg["execution_price"]),
                        execution_lag=int(research_cfg["execution_lag"]),
                        holding_window=int(research_cfg["holding_window"]),
                        sleeve_count=int(sleeve_count),
                    )
                ).run(test_df, targets)[1]
                rows.append(
                    {
                        "top_n": int(top_n),
                        "weighting_method": str(weighting_method),
                        "sleeve_count": int(sleeve_count),
                        **metrics,
                    }
                )
    grid_df = pd.DataFrame(rows)
    if grid_df.empty:
        return {}, grid_df
    best = grid_df.sort_values(
        ["sharpe", "annual_return", "avg_turnover"],
        ascending=[False, False, True],
    ).iloc[0]
    selected = {
        "top_n": int(best["top_n"]),
        "weighting_method": str(best["weighting_method"]),
        "sleeve_count": int(best["sleeve_count"]),
    }
    return selected, grid_df


def _serialize_detail_tables(detail_tables: dict[str, pd.DataFrame]) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for name, df in detail_tables.items():
        if df.empty:
            payload[name] = []
        else:
            payload[name] = json_ready_records(df)
    return payload


def json_ready_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


def _turnover_diagnostics_from_targets(targets: pd.DataFrame) -> pd.DataFrame:
    if targets is None or targets.empty or "trade_reason" not in targets.columns:
        return pd.DataFrame()
    cols = [
        col
        for col in [
            "signal_date",
            "execution_date",
            "sleeve",
            "code",
            "trade_reason",
            "target_weight",
            "score",
            "rank",
            "industry",
            "risk_state",
            "risk_multiplier",
        ]
        if col in targets.columns
    ]
    return targets.loc[targets["trade_reason"].astype(str) != "", cols].reset_index(drop=True)


def run_single_experiment(
    df: pd.DataFrame,
    research_cfg: dict[str, Any],
    backtest_cfg: dict[str, Any],
    scenario: ExperimentScenario,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    scenario_research = {**research_cfg, **scenario.research_overrides}
    scenario_backtest = {**backtest_cfg, **scenario.backtest_overrides}
    prepared_df, metadata = prepare_research_frame(df.copy(), scenario_research)

    dates = sorted(prepared_df["date"].drop_duplicates())
    train_dates, test_dates = _train_test_split_dates(dates)
    if not test_dates:
        raise ValueError(f"Scenario {scenario.name} has no test dates after the train/test split.")

    scored_df, score_details = score_research_frame(
        df=prepared_df,
        research_cfg=scenario_research,
        metadata=metadata,
        train_dates=train_dates,
    )
    test_df = scored_df.loc[scored_df["date"].isin(test_dates)].copy()

    selected_grid: dict[str, Any] = {}
    construction_grid_df = pd.DataFrame()
    if scenario.search_top_n:
        selected_grid, construction_grid_df = _evaluate_construction_grid(
            test_df=test_df,
            research_cfg=scenario_research,
            backtest_cfg=scenario_backtest,
            top_n_grid=scenario.top_n_grid,
            weighting_methods=scenario.weighting_methods,
            sleeve_grid=scenario.sleeve_grid,
        )
        scenario_research = {**scenario_research, **selected_grid}

    targets = build_portfolio_targets(
        test_df,
        score_col="score",
        top_n=int(scenario_research["top_n"]),
        rebalance_every=int(scenario_research["rebalance_every"]),
        sleeve_count=int(scenario_research["sleeve_count"]),
        execution_lag=int(scenario_research["execution_lag"]),
        weighting_method=str(scenario_research["weighting_method"]),
        max_weight=float(scenario_research["max_weight"]),
        industry_cap=float(scenario_research["industry_cap"]),
        min_holdings=int(scenario_research["min_holdings"]),
        score_threshold=float(scenario_research["score_threshold"]),
        softmax_temperature=float(scenario_research["softmax_temperature"]),
        weight_change_threshold=float(scenario_research["weight_change_threshold"]),
        rank_change_threshold=int(scenario_research["rank_change_threshold"]),
        entry_score_advantage_threshold=float(scenario_research.get("entry_score_advantage_threshold", 0.0)),
    )
    result, metrics = DailyBacktester(
        config=BacktestConfig(
            **scenario_backtest,
            signal_time=str(scenario_research["signal_time"]),
            execution_price=str(scenario_research["execution_price"]),
            execution_lag=int(scenario_research["execution_lag"]),
            holding_window=int(scenario_research["holding_window"]),
            sleeve_count=int(scenario_research["sleeve_count"]),
        )
    ).run(test_df, targets)

    label_col = _primary_label_col(scenario_research, metadata)
    ic_df = _compute_ic_series(test_df, score_col="score", label_col=label_col)
    ic_summary = _summarize_ic(ic_df)
    quantile_summary_df, quantile_curve_df = _compute_quantile_tables(test_df, label_col=label_col, score_col="score")
    position_labels_df = _merge_position_labels(test_df, targets, label_col=label_col)
    industry_df = _weighted_group_summary(position_labels_df, group_col="industry", value_col=label_col)
    size_df = _weighted_group_summary(position_labels_df, group_col="size_bucket", value_col=label_col)
    state_df = _market_state_table(test_df, result)
    cost_df = _cost_sensitivity_table(
        test_df,
        targets=targets,
        research_cfg=scenario_research,
        backtest_cfg=scenario_backtest,
        scenarios=scenario.cost_scenarios or {"current": {}},
    )
    latest_target_date = targets["signal_date"].max() if not targets.empty and "signal_date" in targets.columns else pd.NaT
    latest_targets = (
        targets.loc[targets["signal_date"] == latest_target_date].copy()
        if pd.notna(latest_target_date)
        else pd.DataFrame()
    )
    assessment = assessment_payload(metrics=metrics, picks=latest_targets)

    summary_row = {
        "scenario": scenario.name,
        "description": scenario.description,
        "label_type": scenario_research["label_type"],
        "label_horizons": ",".join(str(v) for v in scenario_research.get("label_horizons", [scenario_research["label_horizon"]])),
        "ranker_type": scenario_research["ranker_type"],
        "factor_combination_mode": scenario_research["factor_combination_mode"],
        "signal_time": scenario_research["signal_time"],
        "execution_time": f"t+{int(scenario_research['execution_lag'])} {scenario_research['execution_price']}",
        "execution_lag": int(scenario_research["execution_lag"]),
        "holding_window": int(scenario_research["holding_window"]),
        "top_n": int(scenario_research["top_n"]),
        "weighting_method": scenario_research["weighting_method"],
        "sleeve_count": int(scenario_research["sleeve_count"]),
        "use_liquidity_aware_cost": bool(scenario_backtest["use_liquidity_aware_cost"]),
        "train_dates": int(len(train_dates)),
        "test_dates": int(len(test_dates)),
        "tradeable_ratio": float(metadata["universe_summary"].tradeable_ratio),
        "unknown_industry_ratio": float((prepared_df["industry"] == "Unknown").mean()) if "industry" in prepared_df.columns else 0.0,
        **metrics,
        **ic_summary,
        "selected_grid": selected_grid,
        "display_weights": extract_display_weights(score_details),
        "strategy_classification": assessment["classification"],
        "analyst_view": assessment["analyst_view"],
        "trader_view": assessment["trader_view"],
        "exposure_warning": assessment["exposure_warning"],
    }
    detail_tables = {
        "equity_curve": result,
        "targets": targets,
        "latest_strategy_targets": latest_targets,
        "ic_series": ic_df,
        "quantile_summary": quantile_summary_df,
        "quantile_curve": quantile_curve_df,
        "industry_performance": industry_df,
        "size_performance": size_df,
        "market_state_performance": state_df,
        "construction_grid": construction_grid_df,
        "cost_sensitivity": cost_df,
        "turnover_diagnostics": _turnover_diagnostics_from_targets(targets),
    }
    if not construction_grid_df.empty:
        summary_row["grid_best_sharpe"] = float(construction_grid_df["sharpe"].max())
    return summary_row, detail_tables


def run_ablation_suite(
    data_path: str | Path,
    research_config_path: str | Path,
    backtest_config_path: str | Path,
    output_dir: str | Path,
    data_source: MarketDataSource | None = None,
    data_adjust: str = "none",
    scenarios: list[ExperimentScenario] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    research_cfg = {**DEFAULT_RESEARCH, **load_json(research_config_path)}
    backtest_cfg = {**DEFAULT_BACKTEST, **load_json(backtest_config_path)}
    scenarios = scenarios or build_default_ablation_scenarios()

    if data_source is not None:
        df = data_source.load_daily_bars()
    else:
        df = CSVDataSource(data_path, adjust=data_adjust).load()

    summary_rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, pd.DataFrame]] = {}
    for scenario in scenarios:
        summary_row, detail_tables = run_single_experiment(
            df=df,
            research_cfg=research_cfg,
            backtest_cfg=backtest_cfg,
            scenario=scenario,
        )
        summary_rows.append(summary_row)
        details[scenario.name] = detail_tables

    summary_df = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    write_ablation_outputs(summary_df=summary_df, detail_tables=details, output_dir=output_dir)
    return summary_df, details


def build_ablation_summary_markdown(
    summary_df: pd.DataFrame,
    *,
    change_log_df: pd.DataFrame | None = None,
    leakage_check_df: pd.DataFrame | None = None,
) -> str:
    if summary_df.empty:
        return "# Ablation Summary\n\n_No results._\n"

    lines = [
        "# Ablation Summary",
        "",
        "| scenario | gross_return | net_return | sharpe | max_drawdown | avg_turnover | cost_drag | classification | top_n | weighting | sleeves | cost_model |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            "| {scenario} | {gross_annual_return:.4f} | {annual_return:.4f} | {sharpe:.4f} | {max_drawdown:.4f} | {avg_turnover:.4f} | {cost_drag:.4f} | {classification} | {top_n} | {weighting_method} | {sleeve_count} | {cost_model} |".format(
                scenario=row.scenario,
                gross_annual_return=float(getattr(row, "gross_annual_return", 0.0)),
                annual_return=float(row.annual_return),
                sharpe=float(row.sharpe),
                max_drawdown=float(row.max_drawdown),
                avg_turnover=float(row.avg_turnover),
                cost_drag=float(getattr(row, "after_cost_return_drag", 0.0)),
                classification=getattr(row, "strategy_classification", ""),
                top_n=int(row.top_n),
                weighting_method=row.weighting_method,
                sleeve_count=int(row.sleeve_count),
                cost_model="liquidity_aware" if bool(row.use_liquidity_aware_cost) else "fixed",
            )
        )

    best = summary_df.sort_values("sharpe", ascending=False).iloc[0]
    worst_turnover = summary_df.sort_values("avg_turnover", ascending=False).iloc[0]
    conclusion = dedent(
        f"""
        ## Conclusion

        - Best after-cost Sharpe scenario: `{best['scenario']}` with Sharpe `{float(best['sharpe']):.4f}` and annual return `{float(best['annual_return']):.4f}`.
        - Highest-turnover scenario: `{worst_turnover['scenario']}` with avg_turnover `{float(worst_turnover['avg_turnover']):.4f}`.
        - Prefer scenarios that improve `sharpe` and `max_drawdown` together; higher gross return alone is not enough.
        """
    ).strip()

    sections = ["\n".join(lines), conclusion]
    if change_log_df is not None and not change_log_df.empty:
        sections.extend(
            [
                "## Strategy Change Log",
                "",
                change_log_df.to_markdown(index=False),
            ]
        )
    if leakage_check_df is not None and not leakage_check_df.empty:
        sections.extend(
            [
                "",
                "## Leakage And Bias Checklist",
                "",
                leakage_check_df.to_markdown(index=False),
            ]
        )
    return "\n\n".join(sections) + "\n"


def write_ablation_outputs(
    summary_df: pd.DataFrame,
    detail_tables: dict[str, dict[str, pd.DataFrame]],
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    change_log_df = build_strategy_change_log(summary_df)
    leakage_check_df = build_leakage_bias_checklist(summary_df)
    summary_df.drop(columns=["display_weights"], errors="ignore").to_csv(output_path / "ablation_summary.csv", index=False)
    change_log_df.to_csv(output_path / "strategy_change_log.csv", index=False)
    leakage_check_df.to_csv(output_path / "leakage_bias_checklist.csv", index=False)
    (output_path / "ablation_summary.md").write_text(
        build_ablation_summary_markdown(
            summary_df,
            change_log_df=change_log_df,
            leakage_check_df=leakage_check_df,
        ),
        encoding="utf-8",
    )

    for scenario, tables in detail_tables.items():
        scenario_dir = output_path / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(scenario_dir / f"{name}.csv", index=False)

    json_summary = summary_df.copy()
    if "display_weights" in json_summary.columns:
        json_summary["display_weights"] = json_summary["display_weights"].apply(lambda value: value if isinstance(value, dict) else {})
    (output_path / "ablation_summary.json").write_text(
        json.dumps(json_ready_records(json_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "strategy_change_log.json").write_text(
        json.dumps(json_ready_records(change_log_df), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "leakage_bias_checklist.json").write_text(
        json.dumps(json_ready_records(leakage_check_df), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
