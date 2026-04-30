from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.backtest.engine import BacktestConfig, DailyBacktester
from ashare_quant.data.csv_adapter import CSVDataSource
from ashare_quant.fundamental_veto import FundamentalVetoConfig, apply_fundamental_veto
from ashare_quant.pipeline import (
    DEFAULT_BACKTEST,
    DEFAULT_RESEARCH,
    load_json,
    prepare_research_frame,
    score_research_frame,
)
from ashare_quant.portfolio.construction import build_portfolio_targets
from ashare_quant.strategy_classifier import assessment_payload


@dataclass(frozen=True)
class FocusScenario:
    name: str
    description: str
    veto_enabled: bool = False
    veto_overrides: dict[str, Any] = field(default_factory=dict)


def _train_test_split_dates(dates: list[pd.Timestamp], train_ratio: float = 0.6) -> tuple[set[pd.Timestamp], list[pd.Timestamp]]:
    split = max(int(len(dates) * train_ratio), 1)
    train_dates = set(pd.Timestamp(date) for date in dates[:split])
    test_dates = [pd.Timestamp(date) for date in dates[split:]]
    return train_dates, test_dates


def _best_row(df: pd.DataFrame, prefer_tradable: bool = False) -> dict[str, Any]:
    if df.empty:
        return {}
    candidate_df = df.copy()
    if prefer_tradable and "strategy_classification" in candidate_df.columns:
        tradable_df = candidate_df.loc[candidate_df["strategy_classification"] == "tradable prototype"].copy()
        if not tradable_df.empty:
            candidate_df = tradable_df
    ordered = candidate_df.sort_values(
        ["sharpe", "after_cost_return_drag", "avg_turnover", "max_drawdown", "annual_return"],
        ascending=[False, True, True, False, False],
    ).reset_index(drop=True)
    return ordered.iloc[0].to_dict()


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _latest_targets(targets: pd.DataFrame) -> pd.DataFrame:
    if targets.empty or "signal_date" not in targets.columns:
        return pd.DataFrame()
    latest_date = targets["signal_date"].max()
    if pd.isna(latest_date):
        return pd.DataFrame()
    return targets.loc[targets["signal_date"] == latest_date].copy()


def _apply_veto_overlay(
    scored_df: pd.DataFrame,
    base_research_cfg: dict[str, Any],
    scenario: FocusScenario,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = scored_df.copy()
    out.attrs = {}
    out["fundamental_veto"] = False
    out["fundamental_veto_reason"] = ""

    if not scenario.veto_enabled:
        out["strategy_tradeable"] = out["monitor_eligible"].fillna(False).astype(bool)
        return out, {"enabled": False, "veto_ratio": 0.0, "reason_counts": {}}

    veto_cfg = FundamentalVetoConfig(
        enabled=True,
        min_profit_quality=float(scenario.veto_overrides.get("min_profit_quality", base_research_cfg.get("min_profit_quality", -0.10))),
        min_earnings_growth=float(scenario.veto_overrides.get("min_earnings_growth", base_research_cfg.get("min_earnings_growth", -0.15))),
        min_cashflow_quality=float(scenario.veto_overrides.get("min_cashflow_quality", base_research_cfg.get("min_cashflow_quality", -0.10))),
        min_roe=float(scenario.veto_overrides.get("min_roe", base_research_cfg.get("min_roe", -0.05))),
        max_debt_to_asset=float(scenario.veto_overrides.get("max_debt_to_asset", base_research_cfg.get("max_debt_to_asset", 0.75))),
    )
    out, summary = apply_fundamental_veto(out, config=veto_cfg)
    out["strategy_tradeable"] = out["monitor_eligible"].fillna(False).astype(bool) & ~out["fundamental_veto"].fillna(False).astype(bool)
    out.attrs = {}
    return out, {
        "enabled": True,
        "veto_ratio": float(summary.veto_ratio),
        "vetoed_rows": int(summary.vetoed_rows),
        "reason_counts": dict(summary.reason_counts),
    }


def _evaluate_one(
    test_df: pd.DataFrame,
    base_research_cfg: dict[str, Any],
    backtest_cfg: dict[str, Any],
    scenario: FocusScenario,
    top_n: int,
    sleeve_count: int,
    weight_change_threshold: float,
    rank_change_threshold: int,
) -> dict[str, Any]:
    scenario_research = {
        **base_research_cfg,
        "top_n": int(top_n),
        "sleeve_count": int(sleeve_count),
        "weight_change_threshold": float(weight_change_threshold),
        "rank_change_threshold": int(rank_change_threshold),
    }
    targets = build_portfolio_targets(
        test_df,
        score_col="score",
        top_n=int(top_n),
        rebalance_every=int(scenario_research["rebalance_every"]),
        sleeve_count=int(sleeve_count),
        execution_lag=int(scenario_research["execution_lag"]),
        weighting_method=str(scenario_research["weighting_method"]),
        max_weight=float(scenario_research["max_weight"]),
        industry_cap=float(scenario_research["industry_cap"]),
        min_holdings=int(scenario_research["min_holdings"]),
        score_threshold=float(scenario_research["score_threshold"]),
        softmax_temperature=float(scenario_research["softmax_temperature"]),
        weight_change_threshold=float(weight_change_threshold),
        rank_change_threshold=int(rank_change_threshold),
        entry_score_advantage_threshold=float(scenario_research.get("entry_score_advantage_threshold", 0.0)),
    )
    _, metrics = DailyBacktester(
        config=BacktestConfig(
            **backtest_cfg,
            signal_time=str(scenario_research["signal_time"]),
            execution_price=str(scenario_research["execution_price"]),
            execution_lag=int(scenario_research["execution_lag"]),
            holding_window=int(scenario_research["holding_window"]),
            sleeve_count=int(sleeve_count),
        )
    ).run(test_df, targets)
    latest_targets = _latest_targets(targets)
    assessment = assessment_payload(metrics=metrics, picks=latest_targets)
    return {
        "scenario": scenario.name,
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


def _threshold_grid(
    test_df: pd.DataFrame,
    base_research_cfg: dict[str, Any],
    backtest_cfg: dict[str, Any],
    scenario: FocusScenario,
    weight_thresholds: list[float],
    rank_thresholds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(weight_thresholds) * len(rank_thresholds)
    idx = 0
    for weight_threshold in weight_thresholds:
        for rank_threshold in rank_thresholds:
            idx += 1
            print(
                f"[{scenario.name}] threshold_grid {idx}/{total}: weight_change_threshold={weight_threshold:.2f}, rank_change_threshold={rank_threshold}",
                flush=True,
            )
            rows.append(
                _evaluate_one(
                    test_df=test_df,
                    base_research_cfg=base_research_cfg,
                    backtest_cfg=backtest_cfg,
                    scenario=scenario,
                    top_n=int(base_research_cfg["top_n"]),
                    sleeve_count=int(base_research_cfg["sleeve_count"]),
                    weight_change_threshold=float(weight_threshold),
                    rank_change_threshold=int(rank_threshold),
                )
            )
    return pd.DataFrame(rows).sort_values(
        ["sharpe", "after_cost_return_drag", "avg_turnover", "max_drawdown"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)


def _construction_grid(
    test_df: pd.DataFrame,
    base_research_cfg: dict[str, Any],
    backtest_cfg: dict[str, Any],
    scenario: FocusScenario,
    top_n_grid: list[int],
    sleeve_grid: list[int],
    weight_change_threshold: float,
    rank_change_threshold: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(top_n_grid) * len(sleeve_grid)
    idx = 0
    for top_n in top_n_grid:
        for sleeve_count in sleeve_grid:
            idx += 1
            print(
                f"[{scenario.name}] construction_grid {idx}/{total}: top_n={top_n}, sleeve_count={sleeve_count}",
                flush=True,
            )
            rows.append(
                _evaluate_one(
                    test_df=test_df,
                    base_research_cfg=base_research_cfg,
                    backtest_cfg=backtest_cfg,
                    scenario=scenario,
                    top_n=int(top_n),
                    sleeve_count=int(sleeve_count),
                    weight_change_threshold=float(weight_change_threshold),
                    rank_change_threshold=int(rank_change_threshold),
                )
            )
    return pd.DataFrame(rows).sort_values(
        ["sharpe", "after_cost_return_drag", "avg_turnover", "max_drawdown"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)


def _build_markdown(
    scenario_summary_df: pd.DataFrame,
    recommended_config: dict[str, Any],
) -> str:
    lines = [
        "# Focused QFQ Tuning Summary",
        "",
        "## Scenario Summary",
        "",
        scenario_summary_df.to_markdown(index=False) if not scenario_summary_df.empty else "_No results._",
        "",
        "## Recommended Default",
        "",
        f"- classification: `{recommended_config.get('strategy_classification', '')}`",
        f"- top_n: `{recommended_config.get('top_n')}`",
        f"- sleeve_count: `{recommended_config.get('sleeve_count')}`",
        f"- weight_change_threshold: `{recommended_config.get('weight_change_threshold')}`",
        f"- rank_change_threshold: `{recommended_config.get('rank_change_threshold')}`",
        f"- fundamental_veto: `{recommended_config.get('enable_fundamental_veto')}`",
        "",
        "Recommendation priority remains after-cost Sharpe first, then cost drag, turnover, and drawdown.",
        "",
    ]
    return "\n".join(lines)


def run_focused_qfq_tuning(
    data_path: str | Path,
    research_config_path: str | Path,
    backtest_config_path: str | Path,
    output_dir: str | Path,
    data_adjust: str = "qfq",
    scenarios: list[FocusScenario] | None = None,
    top_n_grid: list[int] | None = None,
    sleeve_grid: list[int] | None = None,
    weight_thresholds: list[float] | None = None,
    rank_thresholds: list[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_research_cfg = {**DEFAULT_RESEARCH, **load_json(research_config_path)}
    backtest_cfg = {**DEFAULT_BACKTEST, **load_json(backtest_config_path)}
    base_research_cfg = {
        **root_research_cfg,
        "enable_fundamental_veto": False,
    }
    scenarios = scenarios or [
        FocusScenario(
            name="recommended_current",
            description="Current recommended config without the fundamental veto layer.",
        ),
        FocusScenario(
            name="recommended_with_fundamental_veto",
            description="Current recommended config plus real Tushare-backed fundamental veto.",
            veto_enabled=True,
            veto_overrides={
                "min_profit_quality": 0.0,
                "min_earnings_growth": -0.10,
                "min_cashflow_quality": -0.10,
                "min_roe": 0.02,
                "max_debt_to_asset": 0.80,
            },
        ),
    ]
    top_n_grid = top_n_grid or [12, 15, 20]
    sleeve_grid = sleeve_grid or [1, 5]
    weight_thresholds = weight_thresholds or [0.02, 0.03, 0.05]
    rank_thresholds = rank_thresholds or [3, 5, 8]

    print(f"[focused_tuning] loading data from {data_path}", flush=True)
    df = CSVDataSource(data_path, adjust=data_adjust).load()
    print(f"[focused_tuning] loaded {len(df)} rows; preparing stable6 research frame", flush=True)
    prepared_df, metadata = prepare_research_frame(df.copy(), base_research_cfg)
    dates = sorted(prepared_df["date"].drop_duplicates())
    train_dates, test_dates = _train_test_split_dates(dates)
    print(
        f"[focused_tuning] prepare complete: train_dates={len(train_dates)}, test_dates={len(test_dates)}; scoring once for reuse",
        flush=True,
    )
    scored_df, _ = score_research_frame(
        df=prepared_df,
        research_cfg=base_research_cfg,
        metadata=metadata,
        train_dates=train_dates,
    )
    print("[focused_tuning] score frame ready; starting scenario sweeps", flush=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scenario_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        print(f"[focused_tuning] scenario={scenario.name} veto_enabled={scenario.veto_enabled}", flush=True)
        scenario_df, veto_summary = _apply_veto_overlay(scored_df, base_research_cfg=base_research_cfg, scenario=scenario)
        test_df = scenario_df.loc[scenario_df["date"].isin(test_dates)].copy()

        threshold_df = _threshold_grid(
            test_df=test_df,
            base_research_cfg=base_research_cfg,
            backtest_cfg=backtest_cfg,
            scenario=scenario,
            weight_thresholds=weight_thresholds,
            rank_thresholds=rank_thresholds,
        )
        best_threshold = _best_row(threshold_df, prefer_tradable=True)

        construction_df = _construction_grid(
            test_df=test_df,
            base_research_cfg=base_research_cfg,
            backtest_cfg=backtest_cfg,
            scenario=scenario,
            top_n_grid=top_n_grid,
            sleeve_grid=sleeve_grid,
            weight_change_threshold=float(best_threshold["weight_change_threshold"]),
            rank_change_threshold=int(best_threshold["rank_change_threshold"]),
        )
        best_construction = _best_row(construction_df, prefer_tradable=True)
        scenario_rows.append(
            {
                "scenario": scenario.name,
                "description": scenario.description,
                "veto_enabled": bool(scenario.veto_enabled),
                "veto_ratio": float(veto_summary["veto_ratio"]),
                "top_n": int(best_construction["top_n"]),
                "sleeve_count": int(best_construction["sleeve_count"]),
                "weight_change_threshold": float(best_threshold["weight_change_threshold"]),
                "rank_change_threshold": int(best_threshold["rank_change_threshold"]),
                "annual_return": float(best_construction["annual_return"]),
                "gross_annual_return": float(best_construction["gross_annual_return"]),
                "sharpe": float(best_construction["sharpe"]),
                "max_drawdown": float(best_construction["max_drawdown"]),
                "avg_turnover": float(best_construction["avg_turnover"]),
                "after_cost_return_drag": float(best_construction["after_cost_return_drag"]),
                "strategy_classification": str(best_construction["strategy_classification"]),
                "veto_overrides": json.dumps(scenario.veto_overrides, ensure_ascii=False),
            }
        )

        scenario_dir = output_path / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        threshold_df.to_csv(scenario_dir / "threshold_grid.csv", index=False)
        construction_df.to_csv(scenario_dir / "construction_grid.csv", index=False)
        (scenario_dir / "veto_summary.json").write_text(
            json.dumps(veto_summary, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        print(f"[focused_tuning] scenario={scenario.name} complete", flush=True)

    scenario_summary_df = pd.DataFrame(scenario_rows).sort_values(
        ["sharpe", "after_cost_return_drag", "avg_turnover", "max_drawdown"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    best = _best_row(scenario_summary_df, prefer_tradable=True)
    best_veto = {}
    for scenario in scenarios:
        if scenario.name == best.get("scenario"):
            best_veto = scenario.veto_overrides
            break
    recommended_config = {
        **root_research_cfg,
        "top_n": int(best["top_n"]),
        "sleeve_count": int(best["sleeve_count"]),
        "weight_change_threshold": float(best["weight_change_threshold"]),
        "rank_change_threshold": int(best["rank_change_threshold"]),
        "enable_fundamental_veto": bool(best["veto_enabled"]),
        "min_profit_quality": float(best_veto.get("min_profit_quality", root_research_cfg.get("min_profit_quality", -0.10))),
        "min_earnings_growth": float(best_veto.get("min_earnings_growth", root_research_cfg.get("min_earnings_growth", -0.15))),
        "min_cashflow_quality": float(best_veto.get("min_cashflow_quality", root_research_cfg.get("min_cashflow_quality", -0.10))),
        "min_roe": float(best_veto.get("min_roe", root_research_cfg.get("min_roe", -0.05))),
        "max_debt_to_asset": float(best_veto.get("max_debt_to_asset", root_research_cfg.get("max_debt_to_asset", 0.75))),
        "recommendation_basis": "focused qfq tuning on stable6 + 500 universe + 2-year train window with after-cost Sharpe priority",
        "strategy_classification": str(best["strategy_classification"]),
    }

    scenario_summary_df.to_csv(output_path / "scenario_summary.csv", index=False)
    (output_path / "recommended_default_config.json").write_text(
        json.dumps(recommended_config, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_path / "focused_tuning_summary.json").write_text(
        json.dumps(
            {
                "recommended_config": recommended_config,
                "scenario_summary": scenario_summary_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output_path / "focused_tuning_summary.md").write_text(
        _build_markdown(scenario_summary_df=scenario_summary_df, recommended_config=recommended_config),
        encoding="utf-8",
    )
    print(f"[focused_tuning] outputs written to {output_path}", flush=True)
    return scenario_summary_df, recommended_config
