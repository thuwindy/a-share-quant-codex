from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

import pandas as pd

from ashare_quant.analysis.audit_report import build_leakage_bias_checklist, build_strategy_change_log
from ashare_quant.analysis.experiment_runner import ExperimentScenario, json_ready_records, run_single_experiment
from ashare_quant.data.csv_adapter import CSVDataSource
from ashare_quant.factors.technical import FACTOR_SETS
from ashare_quant.pipeline import DEFAULT_BACKTEST, DEFAULT_RESEARCH, load_json


STABLE6_COLUMNS = [f"{name}_neu" for name in FACTOR_SETS["stable6"]]
ALL12_COLUMNS = [f"{name}_neu" for name in FACTOR_SETS["all12"]]


def _base_current_monitor_overrides() -> dict[str, Any]:
    return {
        "factor_set": "all12",
        "factor_columns": ALL12_COLUMNS,
        "label_horizons": [3, 5, 10],
        "use_horizons": [3, 5, 10],
        "horizon_score_weights": {"3": 0.3, "5": 0.4, "10": 0.3},
        "holding_window": 5,
        "holding_period": 5,
        "train_window_years": 0,
        "label_type": "neutralized_residual",
        "ranker_type": "dynamic_icir",
        "factor_combination_mode": "group",
        "top_n": 12,
        "rebalance_every": 5,
        "sleeve_count": 5,
        "weighting_method": "rank",
        "max_weight": 0.12,
        "industry_cap": 0.25,
        "min_holdings": 8,
        "weight_change_threshold": 0.01,
        "rank_change_threshold": 2,
        "entry_score_advantage_threshold": 0.0,
        "regime_gate_enabled": False,
        "enable_fundamental_veto": False,
    }


def build_tradable_upgrade_scenarios() -> list[ExperimentScenario]:
    return [
        ExperimentScenario(
            name="baseline_current_default",
            description="Current monitor-oriented default: all12 groups, 3/5/10 horizons, 5 sleeves, and loose no-trade bands.",
            research_overrides=_base_current_monitor_overrides(),
            backtest_overrides={"use_liquidity_aware_cost": True},
        ),
        ExperimentScenario(
            name="stable6",
            description="Switch to the stable6 factor set and 2-year training window while preserving the old multi-horizon structure.",
            research_overrides={
                **_base_current_monitor_overrides(),
                "factor_set": "stable6",
                "factor_columns": STABLE6_COLUMNS,
                "train_window_years": 2,
            },
            backtest_overrides={"use_liquidity_aware_cost": True},
        ),
        ExperimentScenario(
            name="stable6_short",
            description="Stable6 with only 3d/5d labels aligned to a 5-day hold.",
            research_overrides={
                **_base_current_monitor_overrides(),
                "factor_set": "stable6",
                "factor_columns": STABLE6_COLUMNS,
                "train_window_years": 2,
                "label_horizons": [3, 5],
                "use_horizons": [3, 5],
                "horizon_score_weights": {"3": 0.45, "5": 0.55},
            },
            backtest_overrides={"use_liquidity_aware_cost": True},
        ),
        ExperimentScenario(
            name="stable6_short_notrade",
            description="Stable6 short-horizon core plus stronger no-trade bands and entry advantage threshold.",
            research_overrides={
                **_base_current_monitor_overrides(),
                "factor_set": "stable6",
                "factor_columns": STABLE6_COLUMNS,
                "train_window_years": 2,
                "label_horizons": [3, 5],
                "use_horizons": [3, 5],
                "horizon_score_weights": {"3": 0.45, "5": 0.55},
                "weight_change_threshold": 0.03,
                "rank_change_threshold": 5,
                "entry_score_advantage_threshold": 0.03,
            },
            backtest_overrides={"use_liquidity_aware_cost": True},
        ),
        ExperimentScenario(
            name="stable6_short_notrade_topn",
            description="Search Top-N and sleeve choices on the stable6 short-horizon, low-turnover core.",
            research_overrides={
                **_base_current_monitor_overrides(),
                "factor_set": "stable6",
                "factor_columns": STABLE6_COLUMNS,
                "train_window_years": 2,
                "label_horizons": [3, 5],
                "use_horizons": [3, 5],
                "horizon_score_weights": {"3": 0.45, "5": 0.55},
                "weight_change_threshold": 0.03,
                "rank_change_threshold": 5,
                "entry_score_advantage_threshold": 0.03,
            },
            backtest_overrides={"use_liquidity_aware_cost": True},
            search_top_n=True,
            top_n_grid=[12, 15, 20],
            weighting_methods=["rank"],
            sleeve_grid=[1, 5],
        ),
        ExperimentScenario(
            name="stable6_short_notrade_topn_regime",
            description="Add a simple regime gate on top of the stable6 short-horizon tradable core.",
            research_overrides={
                **_base_current_monitor_overrides(),
                "factor_set": "stable6",
                "factor_columns": STABLE6_COLUMNS,
                "train_window_years": 2,
                "label_horizons": [3, 5],
                "use_horizons": [3, 5],
                "horizon_score_weights": {"3": 0.45, "5": 0.55},
                "weight_change_threshold": 0.03,
                "rank_change_threshold": 5,
                "entry_score_advantage_threshold": 0.03,
                "regime_gate_enabled": True,
            },
            backtest_overrides={"use_liquidity_aware_cost": True},
            search_top_n=True,
            top_n_grid=[12, 15, 20],
            weighting_methods=["rank"],
            sleeve_grid=[1, 5],
        ),
        ExperimentScenario(
            name="stable6_short_notrade_topn_regime_veto",
            description="Final tradable prototype candidate with regime gate plus optional fundamental veto layer.",
            research_overrides={
                **_base_current_monitor_overrides(),
                "factor_set": "stable6",
                "factor_columns": STABLE6_COLUMNS,
                "train_window_years": 2,
                "label_horizons": [3, 5],
                "use_horizons": [3, 5],
                "horizon_score_weights": {"3": 0.45, "5": 0.55},
                "weight_change_threshold": 0.03,
                "rank_change_threshold": 5,
                "entry_score_advantage_threshold": 0.03,
                "regime_gate_enabled": True,
                "enable_fundamental_veto": True,
            },
            backtest_overrides={"use_liquidity_aware_cost": True},
            search_top_n=True,
            top_n_grid=[12, 15, 20],
            weighting_methods=["rank"],
            sleeve_grid=[1, 5],
        ),
    ]


def validate_walk_forward_baseline(compare_csv_path: str | Path) -> dict[str, Any]:
    path = Path(compare_csv_path)
    if not path.exists():
        return {"available": False, "stable6_validated": False}
    df = pd.read_csv(path)
    stable = df.loc[
        (df["config"] == "stable6")
        & (df["universe_tag"].astype(str) == "500")
        & (pd.to_numeric(df["train_years"], errors="coerce") == 2)
    ].copy()
    if stable.empty:
        return {"available": True, "stable6_validated": False}
    best = df.sort_values(["mean_sharpe", "worst_fold_drawdown"], ascending=[False, False]).iloc[0]
    stable_row = stable.iloc[0]
    stable_validated = float(stable_row["mean_sharpe"]) >= float(best["mean_sharpe"]) - 0.08
    return {
        "available": True,
        "stable6_validated": stable_validated,
        "stable6_row": stable_row.to_dict(),
        "best_row": best.to_dict(),
    }


def _run_threshold_grid(
    df: pd.DataFrame,
    research_cfg: dict[str, Any],
    backtest_cfg: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for weight_threshold in (0.02, 0.03, 0.05):
        for rank_threshold in (3, 5, 8):
            scenario = ExperimentScenario(
                name=f"ntb_w{weight_threshold:.2f}_r{rank_threshold}",
                description="No-trade band sensitivity.",
                research_overrides={
                    **_base_current_monitor_overrides(),
                    "factor_set": "stable6",
                    "factor_columns": STABLE6_COLUMNS,
                    "train_window_years": 2,
                    "label_horizons": [3, 5],
                    "use_horizons": [3, 5],
                    "horizon_score_weights": {"3": 0.45, "5": 0.55},
                    "weight_change_threshold": weight_threshold,
                    "rank_change_threshold": rank_threshold,
                    "entry_score_advantage_threshold": 0.03,
                },
                backtest_overrides={"use_liquidity_aware_cost": True},
            )
            summary, _ = run_single_experiment(df=df, research_cfg=research_cfg, backtest_cfg=backtest_cfg, scenario=scenario)
            rows.append(
                {
                    "weight_change_threshold": weight_threshold,
                    "rank_change_threshold": rank_threshold,
                    "annual_return": summary["annual_return"],
                    "gross_annual_return": summary.get("gross_annual_return", 0.0),
                    "sharpe": summary["sharpe"],
                    "max_drawdown": summary["max_drawdown"],
                    "avg_turnover": summary["avg_turnover"],
                    "after_cost_return_drag": summary.get("after_cost_return_drag", 0.0),
                }
            )
    return pd.DataFrame(rows).sort_values(["sharpe", "avg_turnover"], ascending=[False, True]).reset_index(drop=True)


def _run_factor_set_compare(
    df: pd.DataFrame,
    research_cfg: dict[str, Any],
    backtest_cfg: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for factor_set in ("all12", "stable6"):
        scenario = ExperimentScenario(
            name=f"factor_set_{factor_set}",
            description="Factor-set comparison on the same short-horizon tradable core.",
            research_overrides={
                **_base_current_monitor_overrides(),
                "factor_set": factor_set,
                "factor_columns": [f"{name}_neu" for name in FACTOR_SETS[factor_set]],
                "train_window_years": 2,
                "label_horizons": [3, 5],
                "use_horizons": [3, 5],
                "horizon_score_weights": {"3": 0.45, "5": 0.55},
                "weight_change_threshold": 0.03,
                "rank_change_threshold": 5,
                "entry_score_advantage_threshold": 0.03,
            },
            backtest_overrides={"use_liquidity_aware_cost": True},
        )
        summary, _ = run_single_experiment(df=df, research_cfg=research_cfg, backtest_cfg=backtest_cfg, scenario=scenario)
        rows.append(
            {
                "factor_set": factor_set,
                "annual_return": summary["annual_return"],
                "gross_annual_return": summary.get("gross_annual_return", 0.0),
                "sharpe": summary["sharpe"],
                "max_drawdown": summary["max_drawdown"],
                "avg_turnover": summary["avg_turnover"],
                "after_cost_return_drag": summary.get("after_cost_return_drag", 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def _run_single_factor_diagnostics(
    df: pd.DataFrame,
    research_cfg: dict[str, Any],
    backtest_cfg: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    quantile_dir = output_dir / "single_factor_quantiles"
    quantile_dir.mkdir(parents=True, exist_ok=True)
    for factor in FACTOR_SETS["stable6"]:
        factor_col = f"{factor}_neu"
        scenario = ExperimentScenario(
            name=f"single_{factor}",
            description="Single-factor stable6 diagnostic.",
            research_overrides={
                **_base_current_monitor_overrides(),
                "factor_set": "custom",
                "factor_columns": [factor_col],
                "factor_combination_mode": "direct",
                "train_window_years": 2,
                "label_horizons": [3, 5],
                "use_horizons": [3, 5],
                "horizon_score_weights": {"3": 0.45, "5": 0.55},
                "weight_change_threshold": 0.03,
                "rank_change_threshold": 5,
                "entry_score_advantage_threshold": 0.03,
            },
            backtest_overrides={"use_liquidity_aware_cost": True},
        )
        summary, details = run_single_experiment(df=df, research_cfg=research_cfg, backtest_cfg=backtest_cfg, scenario=scenario)
        quantile_path = quantile_dir / f"{factor}_quantile_summary.csv"
        details["quantile_summary"].to_csv(quantile_path, index=False)
        rows.append(
            {
                "factor": factor_col,
                "annual_return": summary["annual_return"],
                "gross_annual_return": summary.get("gross_annual_return", 0.0),
                "sharpe": summary["sharpe"],
                "max_drawdown": summary["max_drawdown"],
                "avg_turnover": summary["avg_turnover"],
                "after_cost_return_drag": summary.get("after_cost_return_drag", 0.0),
                "ic_mean": summary["ic_mean"],
                "ic_std": summary["ic_std"],
                "icir": summary["icir"],
                "strategy_classification": summary.get("strategy_classification", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["icir", "sharpe"], ascending=False).reset_index(drop=True)


def _build_delta_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    ordered = summary_df.copy()
    deltas: list[dict[str, Any]] = []
    previous = None
    for row in ordered.to_dict(orient="records"):
        current = {
            "scenario": row["scenario"],
            "annual_return": row["annual_return"],
            "sharpe": row["sharpe"],
            "max_drawdown": row["max_drawdown"],
            "avg_turnover": row["avg_turnover"],
            "after_cost_return_drag": row.get("after_cost_return_drag", 0.0),
        }
        if previous is None:
            deltas.append({"scenario": row["scenario"], "delta_vs_prev": "baseline"})
        else:
            deltas.append(
                {
                    "scenario": row["scenario"],
                    "delta_annual_return": current["annual_return"] - previous["annual_return"],
                    "delta_sharpe": current["sharpe"] - previous["sharpe"],
                    "delta_max_drawdown": current["max_drawdown"] - previous["max_drawdown"],
                    "delta_avg_turnover": current["avg_turnover"] - previous["avg_turnover"],
                    "delta_cost_drag": current["after_cost_return_drag"] - previous["after_cost_return_drag"],
                }
            )
        previous = current
    return pd.DataFrame(deltas)


def _recommended_config_from_results(
    summary_df: pd.DataFrame,
    threshold_grid_df: pd.DataFrame,
) -> dict[str, Any]:
    best_summary = summary_df.sort_values(["sharpe", "max_drawdown"], ascending=[False, False]).iloc[0]
    best_threshold = threshold_grid_df.iloc[0] if not threshold_grid_df.empty else None
    selected_grid = best_summary.get("selected_grid", {}) or {}
    return {
        "factor_set": "stable6",
        "universe": {"max_codes": 500, "suffixes": ["SH", "SZ"]},
        "train_window_years": 2,
        "use_horizons": [3, 5],
        "holding_period": 5,
        "top_n": int(selected_grid.get("top_n", best_summary.get("top_n", 12))),
        "weighting_scheme": str(selected_grid.get("weighting_method", best_summary.get("weighting_method", "rank"))),
        "sleeve_count": int(selected_grid.get("sleeve_count", best_summary.get("sleeve_count", 5))),
        "no_trade_band": {
            "weight_change_threshold": float(best_threshold["weight_change_threshold"]) if best_threshold is not None else 0.03,
            "rank_change_threshold": int(best_threshold["rank_change_threshold"]) if best_threshold is not None else 5,
            "entry_score_advantage_threshold": 0.03,
        },
        "regime_gate": {
            "enabled": bool("regime" in str(best_summary.get("scenario", ""))),
            "half_risk_multiplier": 0.50,
            "low_risk_multiplier": 0.0,
        },
        "fundamental_veto": {
            "enabled": bool("veto" in str(best_summary.get("scenario", ""))),
        },
        "expected_classification": best_summary.get("strategy_classification", "research candidate engine / monitor only"),
    }


def _build_markdown(
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    factor_set_compare_df: pd.DataFrame,
    threshold_grid_df: pd.DataFrame,
    single_factor_df: pd.DataFrame,
    walk_forward_validation: dict[str, Any],
    recommended_config: dict[str, Any],
    change_log_df: pd.DataFrame,
    leakage_check_df: pd.DataFrame,
) -> str:
    lines = [
        "# Tradable Upgrade Summary",
        "",
        "## Walk-Forward Baseline Check",
        "",
        f"- stable6_500_2y_validated: `{walk_forward_validation.get('stable6_validated', False)}`",
    ]
    stable_row = walk_forward_validation.get("stable6_row", {})
    if stable_row:
        lines.extend(
            [
                f"- stable6 mean_sharpe: `{float(stable_row.get('mean_sharpe', 0.0)):.4f}`",
                f"- stable6 mean_annual_return: `{float(stable_row.get('mean_annual_return', 0.0)):.4f}`",
                f"- stable6 worst_fold_drawdown: `{float(stable_row.get('worst_fold_drawdown', 0.0)):.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Step Ablation",
            "",
            "| scenario | annual_return | gross_annual_return | sharpe | max_drawdown | avg_turnover | cost_drag | classification |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary_df.itertuples(index=False):
        lines.append(
            "| {scenario} | {annual_return:.4f} | {gross_annual_return:.4f} | {sharpe:.4f} | {max_drawdown:.4f} | {avg_turnover:.4f} | {cost_drag:.4f} | {classification} |".format(
                scenario=row.scenario,
                annual_return=float(row.annual_return),
                gross_annual_return=float(getattr(row, "gross_annual_return", 0.0)),
                sharpe=float(row.sharpe),
                max_drawdown=float(row.max_drawdown),
                avg_turnover=float(row.avg_turnover),
                cost_drag=float(getattr(row, "after_cost_return_drag", 0.0)),
                classification=getattr(row, "strategy_classification", ""),
            )
        )

    if not factor_set_compare_df.empty:
        lines.extend(
            [
                "",
                "## Stable6 Vs All12",
                "",
                factor_set_compare_df.to_markdown(index=False),
            ]
        )
    if not threshold_grid_df.empty:
        lines.extend(
            [
                "",
                "## No-Trade Band Sensitivity",
                "",
                threshold_grid_df.head(5).to_markdown(index=False),
            ]
        )
    if not single_factor_df.empty:
        lines.extend(
            [
                "",
                "## Stable6 Single-Factor Diagnostics",
                "",
                single_factor_df.head(6).to_markdown(index=False),
            ]
        )
    if not change_log_df.empty:
        lines.extend(
            [
                "",
                "## Strategy Change Log",
                "",
                change_log_df.to_markdown(index=False),
            ]
        )
    if not leakage_check_df.empty:
        lines.extend(
            [
                "",
                "## Leakage And Bias Checklist",
                "",
                leakage_check_df.to_markdown(index=False),
            ]
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- recommended classification: `{recommended_config['expected_classification']}`",
            f"- recommended top_n: `{recommended_config['top_n']}`",
            f"- recommended sleeve_count: `{recommended_config['sleeve_count']}`",
            f"- recommended weighting_scheme: `{recommended_config['weighting_scheme']}`",
            f"- recommended no_trade_band: `{json.dumps(recommended_config['no_trade_band'], ensure_ascii=False)}`",
            "",
            dedent(
                """
                The target state is to preserve the research value of the observation pool while only promoting a configuration to
                the deployable layer when after-cost Sharpe, turnover, and drawdown all improve together. If the recommended
                classification is still `research candidate engine / monitor only`, the strategy should continue to be treated as an
                observation system rather than an automatic buy list.
                """
            ).strip(),
        ]
    )
    return "\n".join(lines) + "\n"


def run_tradable_upgrade_ablation(
    data_path: str | Path,
    research_config_path: str | Path,
    backtest_config_path: str | Path,
    output_dir: str | Path,
    data_adjust: str = "qfq",
    walk_forward_compare_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    research_cfg = {**DEFAULT_RESEARCH, **load_json(research_config_path)}
    backtest_cfg = {**DEFAULT_BACKTEST, **load_json(backtest_config_path)}
    df = CSVDataSource(data_path, adjust=data_adjust).load()
    scenarios = build_tradable_upgrade_scenarios()

    summary_rows: list[dict[str, Any]] = []
    detail_tables: dict[str, dict[str, pd.DataFrame]] = {}
    for scenario in scenarios:
        summary, details = run_single_experiment(df=df, research_cfg=research_cfg, backtest_cfg=backtest_cfg, scenario=scenario)
        summary_rows.append(summary)
        detail_tables[scenario.name] = details
    summary_df = pd.DataFrame(summary_rows)
    scenario_order = {scenario.name: idx for idx, scenario in enumerate(scenarios)}
    summary_df["scenario_order"] = summary_df["scenario"].map(scenario_order)
    summary_df = summary_df.sort_values("scenario_order").reset_index(drop=True)

    threshold_grid_df = _run_threshold_grid(df=df, research_cfg=research_cfg, backtest_cfg=backtest_cfg)
    factor_set_compare_df = _run_factor_set_compare(df=df, research_cfg=research_cfg, backtest_cfg=backtest_cfg)
    single_factor_df = _run_single_factor_diagnostics(
        df=df,
        research_cfg=research_cfg,
        backtest_cfg=backtest_cfg,
        output_dir=Path(output_dir),
    )
    delta_df = _build_delta_table(summary_df)
    change_log_df = build_strategy_change_log(summary_df)
    leakage_check_df = build_leakage_bias_checklist(summary_df)
    walk_forward_validation = validate_walk_forward_baseline(
        walk_forward_compare_path
        or (Path(output_dir).parents[0] / "walk_forward_20260329_config_compare.csv")
    )
    recommended_config = _recommended_config_from_results(summary_df=summary_df, threshold_grid_df=threshold_grid_df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_df.drop(columns=["scenario_order", "display_weights", "selected_grid"], errors="ignore").to_csv(
        output_path / "ablation_summary.csv",
        index=False,
    )
    delta_df.to_csv(output_path / "ablation_deltas.csv", index=False)
    threshold_grid_df.to_csv(output_path / "no_trade_band_grid.csv", index=False)
    factor_set_compare_df.to_csv(output_path / "factor_set_compare.csv", index=False)
    single_factor_df.to_csv(output_path / "single_factor_diagnostics.csv", index=False)
    change_log_df.to_csv(output_path / "strategy_change_log.csv", index=False)
    leakage_check_df.to_csv(output_path / "leakage_bias_checklist.csv", index=False)
    (output_path / "recommended_default_config.json").write_text(
        json.dumps(recommended_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "walk_forward_validation.json").write_text(
        json.dumps(walk_forward_validation, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output_path / "ablation_summary.md").write_text(
        _build_markdown(
            summary_df=summary_df,
            delta_df=delta_df,
            factor_set_compare_df=factor_set_compare_df,
            threshold_grid_df=threshold_grid_df,
            single_factor_df=single_factor_df,
            walk_forward_validation=walk_forward_validation,
            recommended_config=recommended_config,
            change_log_df=change_log_df,
            leakage_check_df=leakage_check_df,
        ),
        encoding="utf-8",
    )
    for scenario_name, tables in detail_tables.items():
        scenario_dir = output_path / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for table_name, table_df in tables.items():
            table_df.to_csv(scenario_dir / f"{table_name}.csv", index=False)

    payload = {
        "recommended_config": recommended_config,
        "walk_forward_validation": walk_forward_validation,
        "summary_records": json_ready_records(summary_df.drop(columns=["scenario_order"], errors="ignore")),
    }
    (output_path / "ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
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
    return summary_df, payload
