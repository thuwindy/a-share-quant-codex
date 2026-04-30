from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from ashare_quant.analysis.candidate_pool_quality import (
    add_candidate_path_columns,
    summarize_failure_attribution,
    summarize_candidate_pool_quality,
)
from ashare_quant.backtest.engine import BacktestConfig, DailyBacktester
from ashare_quant.data.base import MarketDataSource
from ashare_quant.data.csv_adapter import CSVDataSource, ensure_price_views
from ashare_quant.data.universe import UniverseFilterConfig, apply_basic_universe_filters
from ashare_quant.factors.factor_registry import (
    blocked_production_factors,
    resolve_family_factor_request,
    summarize_factor_families,
)
from ashare_quant.factors.factor_preprocess import FactorPreprocessConfig
from ashare_quant.factors.technical import (
    FACTOR_GROUPS,
    add_group_scores,
    add_technical_factors,
    default_horizon_factor_map,
    factor_set_columns,
    normalize_requested_factor_columns,
    requested_raw_factor_columns,
)
from ashare_quant.factors.neutralize import neutralize_by_size_and_industry
from ashare_quant.fundamental_veto import FundamentalVetoConfig, apply_fundamental_veto
from ashare_quant.labels.label_builder import ExecutionSpec, add_label_columns, label_column_name
from ashare_quant.models.dynamic_weighting import DynamicWeightConfig
from ashare_quant.models.boost_ranker import BoostRankerConfig, TreeBoostRanker
from ashare_quant.models.linear_ranker import FixedWeightRanker, ICWeightedRanker, RollingICIRRanker
from ashare_quant.models.ml_ranker import MLLogisticConfig, MLLogisticRanker, MLRidgeConfig, MLRidgeRanker
from ashare_quant.models.offline_signal_model import OfflineSignalPredictor
from ashare_quant.portfolio.construction import build_portfolio_targets
from ashare_quant.premium_constraints import PremiumConstraintConfig, apply_premium_dynamic_constraints
from ashare_quant.regime_gate import RegimeGateConfig, apply_regime_gate
from ashare_quant.strategy_classifier import assessment_payload


DEFAULT_RESEARCH = {
    "label_horizon": 5,
    "label_horizons": [5],
    "use_horizons": [],
    "label_type": "raw",
    "signal_time": "close",
    "execution_price": "close",
    "execution_lag": 1,
    "holding_window": 5,
    "holding_period": 5,
    "train_window_years": 0,
    "top_n": 8,
    "rebalance_every": 5,
    "sleeve_count": 1,
    "ranker_type": "dynamic_icir",
    "signal_mode": "factor",
    "model_type": "",
    "model_path": "",
    "prediction_type": "regression",
    "buy_threshold": 0.0,
    "sell_threshold": 0.0,
    "factor_combination_mode": "direct",
    "factor_set": "all12",
    "factor_families": [],
    "is_production_config": None,
    "allow_low_freq_experimental_in_production": False,
    "weighting_method": "equal",
    "max_weight": 0.2,
    "industry_cap": 0.4,
    "min_holdings": 5,
    "score_threshold": 0.0,
    "softmax_temperature": 1.0,
    "weight_change_threshold": 0.0,
    "rank_change_threshold": 0,
    "entry_score_advantage_threshold": 0.0,
    "entry_filter_overhead_enabled": False,
    "entry_filter_min_overhead_resistance": float("-inf"),
    "entry_filter_score_floor": -9999.0,
    "dynamic_window": 126,
    "dynamic_ewma_span": 63,
    "dynamic_icir_threshold": 0.0,
    "dynamic_sign_flip_threshold": 0.5,
    "dynamic_min_history": 20,
    "ml_alpha_grid": [0.1, 1.0, 10.0],
    "ml_cv_folds": 3,
    "ml_min_rows": 200,
    "ml_c_grid": [0.1, 1.0, 10.0],
    "ml_positive_threshold": 0.0,
    "boost_learning_rates": [0.05, 0.1],
    "boost_n_estimators_grid": [100, 200],
    "boost_max_depth_grid": [3, 5],
    "boost_allow_sklearn_fallback": True,
    "fixed_factor_weights": {},
    "clip_method": "mad",
    "mad_scale": 5.0,
    "quantile_lower": 0.01,
    "quantile_upper": 0.99,
    "min_listing_days": 120,
    "min_avg_amount_20": 20_000_000.0,
    "min_market_cap": 3_000_000_000.0,
    "min_price": 3.0,
    "max_price": float("inf"),
    "near_limit_buffer": 0.005,
    "include_technical": True,
    "include_fundamental": True,
    "include_flow": True,
    "include_regime": True,
    "dynamic_horizon_weighting": False,
    "horizon_score_weights": {"5": 1.0},
    "score_penalty_columns": [],
    "score_penalty_weight": 0.0,
    "use_ml_score": False,
    "ml_model_type": "xgboost",
    "ml_model_path": "",
    "ml_score_weight": 0.3,
    "original_score_weight": 0.7,
    "ml_label_type": "high_5d_up",
    "ml_threshold_up": 0.05,
    "ranker_identity": "",
    "research_audit_required": False,
    "audit_result": None,
    "candidate_pool_top_ns": [20, 10, 5],
    "system_mode": "stable_observation_cycle",
    "main_score_frozen": True,
    "execution_layer_frozen": True,
    "observation_layer_frozen": True,
    "allowed_change_scope": ["bugfix", "monitoring", "quality_dashboard", "log_report_structure"],
    "new_research_gate_passed": False,
    "new_research_gate_reason": "does not improve candidate pool quality or action clarity under the stable observation cycle",
    "system_layers": {
        "observation": "monitor / observation pool",
        "strategy": "tradable strategy / deployable prototype",
    },
    "universe_max_codes": 500,
    "regime_gate_enabled": False,
    "regime_breadth_lookback": 10,
    "regime_breadth_full_risk": 0.52,
    "regime_breadth_low_risk": 0.45,
    "regime_market_vol_lookback": 20,
    "regime_market_vol_half_quantile": 0.60,
    "regime_market_vol_low_quantile": 0.80,
    "regime_style_lookback": 20,
    "regime_style_half_threshold": -0.0005,
    "regime_style_low_threshold": -0.0020,
    "regime_half_risk_multiplier": 0.50,
    "regime_low_risk_multiplier": 0.0,
    "premium_dynamic_constraints_enabled": False,
    "premium_entry_filter_smart_money_enabled": False,
    "premium_entry_filter_min_smart_money_inflow": float("-inf"),
    "premium_entry_filter_profit_warning_enabled": False,
    "premium_entry_filter_distress_risk_enabled": False,
    "premium_limit_sentiment_gate_enabled": False,
    "premium_limit_sentiment_half_threshold": 0.0,
    "premium_limit_sentiment_low_threshold": -0.20,
    "premium_half_risk_multiplier": 0.50,
    "premium_low_risk_multiplier": 0.0,
    "enable_fundamental_veto": False,
    "min_profit_quality": -0.10,
    "min_earnings_growth": -0.15,
    "min_cashflow_quality": -0.10,
    "min_roe": -0.05,
    "max_debt_to_asset": 0.75,
    "min_overhead_resistance": float("-inf"),
    "stop_loss_pct": 0.0,
    "take_profit_pct": 0.0,
    "trailing_stop_pct": 0.0,
}

DEFAULT_BACKTEST = {
    "commission": 0.0003,
    "slippage": 0.0005,
    "sell_tax": 0.001,
    "annual_trading_days": 252,
    "use_liquidity_aware_cost": True,
    "portfolio_notional": 10_000_000.0,
    "slippage_adv_coef": 0.1,
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_factor_columns(research_cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    factor_set = research_cfg.get("factor_set")
    explicit_requested = research_cfg.get("factor_columns")
    family_requested = research_cfg.get("factor_families") or []
    requested = resolve_family_factor_request(
        factor_families=list(family_requested) if isinstance(family_requested, (list, tuple)) else [family_requested],
        factor_columns=list(explicit_requested) if isinstance(explicit_requested, (list, tuple)) else explicit_requested,
    )
    if factor_set and str(factor_set).lower() == "custom" and not requested:
        raise ValueError("factor_set='custom' requires an explicit factor_columns list.")
    if (not requested) and factor_set and str(factor_set).lower() != "custom":
        requested = [f"{name}_neu" for name in factor_set_columns(str(factor_set))]
    factor_cols = normalize_requested_factor_columns(requested)
    raw_factor_cols = requested_raw_factor_columns(requested)
    validate_factor_usage(research_cfg=research_cfg, raw_factor_cols=raw_factor_cols)
    return raw_factor_cols, factor_cols


def classify_config_mode(research_cfg: dict[str, Any]) -> str:
    explicit = research_cfg.get("is_production_config")
    if explicit is True:
        return "production"
    if explicit is False:
        return "research"
    return "production" if infer_ranker_identity(research_cfg) == "production_primary_ranker" else "research"


def validate_factor_usage(*, research_cfg: dict[str, Any], raw_factor_cols: list[str]) -> None:
    if classify_config_mode(research_cfg) != "production":
        return
    blocked = blocked_production_factors(raw_factor_cols)
    if not blocked:
        return
    if bool(research_cfg.get("allow_low_freq_experimental_in_production", False)):
        warnings.warn(
            "production config override enabled for low-frequency experimental factors: "
            + ", ".join(sorted(blocked)),
            stacklevel=2,
        )
        return
    raise ValueError(
        "low-frequency experimental factors are blocked in production configs: "
        + ", ".join(sorted(blocked))
        + ". Set allow_low_freq_experimental_in_production=true only if you explicitly accept the visible-time risk."
    )


def resolve_label_horizons(research_cfg: dict[str, Any]) -> list[int]:
    if research_cfg.get("use_horizons"):
        return sorted({int(v) for v in research_cfg["use_horizons"] if int(v) > 0})
    if research_cfg.get("label_horizons"):
        return sorted({int(v) for v in research_cfg["label_horizons"] if int(v) > 0})
    return [int(research_cfg["label_horizon"])]


def build_execution_spec(research_cfg: dict[str, Any]) -> ExecutionSpec:
    holding_window = int(research_cfg.get("holding_period", research_cfg["holding_window"]))
    return ExecutionSpec(
        signal_time=str(research_cfg["signal_time"]),
        execution_price=str(research_cfg["execution_price"]),
        execution_lag=int(research_cfg["execution_lag"]),
        holding_window=holding_window,
    )


def build_factor_preprocess_config(research_cfg: dict[str, Any]) -> FactorPreprocessConfig:
    return FactorPreprocessConfig(
        clip_method=str(research_cfg["clip_method"]),
        mad_scale=float(research_cfg["mad_scale"]),
        quantile_lower=float(research_cfg["quantile_lower"]),
        quantile_upper=float(research_cfg["quantile_upper"]),
    )


def build_dynamic_weight_config(research_cfg: dict[str, Any]) -> DynamicWeightConfig:
    return DynamicWeightConfig(
        rolling_window=int(research_cfg["dynamic_window"]),
        ewma_span=int(research_cfg["dynamic_ewma_span"]),
        icir_threshold=float(research_cfg["dynamic_icir_threshold"]),
        sign_flip_threshold=float(research_cfg["dynamic_sign_flip_threshold"]),
        min_history=int(research_cfg["dynamic_min_history"]),
    )


def build_universe_filter_config(research_cfg: dict[str, Any]) -> UniverseFilterConfig:
    return UniverseFilterConfig(
        min_listing_days=int(research_cfg["min_listing_days"]),
        min_avg_amount_20=float(research_cfg["min_avg_amount_20"]),
        min_market_cap=float(research_cfg["min_market_cap"]),
        min_price=float(research_cfg["min_price"]),
        max_price=float(research_cfg.get("max_price", float("inf"))),
        near_limit_buffer=float(research_cfg["near_limit_buffer"]),
    )


def build_regime_gate_config(research_cfg: dict[str, Any]) -> RegimeGateConfig:
    return RegimeGateConfig(
        enabled=bool(research_cfg.get("regime_gate_enabled", False)),
        breadth_lookback=int(research_cfg.get("regime_breadth_lookback", 10)),
        breadth_full_risk=float(research_cfg.get("regime_breadth_full_risk", 0.52)),
        breadth_low_risk=float(research_cfg.get("regime_breadth_low_risk", 0.45)),
        market_vol_lookback=int(research_cfg.get("regime_market_vol_lookback", 20)),
        market_vol_half_quantile=float(research_cfg.get("regime_market_vol_half_quantile", 0.60)),
        market_vol_low_quantile=float(research_cfg.get("regime_market_vol_low_quantile", 0.80)),
        style_lookback=int(research_cfg.get("regime_style_lookback", 20)),
        style_half_threshold=float(research_cfg.get("regime_style_half_threshold", -0.0005)),
        style_low_threshold=float(research_cfg.get("regime_style_low_threshold", -0.0020)),
        half_risk_multiplier=float(research_cfg.get("regime_half_risk_multiplier", 0.50)),
        low_risk_multiplier=float(research_cfg.get("regime_low_risk_multiplier", 0.0)),
    )


def build_fundamental_veto_config(research_cfg: dict[str, Any]) -> FundamentalVetoConfig:
    return FundamentalVetoConfig(
        enabled=bool(research_cfg.get("enable_fundamental_veto", False)),
        min_profit_quality=float(research_cfg.get("min_profit_quality", -0.10)),
        min_earnings_growth=float(research_cfg.get("min_earnings_growth", -0.15)),
        min_cashflow_quality=float(research_cfg.get("min_cashflow_quality", -0.10)),
        min_roe=float(research_cfg.get("min_roe", -0.05)),
        max_debt_to_asset=float(research_cfg.get("max_debt_to_asset", 0.75)),
        min_overhead_resistance=float(research_cfg.get("min_overhead_resistance", float("-inf"))),
    )


def build_premium_constraint_config(research_cfg: dict[str, Any]) -> PremiumConstraintConfig:
    return PremiumConstraintConfig(
        enabled=bool(research_cfg.get("premium_dynamic_constraints_enabled", False)),
        entry_filter_smart_money_enabled=bool(research_cfg.get("premium_entry_filter_smart_money_enabled", False)),
        min_smart_money_inflow=float(research_cfg.get("premium_entry_filter_min_smart_money_inflow", float("-inf"))),
        entry_filter_profit_warning_enabled=bool(research_cfg.get("premium_entry_filter_profit_warning_enabled", False)),
        entry_filter_distress_risk_enabled=bool(research_cfg.get("premium_entry_filter_distress_risk_enabled", False)),
        market_gate_limit_sentiment_enabled=bool(research_cfg.get("premium_limit_sentiment_gate_enabled", False)),
        limit_sentiment_half_threshold=float(research_cfg.get("premium_limit_sentiment_half_threshold", 0.0)),
        limit_sentiment_low_threshold=float(research_cfg.get("premium_limit_sentiment_low_threshold", -0.20)),
        half_risk_multiplier=float(research_cfg.get("premium_half_risk_multiplier", 0.50)),
        low_risk_multiplier=float(research_cfg.get("premium_low_risk_multiplier", 0.0)),
    )


def _group_feature_columns(df: pd.DataFrame) -> list[str]:
    return [f"group_{name}_score" for name in FACTOR_GROUPS if f"group_{name}_score" in df.columns]


def resolve_horizon_feature_map(
    df: pd.DataFrame,
    research_cfg: dict[str, Any],
    raw_factor_cols: list[str],
) -> dict[int, list[str]]:
    horizons = resolve_label_horizons(research_cfg)
    if research_cfg.get("factor_combination_mode") == "group":
        group_cols = _group_feature_columns(df)
        mapping = {h: group_cols for h in horizons}
        custom = research_cfg.get("horizon_group_map") or {}
        for key, group_names in custom.items():
            horizon = int(key)
            cols = [f"group_{name}_score" for name in group_names if f"group_{name}_score" in df.columns]
            if cols:
                mapping[horizon] = cols
        return mapping

    custom = research_cfg.get("horizon_factor_map") or {}
    default_map = default_horizon_factor_map()
    mapping: dict[int, list[str]] = {}
    for horizon in horizons:
        raw_names = custom.get(str(horizon)) or custom.get(horizon) or default_map.get(horizon) or raw_factor_cols
        cols = [f"{name}_neu" for name in raw_names if f"{name}_neu" in df.columns]
        if not cols:
            cols = [f"{name}_neu" for name in raw_factor_cols if f"{name}_neu" in df.columns]
        mapping[horizon] = cols
    return mapping


def infer_ranker_identity(research_cfg: dict[str, Any]) -> str:
    explicit = str(research_cfg.get("ranker_identity", "")).strip()
    if explicit:
        return explicit
    if bool(research_cfg.get("use_ml_score", False)):
        return "observation_score"
    if str(research_cfg.get("signal_mode", "factor")).lower() == "ml":
        return "research_ranker"
    if str(research_cfg.get("ranker_type", "")).lower() in {"ml_ridge", "ml_logistic", "xgboost", "lightgbm"}:
        return "research_ranker"
    return "production_primary_ranker"


def assign_role_assignment(
    *,
    research_status: str = "",
    has_system_alpha: bool = False,
    has_explanatory_value: bool = False,
    actionability_clear: bool = False,
    cross_time_stable: bool = False,
    config_mode: str = "research",
    ranker_identity: str = "",
) -> str:
    if research_status == "reject":
        return "archived_reject"
    if has_system_alpha:
        return "production_core_factor"
    if actionability_clear and cross_time_stable:
        return "execution_rule_candidate"
    if config_mode == "production" and ranker_identity == "production_primary_ranker":
        return "production_core_factor"
    if has_explanatory_value or ranker_identity == "observation_score":
        return "observation_label"
    return "research_factor"


def build_audit_summary(research_cfg: dict[str, Any]) -> dict[str, Any]:
    required = bool(research_cfg.get("research_audit_required", False))
    raw_audit = research_cfg.get("audit_result") or {}
    if not isinstance(raw_audit, dict):
        raw_audit = {"summary": str(raw_audit)}
    if not required:
        return {
            "audit_required": False,
            "audit_status": "not_required",
            "audit_summary": str(raw_audit.get("summary", "research audit not required")),
            "audit_warnings_count": int(raw_audit.get("warnings_count", 0) or 0),
            "audit_errors_count": int(raw_audit.get("errors_count", 0) or 0),
            "upgrade_recommendation": "research can continue without explicit audit gate",
        }
    if not raw_audit:
        warnings.warn("research_audit_required=true but no audit_result was provided", stacklevel=2)
        return {
            "audit_required": True,
            "audit_status": "missing",
            "audit_summary": "audit required but missing",
            "audit_warnings_count": 1,
            "audit_errors_count": 0,
            "upgrade_recommendation": "do not upgrade beyond research config until audit is attached",
        }

    errors = int(raw_audit.get("errors_count", 0) or 0)
    warnings_count = int(raw_audit.get("warnings_count", 0) or 0)
    stated_status = str(raw_audit.get("status") or raw_audit.get("overall_status") or "").strip().lower()
    if errors > 0:
        status = "failed"
    elif warnings_count > 0:
        status = "warning"
    elif stated_status in {"passed", "pass", "ok"}:
        status = "passed"
    elif stated_status in {"failed", "fail", "error"}:
        status = "failed"
    elif stated_status in {"warning", "warn"}:
        status = "warning"
    else:
        status = "passed"
    return {
        "audit_required": True,
        "audit_status": status,
        "audit_summary": str(raw_audit.get("summary", raw_audit.get("message", "audit result attached"))),
        "audit_warnings_count": warnings_count,
        "audit_errors_count": errors,
        "upgrade_recommendation": (
            "eligible for next-stage research only after warning review"
            if status == "warning"
            else "do not upgrade until audit failures are fixed"
            if status == "failed"
            else "eligible for next-stage research or candidate config review"
        ),
    }


def build_research_summary(
    *,
    research_cfg: dict[str, Any],
    raw_factor_cols: list[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family_summary = summarize_factor_families(raw_factor_cols)
    execution_lag = int(research_cfg.get("execution_lag", 0))
    audit_gate = build_audit_summary(research_cfg)
    ranker_identity = infer_ranker_identity(research_cfg)
    config_mode = classify_config_mode(research_cfg)
    system_mode = str(research_cfg.get("system_mode", "stable_observation_cycle")).strip() or "stable_observation_cycle"
    new_research_gate_passed = bool(research_cfg.get("new_research_gate_passed", False))
    new_research_gate_reason = str(
        research_cfg.get("new_research_gate_reason")
        or "does not improve candidate pool quality or action clarity"
    ).strip()
    return {
        "factor_families": family_summary["families"],
        "factor_family_ratio": family_summary["family_ratio"],
        "contains_low_freq_experimental": family_summary["contains_low_freq_experimental"],
        "low_freq_experimental_factors": family_summary["low_freq_experimental_factors"],
        "label_type": str(research_cfg.get("label_type", "")),
        "label_horizons": list(metadata.get("label_horizons", [])) if metadata else resolve_label_horizons(research_cfg),
        "config_mode": config_mode,
        "system_mode": system_mode,
        "main_score_frozen": bool(research_cfg.get("main_score_frozen", False)),
        "execution_layer_frozen": bool(research_cfg.get("execution_layer_frozen", False)),
        "observation_layer_frozen": bool(research_cfg.get("observation_layer_frozen", False)),
        "allowed_change_scope": list(research_cfg.get("allowed_change_scope", [])),
        "new_research_gate_passed": new_research_gate_passed,
        "new_research_gate_reason": new_research_gate_reason,
        "next_day_execution_checked": execution_lag >= 1,
        "research_audit_passed": audit_gate["audit_status"] == "passed",
        "ranker_identity": ranker_identity,
        "role_assignment": assign_role_assignment(
            config_mode=config_mode,
            ranker_identity=ranker_identity,
            has_explanatory_value=ranker_identity == "observation_score",
        ),
        **audit_gate,
    }


def build_candidate_pool_quality_payload(
    scored_df: pd.DataFrame,
    *,
    research_cfg: dict[str, Any],
    selection_dates: set[pd.Timestamp] | None = None,
) -> dict[str, Any]:
    holding_horizon = int(research_cfg.get("holding_period", research_cfg.get("holding_window", 5)))
    failure_horizon = int(research_cfg.get("candidate_pool_failure_horizon", min(holding_horizon, 5)))
    top_ns = [int(value) for value in research_cfg.get("candidate_pool_top_ns", [20, 10, 5])]
    frame = scored_df.copy()
    frame = add_candidate_path_columns(frame, horizons={holding_horizon, failure_horizon})
    if selection_dates:
        allowed_dates = {pd.Timestamp(value) for value in selection_dates}
        frame = frame.loc[frame["date"].isin(allowed_dates)].copy()
    summary_df = summarize_candidate_pool_quality(
        frame,
        score_col="score",
        eligibility_col="strategy_tradeable",
        top_ns=top_ns,
        holding_horizon=holding_horizon,
    )
    failure_df = summarize_failure_attribution(
        frame,
        score_col="score",
        eligibility_col="strategy_tradeable",
        top_ns=top_ns,
        evaluation_horizon=failure_horizon,
    )
    if summary_df.empty:
        return {
            "holding_horizon": holding_horizon,
            "summary_rows": [],
            "failure_attribution_rows": [],
            "headline": "候选池质量暂无有效样本。",
        }
    summary_rows = summary_df.to_dict(orient="records")
    best_row = summary_df.sort_values(["avg_return", "win_rate"], ascending=False).iloc[0]
    headline = (
        f"候选池质量先看 top{int(best_row['top_n'])}："
        f"平均收益 {float(best_row['avg_return']):.4f}，"
        f"命中率 {float(best_row['hit_rate']):.2%}，"
        f"胜率 {float(best_row['win_rate']):.2%}，"
        f"平均回撤 {float(best_row['avg_drawdown']):.4f}。"
    )
    return {
        "holding_horizon": holding_horizon,
        "summary_rows": summary_rows,
        "failure_attribution_rows": failure_df.to_dict(orient="records"),
        "headline": headline,
    }


def _fill_market_cap_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill market cap gaps caused by late daily_basic availability."""

    if "market_cap" not in df.columns:
        return df
    out = df.sort_values(["code", "date"]).copy()
    market_cap = pd.to_numeric(out["market_cap"], errors="coerce")
    out["market_cap"] = market_cap.groupby(out["code"].astype(str)).ffill()
    return out


def prepare_research_frame(df: pd.DataFrame, research_cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_factor_cols, factor_cols = resolve_factor_columns(research_cfg)
    df = ensure_price_views(df)
    df = _fill_market_cap_gaps(df)
    df, universe_summary = apply_basic_universe_filters(
        df,
        config=build_universe_filter_config(research_cfg),
        return_summary=True,
    )
    df["monitor_eligible"] = df["tradeable"].fillna(False).astype(bool)
    df = add_technical_factors(
        df,
        include_technical=bool(research_cfg["include_technical"]),
        include_fundamental=bool(research_cfg["include_fundamental"]),
        include_flow=bool(research_cfg["include_flow"]),
        include_regime=bool(research_cfg["include_regime"]),
    )
    df = neutralize_by_size_and_industry(
        df,
        raw_factor_cols,
        preprocess_config=build_factor_preprocess_config(research_cfg),
    )
    df = add_group_scores(df, raw_factor_cols, use_neutralized=True)
    label_horizons = resolve_label_horizons(research_cfg)
    requested_label_types = [str(research_cfg["label_type"])]
    if bool(research_cfg.get("use_ml_score", False)):
        requested_label_types.append(str(research_cfg.get("ml_label_type", "high_5d_up")))
    df = add_label_columns(
        df,
        horizons=label_horizons,
        spec=build_execution_spec(research_cfg),
        label_types=requested_label_types,
        binary_up_threshold=float(research_cfg.get("ml_threshold_up", 0.05)),
    )
    df, veto_summary = apply_fundamental_veto(df, config=build_fundamental_veto_config(research_cfg))
    df["strategy_tradeable"] = df["monitor_eligible"] & ~df["fundamental_veto"].fillna(False).astype(bool)
    df, regime_summary = apply_regime_gate(df, config=build_regime_gate_config(research_cfg))
    df, premium_summary = apply_premium_dynamic_constraints(df, config=build_premium_constraint_config(research_cfg))
    horizon_feature_map = resolve_horizon_feature_map(df, research_cfg=research_cfg, raw_factor_cols=raw_factor_cols)
    metadata = {
        "raw_factor_cols": raw_factor_cols,
        "factor_cols": factor_cols,
        "label_horizons": label_horizons,
        "horizon_feature_map": horizon_feature_map,
        "label_type": research_cfg["label_type"],
        "execution_spec": build_execution_spec(research_cfg),
        "universe_summary": universe_summary,
        "fundamental_veto_summary": veto_summary,
        "regime_gate_summary": regime_summary,
        "premium_constraint_summary": premium_summary,
    }
    metadata["research_summary"] = build_research_summary(
        research_cfg=research_cfg,
        raw_factor_cols=raw_factor_cols,
        metadata=metadata,
    )
    metadata.update(
        {
            "audit_required": metadata["research_summary"]["audit_required"],
            "audit_status": metadata["research_summary"]["audit_status"],
            "audit_summary": metadata["research_summary"]["audit_summary"],
            "audit_warnings_count": metadata["research_summary"]["audit_warnings_count"],
            "audit_errors_count": metadata["research_summary"]["audit_errors_count"],
        }
    )
    # The downstream portfolio construction code slices by date many times; clearing
    # attrs here avoids repeated pandas deepcopy overhead on large real datasets.
    df.attrs = {}
    return df, metadata


def _label_col_for_horizon(research_cfg: dict[str, Any], horizon: int) -> str:
    return label_column_name(horizon, str(research_cfg["label_type"]))


def _available_date_col(horizon: int) -> str:
    return f"label_available_date_{horizon}d"


def restrict_train_dates(
    train_dates: set[pd.Timestamp],
    train_window_years: int,
) -> set[pd.Timestamp]:
    if not train_dates or int(train_window_years) <= 0:
        return train_dates
    max_train_date = max(pd.Timestamp(date) for date in train_dates)
    cutoff = max_train_date - pd.DateOffset(years=int(train_window_years))
    return {pd.Timestamp(date) for date in train_dates if pd.Timestamp(date) >= cutoff}


def _fit_ranker(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    available_date_col: str,
    research_cfg: dict[str, Any],
):
    signal_mode = str(research_cfg.get("signal_mode", "factor")).lower()
    if signal_mode == "ml" and str(research_cfg.get("model_path", "")).strip():
        # Online/backtest runtime must not fit. It may only load a previously
        # trained artifact and call transform/predict to avoid future leakage.
        return OfflineSignalPredictor.load(
            path=research_cfg["model_path"],
            expected_factor_cols=feature_cols,
        )
    if research_cfg["ranker_type"] == "static":
        ranker = ICWeightedRanker(factor_cols=feature_cols, label_col=label_col).fit(train_df)
    elif research_cfg["ranker_type"] == "fixed_weights":
        ranker = FixedWeightRanker(
            factor_cols=feature_cols,
            factor_weights=dict(research_cfg.get("fixed_factor_weights", {})),
        ).fit(train_df)
    elif research_cfg["ranker_type"] == "ml_ridge":
        ranker = MLRidgeRanker(
            factor_cols=feature_cols,
            label_col=label_col,
            config=MLRidgeConfig(
                alpha_grid=[float(v) for v in research_cfg.get("ml_alpha_grid", [0.1, 1.0, 10.0])],
                cv_folds=int(research_cfg.get("ml_cv_folds", 3)),
                min_rows=int(research_cfg.get("ml_min_rows", 200)),
            ),
        ).fit(train_df)
    elif research_cfg["ranker_type"] == "ml_logistic":
        ranker = MLLogisticRanker(
            factor_cols=feature_cols,
            label_col=label_col,
            config=MLLogisticConfig(
                c_grid=[float(v) for v in research_cfg.get("ml_c_grid", [0.1, 1.0, 10.0])],
                cv_folds=int(research_cfg.get("ml_cv_folds", 3)),
                min_rows=int(research_cfg.get("ml_min_rows", 200)),
                positive_threshold=float(research_cfg.get("ml_positive_threshold", 0.0)),
            ),
        ).fit(train_df)
    elif research_cfg["ranker_type"] in {"xgboost", "lightgbm"}:
        ranker = TreeBoostRanker(
            factor_cols=feature_cols,
            label_col=label_col,
            config=BoostRankerConfig(
                backend=str(research_cfg["ranker_type"]),
                cv_folds=int(research_cfg.get("ml_cv_folds", 3)),
                min_rows=int(research_cfg.get("ml_min_rows", 300)),
                learning_rates=[float(v) for v in research_cfg.get("boost_learning_rates", [0.05, 0.1])],
                n_estimators_grid=[int(v) for v in research_cfg.get("boost_n_estimators_grid", [100, 200])],
                max_depth_grid=[int(v) for v in research_cfg.get("boost_max_depth_grid", [3, 5])],
                allow_sklearn_fallback=bool(research_cfg.get("boost_allow_sklearn_fallback", True)),
            ),
        ).fit(train_df)
    else:
        ranker = RollingICIRRanker(
            factor_cols=feature_cols,
            label_col=label_col,
            available_date_col=available_date_col,
            config=build_dynamic_weight_config(research_cfg),
        ).fit(df.loc[df[label_col].notna()].copy())
    return ranker


def _score_frame_by_horizon(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    research_cfg: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    horizon_weights_cfg = research_cfg.get("horizon_score_weights") or {}
    score_details: dict[str, Any] = {"horizon_models": {}, "horizon_weight_history": {}}
    horizons = metadata["label_horizons"]
    raw_horizon_strengths: dict[int, pd.Series] = {}

    for horizon in horizons:
        label_col = _label_col_for_horizon(research_cfg, horizon)
        available_date_col = _available_date_col(horizon)
        feature_cols = metadata["horizon_feature_map"][horizon]
        labeled_train = train_df.loc[train_df[label_col].notna()].copy()
        if labeled_train.empty:
            out[f"score_{horizon}d"] = 0.0
            continue
        ranker = _fit_ranker(
            df=df,
            train_df=labeled_train,
            feature_cols=feature_cols,
            label_col=label_col,
            available_date_col=available_date_col,
            research_cfg=research_cfg,
        )
        out[f"score_{horizon}d"] = ranker.predict(out)
        score_details["horizon_models"][str(horizon)] = {
            "label_col": label_col,
            "feature_cols": feature_cols,
            "weights": getattr(ranker, "weights_", {}),
        }
        if hasattr(ranker, "raw_strength_history_") and not ranker.raw_strength_history_.empty:
            strength_df = ranker.raw_strength_history_.copy()
            feature_cols_present = [col for col in feature_cols if col in strength_df.columns]
            if feature_cols_present:
                raw_horizon_strengths[horizon] = strength_df.set_index("date")[feature_cols_present].abs().sum(axis=1)

    if len(horizons) == 1:
        only = horizons[0]
        out["score"] = out[f"score_{only}d"]
        score_details["final_horizon_weights"] = {str(only): 1.0}
    elif research_cfg.get("dynamic_horizon_weighting") and raw_horizon_strengths:
        weight_history = []
        for date in sorted(out["date"].drop_duplicates()):
            strengths = {str(h): float(raw_horizon_strengths.get(h, pd.Series(dtype=float)).get(date, 0.0)) for h in horizons}
            denom = sum(max(v, 0.0) for v in strengths.values())
            if denom <= 0:
                weights = {str(h): 1.0 / len(horizons) for h in horizons}
            else:
                weights = {key: max(value, 0.0) / denom for key, value in strengths.items()}
            weight_history.append({"date": pd.Timestamp(date), **weights})
            mask = out["date"] == date
            out.loc[mask, "score"] = 0.0
            for h in horizons:
                out.loc[mask, "score"] += out.loc[mask, f"score_{h}d"] * weights[str(h)]
        history_df = pd.DataFrame(weight_history)
        weight_history_records = []
        for record in history_df.to_dict(orient="records"):
            record["date"] = pd.Timestamp(record["date"]).strftime("%Y-%m-%d")
            weight_history_records.append(record)
        score_details["horizon_weight_history"] = weight_history_records
        score_details["final_horizon_weights"] = (
            history_df.iloc[-1].drop(labels=["date"]).to_dict()
            if not history_df.empty
            else {}
        )
    else:
        raw_weights = {str(h): float(horizon_weights_cfg.get(str(h), horizon_weights_cfg.get(h, 1.0))) for h in horizons}
        denom = sum(abs(v) for v in raw_weights.values()) or 1.0
        final_weights = {key: value / denom for key, value in raw_weights.items()}
        out["score"] = 0.0
        for h in horizons:
            out["score"] += out[f"score_{h}d"] * final_weights[str(h)]
        score_details["final_horizon_weights"] = final_weights

    penalty_cols = normalize_requested_factor_columns(research_cfg.get("score_penalty_columns") or [])
    penalty_cols = [col for col in penalty_cols if col in out.columns]
    penalty_weight = float(research_cfg.get("score_penalty_weight", 0.0))
    if penalty_cols and abs(penalty_weight) > 0:
        penalty = out[penalty_cols].fillna(0.0).mean(axis=1)
        out["score_raw"] = out["score"]
        out["score_penalty"] = penalty
        out["score"] = out["score"] - penalty_weight * penalty
        score_details["score_penalty"] = {
            "columns": penalty_cols,
            "weight": penalty_weight,
        }
    if bool(research_cfg.get("use_ml_score", False)) and str(research_cfg.get("ml_model_path", "")).strip():
        predictor = OfflineSignalPredictor.load(
            path=research_cfg["ml_model_path"],
            expected_factor_cols=metadata["factor_cols"] or None,
        )
        out["score_original"] = out["score"]
        out["ml_score"] = predictor.predict(out)
        original_weight = float(research_cfg.get("original_score_weight", 0.7))
        ml_weight = float(research_cfg.get("ml_score_weight", 0.3))
        denom = abs(original_weight) + abs(ml_weight)
        if denom <= 0:
            raise ValueError("original_score_weight and ml_score_weight cannot both be zero.")
        out["score"] = (original_weight * out["score_original"] + ml_weight * out["ml_score"]) / denom
        score_details["ml_score"] = {
            "enabled": True,
            "model_type": str(research_cfg.get("ml_model_type", "")),
            "model_path": str(research_cfg.get("ml_model_path", "")),
            "prediction_type": str(predictor.prediction_type),
            "original_score_weight": original_weight,
            "ml_score_weight": ml_weight,
            "label_type": str(research_cfg.get("ml_label_type", "high_5d_up")),
            "threshold_up": float(research_cfg.get("ml_threshold_up", 0.05)),
        }
    return out, score_details


def score_research_frame(
    df: pd.DataFrame,
    research_cfg: dict[str, Any],
    metadata: dict[str, Any],
    train_dates: set[pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    effective_train_dates = restrict_train_dates(
        train_dates=train_dates,
        train_window_years=int(research_cfg.get("train_window_years", 0)),
    )
    train_df = df[df["date"].isin(effective_train_dates)].copy()
    return _score_frame_by_horizon(
        df=df,
        train_df=train_df,
        research_cfg=research_cfg,
        metadata=metadata,
    )


def run_research_pipeline(
    data_path: str | Path,
    research_config_path: str | Path,
    backtest_config_path: str | Path,
    data_source: MarketDataSource | None = None,
    data_adjust: str = "none",
) -> tuple[pd.DataFrame, dict, dict[str, Any], pd.DataFrame]:
    research_cfg = {**DEFAULT_RESEARCH, **load_json(research_config_path)}
    backtest_cfg = {**DEFAULT_BACKTEST, **load_json(backtest_config_path)}

    if data_source is not None:
        df = ensure_price_views(data_source.load_daily_bars())
    else:
        df = CSVDataSource(data_path, adjust=data_adjust).load()
    df, metadata = prepare_research_frame(df, research_cfg)

    dates = sorted(df["date"].drop_duplicates())
    split = int(len(dates) * 0.6)
    full_train_dates = set(dates[:split])
    train_dates = restrict_train_dates(
        train_dates=full_train_dates,
        train_window_years=int(research_cfg.get("train_window_years", 0)),
    )
    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[~df["date"].isin(full_train_dates)].copy()

    scored_df, score_details = score_research_frame(
        df=df,
        research_cfg=research_cfg,
        metadata=metadata,
        train_dates=train_dates,
    )
    test_df = scored_df[~scored_df["date"].isin(full_train_dates)].copy()
    candidate_pool_quality = build_candidate_pool_quality_payload(
        scored_df,
        research_cfg=research_cfg,
        selection_dates=set(test_df["date"].drop_duplicates()),
    )

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

    backtester = DailyBacktester(
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
    )
    result, metrics = backtester.run(test_df, targets)
    metrics["universe_filter_summary"] = metadata["universe_summary"].__dict__
    metrics["fundamental_veto_summary"] = metadata["fundamental_veto_summary"].__dict__
    metrics["regime_gate_summary"] = metadata["regime_gate_summary"].__dict__
    metrics["premium_constraint_summary"] = metadata["premium_constraint_summary"].__dict__
    metrics["score_details"] = score_details
    metrics["research_summary"] = metadata.get("research_summary", {})
    metrics["candidate_pool_quality"] = candidate_pool_quality
    metrics["candidate_pool_quality_headline"] = candidate_pool_quality.get("headline", "")
    metrics["audit_required"] = metadata.get("audit_required", False)
    metrics["audit_status"] = metadata.get("audit_status", "not_required")
    metrics["audit_summary"] = metadata.get("audit_summary", "")
    metrics["audit_warnings_count"] = metadata.get("audit_warnings_count", 0)
    metrics["audit_errors_count"] = metadata.get("audit_errors_count", 0)
    metrics["system_mode"] = metrics["research_summary"].get("system_mode", "")
    metrics["main_score_frozen"] = metrics["research_summary"].get("main_score_frozen", False)
    metrics["execution_layer_frozen"] = metrics["research_summary"].get("execution_layer_frozen", False)
    metrics["observation_layer_frozen"] = metrics["research_summary"].get("observation_layer_frozen", False)
    metrics["new_research_gate_passed"] = metrics["research_summary"].get("new_research_gate_passed", False)
    metrics["new_research_gate_reason"] = metrics["research_summary"].get("new_research_gate_reason", "")
    metrics["research_summary"]["candidate_pool_quality"] = candidate_pool_quality
    metrics["research_summary"]["candidate_pool_quality_headline"] = candidate_pool_quality.get("headline", "")
    latest_target_date = targets["signal_date"].max() if not targets.empty and "signal_date" in targets.columns else pd.NaT
    latest_targets = (
        targets.loc[targets["signal_date"] == latest_target_date].copy()
        if pd.notna(latest_target_date)
        else pd.DataFrame()
    )
    metrics["strategy_assessment"] = assessment_payload(metrics, latest_targets)
    return result, metrics, score_details, targets
