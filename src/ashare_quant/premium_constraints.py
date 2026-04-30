from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PremiumConstraintConfig:
    enabled: bool = False
    entry_filter_smart_money_enabled: bool = False
    min_smart_money_inflow: float = float("-inf")
    entry_filter_profit_warning_enabled: bool = False
    entry_filter_distress_risk_enabled: bool = False
    market_gate_limit_sentiment_enabled: bool = False
    limit_sentiment_half_threshold: float = 0.0
    limit_sentiment_low_threshold: float = -0.2
    half_risk_multiplier: float = 0.50
    low_risk_multiplier: float = 0.0


@dataclass(frozen=True)
class PremiumConstraintSummary:
    enabled: bool
    blocked_rows: int
    blocked_ratio: float
    reason_counts: dict[str, int]
    full_risk_days: int
    half_risk_days: int
    low_risk_days: int
    avg_risk_multiplier: float


def _optional_numeric(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def _optional_bool(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[name].fillna(False).astype(bool)


def _risk_order(series: pd.Series) -> pd.Series:
    order = {"low_risk": 0, "half_risk": 1, "full_risk": 2}
    return series.map(order).fillna(order["full_risk"]).astype(int)


def _combine_risk_layers(out: pd.DataFrame) -> pd.DataFrame:
    base_state = out.get("risk_state", pd.Series("full_risk", index=out.index)).fillna("full_risk").astype(str)
    base_mult = pd.to_numeric(out.get("risk_multiplier", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
    premium_state = out.get("premium_risk_state", pd.Series("full_risk", index=out.index)).fillna("full_risk").astype(str)
    premium_mult = pd.to_numeric(
        out.get("premium_risk_multiplier", pd.Series(1.0, index=out.index)),
        errors="coerce",
    ).fillna(1.0)

    out["risk_state_base"] = base_state
    out["risk_multiplier_base"] = base_mult
    out["risk_state_premium"] = premium_state
    out["risk_multiplier_premium"] = premium_mult

    base_order = _risk_order(base_state)
    premium_order = _risk_order(premium_state)
    stricter_is_base = base_order.le(premium_order)
    out["risk_state"] = np.where(stricter_is_base, base_state, premium_state)
    out["risk_multiplier"] = np.minimum(base_mult, premium_mult)

    zero_mask = out["risk_multiplier"].le(1e-12)
    out.loc[zero_mask, "risk_state"] = "low_risk"
    return out


def apply_premium_dynamic_constraints(
    df: pd.DataFrame,
    config: PremiumConstraintConfig | None = None,
) -> tuple[pd.DataFrame, PremiumConstraintSummary]:
    cfg = config or PremiumConstraintConfig()
    out = df.copy()
    out["premium_entry_filter"] = False
    out["premium_entry_filter_reason"] = ""
    out["premium_risk_state"] = "full_risk"
    out["premium_risk_multiplier"] = 1.0

    if not cfg.enabled:
        out = _combine_risk_layers(out)
        summary = PremiumConstraintSummary(
            enabled=False,
            blocked_rows=0,
            blocked_ratio=0.0,
            reason_counts={},
            full_risk_days=int(out["date"].nunique()),
            half_risk_days=0,
            low_risk_days=0,
            avg_risk_multiplier=float(pd.to_numeric(out["risk_multiplier"], errors="coerce").fillna(1.0).mean())
            if len(out)
            else 1.0,
        )
        out.attrs["premium_constraint_summary"] = asdict(summary)
        return out, summary

    smart_money = _optional_numeric(out, "smart_money_inflow_20")
    risk_flags = {
        "profit_warning": _optional_bool(out, "profit_warning_flag"),
        "distress_risk": _optional_bool(out, "distress_risk_flag"),
    }

    reason_masks: dict[str, pd.Series] = {}
    if cfg.entry_filter_smart_money_enabled:
        reason_masks["weak_smart_money"] = smart_money.lt(cfg.min_smart_money_inflow)
    if cfg.entry_filter_profit_warning_enabled:
        reason_masks["profit_warning"] = risk_flags["profit_warning"]
    if cfg.entry_filter_distress_risk_enabled:
        reason_masks["distress_risk"] = risk_flags["distress_risk"]

    blocked_mask = pd.Series(False, index=out.index, dtype=bool)
    reason_counts: dict[str, int] = {}
    for reason, mask in reason_masks.items():
        clean = mask.fillna(False).astype(bool)
        reason_counts[reason] = int(clean.sum())
        blocked_mask |= clean

    out["premium_entry_filter"] = blocked_mask
    reasons = pd.Series("", index=out.index, dtype="object")
    for idx in out.index[blocked_mask]:
        row_reasons = [reason for reason, mask in reason_masks.items() if bool(mask.fillna(False).astype(bool).loc[idx])]
        reasons.loc[idx] = "|".join(row_reasons)
    out["premium_entry_filter_reason"] = reasons.str.strip("|")

    if cfg.market_gate_limit_sentiment_enabled and "market_limit_sentiment_score" in out.columns:
        daily_score = (
            out[["date", "market_limit_sentiment_score"]]
            .dropna(subset=["date"])
            .drop_duplicates(subset=["date"], keep="last")
            .copy()
        )
        if not daily_score.empty:
            sentiment = pd.to_numeric(daily_score["market_limit_sentiment_score"], errors="coerce")
            states = np.where(
                sentiment.le(cfg.limit_sentiment_low_threshold),
                "low_risk",
                np.where(sentiment.le(cfg.limit_sentiment_half_threshold), "half_risk", "full_risk"),
            )
            multipliers = np.where(
                sentiment.le(cfg.limit_sentiment_low_threshold),
                float(cfg.low_risk_multiplier),
                np.where(sentiment.le(cfg.limit_sentiment_half_threshold), float(cfg.half_risk_multiplier), 1.0),
            )
            daily_score["premium_risk_state"] = states
            daily_score["premium_risk_multiplier"] = multipliers
            out = out.merge(
                daily_score[["date", "premium_risk_state", "premium_risk_multiplier"]],
                on="date",
                how="left",
                suffixes=("", "_premium"),
            )
            if "premium_risk_state_premium" in out.columns:
                out["premium_risk_state"] = out["premium_risk_state_premium"].combine_first(out["premium_risk_state"])
                out = out.drop(columns=["premium_risk_state_premium"])
            if "premium_risk_multiplier_premium" in out.columns:
                out["premium_risk_multiplier"] = pd.to_numeric(
                    out["premium_risk_multiplier_premium"],
                    errors="coerce",
                ).combine_first(pd.to_numeric(out["premium_risk_multiplier"], errors="coerce"))
                out = out.drop(columns=["premium_risk_multiplier_premium"])

    out["premium_risk_state"] = out["premium_risk_state"].fillna("full_risk")
    out["premium_risk_multiplier"] = pd.to_numeric(out["premium_risk_multiplier"], errors="coerce").fillna(1.0)
    out = _combine_risk_layers(out)

    premium_state_counts = out.groupby("date")["premium_risk_state"].first().value_counts().to_dict()
    summary = PremiumConstraintSummary(
        enabled=True,
        blocked_rows=int(blocked_mask.sum()),
        blocked_ratio=float(blocked_mask.mean()) if len(out) else 0.0,
        reason_counts={key: value for key, value in reason_counts.items() if value > 0},
        full_risk_days=int(premium_state_counts.get("full_risk", 0)),
        half_risk_days=int(premium_state_counts.get("half_risk", 0)),
        low_risk_days=int(premium_state_counts.get("low_risk", 0)),
        avg_risk_multiplier=float(out.groupby("date")["premium_risk_multiplier"].first().mean()) if len(out) else 1.0,
    )
    out.attrs["premium_constraint_summary"] = asdict(summary)
    return out, summary
