from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class FundamentalVetoConfig:
    enabled: bool = False
    min_profit_quality: float = -0.10
    min_earnings_growth: float = -0.15
    min_cashflow_quality: float = -0.10
    min_roe: float = -0.05
    max_debt_to_asset: float = 0.75
    min_overhead_resistance: float = float("-inf")


@dataclass(frozen=True)
class FundamentalVetoSummary:
    enabled: bool
    vetoed_rows: int
    veto_ratio: float
    missing_signal_rows: int
    reason_counts: dict[str, int]


def _optional_numeric(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def _optional_bool(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[name].fillna(False).astype(bool)


def apply_fundamental_veto(
    df: pd.DataFrame,
    config: FundamentalVetoConfig | None = None,
) -> tuple[pd.DataFrame, FundamentalVetoSummary]:
    cfg = config or FundamentalVetoConfig()
    out = df.copy()
    out["fundamental_veto"] = False
    out["fundamental_veto_reason"] = ""

    if not cfg.enabled:
        summary = FundamentalVetoSummary(
            enabled=False,
            vetoed_rows=0,
            veto_ratio=0.0,
            missing_signal_rows=int(len(out)),
            reason_counts={},
        )
        out.attrs["fundamental_veto_summary"] = asdict(summary)
        return out, summary

    signals = {
        "profit_quality": _optional_numeric(out, "profit_quality_factor"),
        "earnings_growth": _optional_numeric(out, "earnings_growth_factor"),
        "cashflow_quality": _optional_numeric(out, "cashflow_quality_factor"),
        "roe": _optional_numeric(out, "roe_factor"),
        "debt_to_asset": _optional_numeric(out, "debt_to_asset"),
        "overhead_resistance": _optional_numeric(out, "overhead_resistance"),
    }
    risk_flags = {
        "distress_risk": _optional_bool(out, "distress_risk_flag"),
        "profit_warning": _optional_bool(out, "profit_warning_flag"),
    }

    reason_masks = {
        "profit_quality": signals["profit_quality"].lt(cfg.min_profit_quality),
        "earnings_growth": signals["earnings_growth"].lt(cfg.min_earnings_growth),
        "cashflow_quality": signals["cashflow_quality"].lt(cfg.min_cashflow_quality),
        "roe": signals["roe"].lt(cfg.min_roe),
        "high_leverage": signals["debt_to_asset"].gt(cfg.max_debt_to_asset),
        "overhead_resistance": signals["overhead_resistance"].lt(cfg.min_overhead_resistance),
        "distress_risk": risk_flags["distress_risk"],
        "profit_warning": risk_flags["profit_warning"],
    }

    veto_mask = pd.Series(False, index=out.index, dtype=bool)
    reason_counts: dict[str, int] = {}
    for reason, mask in reason_masks.items():
        valid_mask = mask.fillna(False).astype(bool)
        reason_counts[reason] = int(valid_mask.sum())
        veto_mask |= valid_mask
    reasons = pd.Series("", index=out.index, dtype="object")
    for idx in out.index[veto_mask]:
        row_reasons = [reason for reason, mask in reason_masks.items() if bool(mask.fillna(False).astype(bool).loc[idx])]
        reasons.loc[idx] = "|".join(row_reasons)

    out["fundamental_veto"] = veto_mask
    out["fundamental_veto_reason"] = reasons.str.strip("|")

    observed_signals = pd.concat([*signals.values(), *risk_flags.values()], axis=1)
    missing_signal_rows = int(observed_signals.isna().all(axis=1).sum())
    summary = FundamentalVetoSummary(
        enabled=True,
        vetoed_rows=int(veto_mask.sum()),
        veto_ratio=float(veto_mask.mean()) if len(out) else 0.0,
        missing_signal_rows=missing_signal_rows,
        reason_counts={key: value for key, value in reason_counts.items() if value > 0},
    )
    out.attrs["fundamental_veto_summary"] = asdict(summary)
    return out, summary
