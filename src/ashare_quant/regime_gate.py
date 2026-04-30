from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeGateConfig:
    enabled: bool = False
    breadth_lookback: int = 10
    breadth_full_risk: float = 0.52
    breadth_low_risk: float = 0.45
    market_vol_lookback: int = 20
    market_vol_half_quantile: float = 0.60
    market_vol_low_quantile: float = 0.80
    style_lookback: int = 20
    style_half_threshold: float = -0.0005
    style_low_threshold: float = -0.0020
    half_risk_multiplier: float = 0.50
    low_risk_multiplier: float = 0.0


@dataclass(frozen=True)
class RegimeGateSummary:
    enabled: bool
    full_risk_days: int
    half_risk_days: int
    low_risk_days: int
    avg_risk_multiplier: float


def _price_col(df: pd.DataFrame, base: str) -> str:
    research_col = f"research_{base}"
    return research_col if research_col in df.columns else base


def apply_regime_gate(
    df: pd.DataFrame,
    config: RegimeGateConfig | None = None,
) -> tuple[pd.DataFrame, RegimeGateSummary]:
    cfg = config or RegimeGateConfig()
    out = df.copy()
    out["market_breadth_lookback"] = np.nan
    out["market_vol_lookback"] = np.nan
    out["style_small_vs_large_lookback"] = np.nan

    if not cfg.enabled:
        summary = RegimeGateSummary(
            enabled=False,
            full_risk_days=int(out["date"].nunique()),
            half_risk_days=0,
            low_risk_days=0,
            avg_risk_multiplier=1.0,
        )
        out.attrs["regime_gate_summary"] = asdict(summary)
        return out, summary

    close_col = _price_col(out, "close")
    market = out.sort_values(["code", "date"]).copy()
    market["ret_1"] = market.groupby("code")[close_col].pct_change()

    breadth = market.groupby("date")["ret_1"].apply(lambda s: float((s > 0).mean()) if len(s.dropna()) else np.nan)
    market_ret = market.groupby("date")["ret_1"].mean()

    size_df = market[["date", "code", "market_cap", "ret_1"]].copy()
    size_df["size_bucket"] = "Mid"
    for date, idx in size_df.groupby("date").groups.items():
        sl = size_df.loc[idx]
        valid = sl["market_cap"].notna()
        if valid.sum() < 6:
            continue
        rank_pct = sl.loc[valid, "market_cap"].rank(method="first", pct=True)
        buckets = pd.cut(
            rank_pct,
            bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=["Small", "Mid", "Large"],
            include_lowest=True,
        )
        size_df.loc[sl.index[valid], "size_bucket"] = buckets.astype(str)

    size_pivot = (
        size_df.dropna(subset=["ret_1"])
        .groupby(["date", "size_bucket"], as_index=False)["ret_1"]
        .mean()
        .pivot(index="date", columns="size_bucket", values="ret_1")
        .sort_index()
    )
    style_small_vs_large = size_pivot.get("Small", pd.Series(dtype=float)) - size_pivot.get("Large", pd.Series(dtype=float))

    regime_df = pd.DataFrame(
        {
            "date": pd.Index(sorted(out["date"].drop_duplicates()), name="date"),
        }
    )
    regime_df["market_breadth_lookback"] = breadth.reindex(regime_df["date"]).rolling(
        cfg.breadth_lookback,
        min_periods=min(cfg.breadth_lookback, max(2, cfg.breadth_lookback // 2)),
    ).mean().to_numpy()
    regime_df["market_vol_lookback"] = market_ret.reindex(regime_df["date"]).rolling(
        cfg.market_vol_lookback,
        min_periods=min(cfg.market_vol_lookback, max(2, cfg.market_vol_lookback // 2)),
    ).std().to_numpy()
    regime_df["style_small_vs_large_lookback"] = style_small_vs_large.reindex(regime_df["date"]).rolling(
        cfg.style_lookback,
        min_periods=min(cfg.style_lookback, max(2, cfg.style_lookback // 2)),
    ).mean().to_numpy()

    vol_series = regime_df["market_vol_lookback"].dropna()
    half_vol_threshold = float(vol_series.quantile(cfg.market_vol_half_quantile)) if not vol_series.empty else np.inf
    low_vol_threshold = float(vol_series.quantile(cfg.market_vol_low_quantile)) if not vol_series.empty else np.inf

    states: list[str] = []
    multipliers: list[float] = []
    for row in regime_df.itertuples(index=False):
        severe_flags = 0
        soft_flags = 0
        if pd.notna(row.market_breadth_lookback):
            if row.market_breadth_lookback < cfg.breadth_low_risk:
                severe_flags += 1
            elif row.market_breadth_lookback < cfg.breadth_full_risk:
                soft_flags += 1
        if pd.notna(row.market_vol_lookback):
            if row.market_vol_lookback >= low_vol_threshold:
                severe_flags += 1
            elif row.market_vol_lookback >= half_vol_threshold:
                soft_flags += 1
        if pd.notna(row.style_small_vs_large_lookback):
            if row.style_small_vs_large_lookback <= cfg.style_low_threshold:
                severe_flags += 1
            elif row.style_small_vs_large_lookback <= cfg.style_half_threshold:
                soft_flags += 1

        if severe_flags >= 2:
            states.append("low_risk")
            multipliers.append(float(cfg.low_risk_multiplier))
        elif severe_flags == 1 or soft_flags >= 1:
            states.append("half_risk")
            multipliers.append(float(cfg.half_risk_multiplier))
        else:
            states.append("full_risk")
            multipliers.append(1.0)

    regime_df["risk_state"] = states
    regime_df["risk_multiplier"] = multipliers
    out = out.merge(regime_df, on="date", how="left", suffixes=("", "_gate"))
    out["risk_state"] = out["risk_state"].fillna("full_risk")
    out["risk_multiplier"] = pd.to_numeric(out["risk_multiplier"], errors="coerce").fillna(1.0)

    day_state = regime_df["risk_state"].value_counts().to_dict()
    summary = RegimeGateSummary(
        enabled=True,
        full_risk_days=int(day_state.get("full_risk", 0)),
        half_risk_days=int(day_state.get("half_risk", 0)),
        low_risk_days=int(day_state.get("low_risk", 0)),
        avg_risk_multiplier=float(regime_df["risk_multiplier"].mean()) if len(regime_df) else 1.0,
    )
    out.attrs["regime_gate_summary"] = asdict(summary)
    return out, summary
