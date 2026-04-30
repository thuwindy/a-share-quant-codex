from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from ashare_quant.portfolio.portfolio_optimizer import PortfolioConfig, optimize_portfolio_weights
from ashare_quant.turnover_diagnostics import TurnoverRuleConfig, build_trade_log


@dataclass(frozen=True)
class ConstructionStats:
    signal_dates: int
    execution_dates: int
    average_selected: float
    sleeve_count: int
    top_n: int


def _next_trade_date_map(dates: list[pd.Timestamp], lag: int) -> dict[pd.Timestamp, pd.Timestamp]:
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for idx, date in enumerate(dates):
        target_idx = idx + lag
        if target_idx < len(dates):
            mapping[pd.Timestamp(date)] = pd.Timestamp(dates[target_idx])
    return mapping


def _apply_entry_filter_mask(
    eligible: pd.DataFrame,
    *,
    previous_weights: dict[str, float] | None,
    score_col: str,
    overhead_enabled: bool,
    min_overhead_resistance: float,
    score_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocked_frames: list[pd.DataFrame] = []
    has_premium_filter = "premium_entry_filter" in eligible.columns
    has_overhead_filter = overhead_enabled and "overhead_resistance" in eligible.columns
    if not has_premium_filter and not has_overhead_filter:
        return eligible, pd.DataFrame(columns=["code", "blocked_reason", "blocked_metric", "blocked_threshold"])

    out = eligible.copy()
    previous_weights = previous_weights or {}
    held_codes = {str(code) for code, weight in previous_weights.items() if float(weight) > 1e-12}
    is_new_entry = ~out["code"].astype(str).isin(held_codes)
    if has_premium_filter:
        premium_mask = is_new_entry & out["premium_entry_filter"].fillna(False).astype(bool)
        if premium_mask.any():
            out.loc[premium_mask, score_col] = float(score_floor)
            blocked = out.loc[premium_mask, ["code"]].copy()
            blocked["blocked_reason"] = out.loc[premium_mask, "premium_entry_filter_reason"].replace("", "premium_entry_filter").to_numpy()
            blocked["blocked_metric"] = pd.NA
            blocked["blocked_threshold"] = pd.NA
            blocked_frames.append(blocked.reset_index(drop=True))

    if has_overhead_filter:
        overhead = pd.to_numeric(out["overhead_resistance"], errors="coerce")
        overhead_mask = is_new_entry & overhead.notna() & overhead.lt(min_overhead_resistance)
        if overhead_mask.any():
            out.loc[overhead_mask, score_col] = float(score_floor)
            blocked = out.loc[overhead_mask, ["code", "overhead_resistance"]].copy()
            blocked["blocked_reason"] = "overhead_entry_filter"
            blocked["blocked_metric"] = pd.to_numeric(blocked["overhead_resistance"], errors="coerce")
            blocked["blocked_threshold"] = float(min_overhead_resistance)
            blocked = blocked.drop(columns=["overhead_resistance"]).reset_index(drop=True)
            blocked_frames.append(blocked)

    if not blocked_frames:
        return out, pd.DataFrame(columns=["code", "blocked_reason", "blocked_metric", "blocked_threshold"])
    blocked = pd.concat(blocked_frames, ignore_index=True, sort=False)
    return out, blocked


def build_portfolio_targets(
    df: pd.DataFrame,
    score_col: str,
    top_n: int = 10,
    rebalance_every: int = 5,
    sleeve_count: int = 1,
    execution_lag: int = 1,
    weighting_method: str = "equal",
    max_weight: float = 0.2,
    industry_cap: float = 0.4,
    min_holdings: int = 5,
    score_threshold: float = 0.0,
    softmax_temperature: float = 1.0,
    weight_change_threshold: float = 0.0,
    rank_change_threshold: int = 0,
    entry_score_advantage_threshold: float = 0.0,
    entry_filter_overhead_enabled: bool = False,
    entry_filter_min_overhead_resistance: float = float("-inf"),
    entry_filter_score_floor: float = -9999.0,
) -> pd.DataFrame:
    """Create long-only portfolio targets with optional sleeve rotation."""

    rows = []
    dates = sorted(pd.to_datetime(df["date"].drop_duplicates()))
    execution_map = _next_trade_date_map(dates, lag=execution_lag)
    sleeve_count = max(int(sleeve_count), 1)
    step = 1 if sleeve_count > 1 else max(int(rebalance_every), 1)
    rebalance_dates = dates[::step]
    previous_targets: dict[int, dict[str, float]] = {sleeve: {} for sleeve in range(sleeve_count)}
    previous_ranks: dict[int, dict[str, int]] = {sleeve: {} for sleeve in range(sleeve_count)}
    trade_logs: list[dict[str, object]] = []

    for rebalance_idx, signal_date in enumerate(rebalance_dates):
        if signal_date not in execution_map:
            continue
        sleeve = rebalance_idx % sleeve_count
        sl = df.loc[df["date"] == signal_date].copy()
        eligibility_col = "strategy_tradeable" if "strategy_tradeable" in sl.columns else "tradeable"
        eligible = sl.loc[sl[eligibility_col].fillna(False)].dropna(subset=[score_col]).sort_values(score_col, ascending=False)
        if eligible.empty:
            continue
        eligible, blocked_entries = _apply_entry_filter_mask(
            eligible,
            previous_weights=previous_targets[sleeve],
            score_col=score_col,
            overhead_enabled=bool(entry_filter_overhead_enabled),
            min_overhead_resistance=float(entry_filter_min_overhead_resistance),
            score_floor=float(entry_filter_score_floor),
        )
        eligible = eligible.sort_values(score_col, ascending=False).reset_index(drop=True)
        risk_state = str(sl["risk_state"].iloc[0]) if "risk_state" in sl.columns else "full_risk"
        risk_multiplier = float(pd.to_numeric(sl["risk_multiplier"], errors="coerce").iloc[0]) if "risk_multiplier" in sl.columns else 1.0
        if risk_state == "low_risk" or risk_multiplier <= 1e-12:
            continue

        target_cfg = PortfolioConfig(
            top_n=top_n,
            weighting_method=weighting_method,
            score_threshold=score_threshold,
            softmax_temperature=softmax_temperature,
            max_weight=max_weight,
            industry_cap=industry_cap,
            min_holdings=min_holdings,
            weight_change_threshold=weight_change_threshold,
            rank_change_threshold=rank_change_threshold,
            entry_score_advantage_threshold=entry_score_advantage_threshold,
        )
        selected = optimize_portfolio_weights(
            eligible,
            cfg=target_cfg,
            score_col=score_col,
            previous_weights=previous_targets[sleeve],
            previous_ranks=previous_ranks[sleeve],
        )
        if selected.empty:
            continue
        selected["target_weight"] = pd.to_numeric(selected["target_weight"], errors="coerce").fillna(0.0) * risk_multiplier

        execution_date = execution_map[pd.Timestamp(signal_date)]
        diag_df = build_trade_log(
            selected=selected,
            previous_weights=previous_targets[sleeve],
            previous_ranks=previous_ranks[sleeve],
            eligible=eligible,
            cfg=TurnoverRuleConfig(
                weight_change_threshold=target_cfg.weight_change_threshold,
                rank_change_threshold=target_cfg.rank_change_threshold,
                entry_score_advantage_threshold=target_cfg.entry_score_advantage_threshold,
            ),
        )
        previous_targets[sleeve] = dict(zip(selected["code"], selected["target_weight"]))
        previous_ranks[sleeve] = dict(zip(selected["code"], selected["rank"]))
        if isinstance(diag_df, pd.DataFrame) and not diag_df.empty:
            diag_df = diag_df.copy()
            diag_df["signal_date"] = pd.Timestamp(signal_date)
            diag_df["execution_date"] = pd.Timestamp(execution_date)
            diag_df["sleeve"] = sleeve
            diag_df["risk_state"] = risk_state
            diag_df["risk_multiplier"] = risk_multiplier
            trade_logs.extend(diag_df.to_dict(orient="records"))
        if isinstance(blocked_entries, pd.DataFrame) and not blocked_entries.empty:
            blocked_entries = blocked_entries.copy()
            blocked_entries["signal_date"] = pd.Timestamp(signal_date)
            blocked_entries["execution_date"] = pd.Timestamp(execution_date)
            blocked_entries["sleeve"] = sleeve
            blocked_entries["risk_state"] = risk_state
            blocked_entries["risk_multiplier"] = risk_multiplier
            trade_logs.extend(blocked_entries.to_dict(orient="records"))
        for row in selected.itertuples(index=False):
            row_reason = ""
            if isinstance(diag_df, pd.DataFrame) and not diag_df.empty:
                matched = diag_df.loc[diag_df["code"] == getattr(row, "code"), "trade_reason"]
                if not matched.empty:
                    row_reason = str(matched.iloc[0])
            rows.append(
                {
                    "date": pd.Timestamp(signal_date),
                    "signal_date": pd.Timestamp(signal_date),
                    "execution_date": pd.Timestamp(execution_date),
                    "sleeve": sleeve,
                    "code": getattr(row, "code"),
                    "target_weight": float(getattr(row, "target_weight")),
                    "score": float(getattr(row, score_col)),
                    "rank": int(getattr(row, "rank")),
                    "industry": getattr(row, "industry", ""),
                    "trade_reason": row_reason,
                    "risk_state": risk_state,
                    "risk_multiplier": risk_multiplier,
                }
            )
    targets = pd.DataFrame(rows)
    if targets.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "signal_date",
                "execution_date",
                "sleeve",
                "code",
                "target_weight",
                "score",
                "rank",
                "industry",
                "trade_reason",
                "risk_state",
                "risk_multiplier",
            ]
        )
    if not targets.empty:
        targets.attrs["construction_stats"] = asdict(
            ConstructionStats(
                signal_dates=int(targets["signal_date"].nunique()),
                execution_dates=int(targets["execution_date"].nunique()),
                average_selected=float(targets.groupby(["signal_date", "sleeve"])["code"].count().mean()),
                sleeve_count=sleeve_count,
                top_n=top_n,
            )
        )
    targets.attrs["trade_logs"] = trade_logs
    return targets


def build_top_n_portfolio(
    df: pd.DataFrame,
    score_col: str,
    top_n: int = 10,
    rebalance_every: int = 5,
) -> pd.DataFrame:
    """Backward-compatible wrapper for the old equal-weight baseline."""

    return build_portfolio_targets(
        df,
        score_col=score_col,
        top_n=top_n,
        rebalance_every=rebalance_every,
        sleeve_count=1,
        execution_lag=1,
        weighting_method="equal",
        max_weight=1.0,
        industry_cap=1.0,
        min_holdings=top_n,
    )
