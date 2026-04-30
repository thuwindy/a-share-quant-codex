from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TurnoverRuleConfig:
    weight_change_threshold: float = 0.0
    rank_change_threshold: int = 0
    entry_score_advantage_threshold: float = 0.0


def build_trade_log(
    selected: pd.DataFrame,
    previous_weights: dict[str, float] | None,
    previous_ranks: dict[str, int] | None,
    eligible: pd.DataFrame,
    cfg: TurnoverRuleConfig,
) -> pd.DataFrame:
    previous_weights = previous_weights or {}
    previous_ranks = previous_ranks or {}
    selected = selected.copy()
    selected_codes = set(selected["code"]) if not selected.empty else set()
    eligible_codes = set(eligible["code"]) if eligible is not None and not eligible.empty else set()
    eligible_score_map = (
        dict(zip(eligible["code"], pd.to_numeric(eligible["score"], errors="coerce")))
        if eligible is not None and not eligible.empty and "score" in eligible.columns
        else {}
    )
    selected_score_map = (
        dict(zip(selected["code"], pd.to_numeric(selected["score"], errors="coerce")))
        if not selected.empty and "score" in selected.columns
        else {}
    )
    selected_rank_map = dict(zip(selected["code"], selected["rank"])) if not selected.empty and "rank" in selected.columns else {}

    records: list[dict[str, object]] = []
    for code, new_weight in zip(selected.get("code", []), selected.get("target_weight", [])):
        prev_weight = float(previous_weights.get(code, 0.0))
        prev_rank = int(previous_ranks.get(code, 10_000))
        new_rank = int(selected_rank_map.get(code, prev_rank))
        score_advantage = float(selected_score_map.get(code, 0.0) - eligible_score_map.get(code, 0.0))
        if code not in previous_weights:
            reason = "new_entry_stronger"
        elif abs(float(new_weight) - prev_weight) >= cfg.weight_change_threshold:
            if abs(new_rank - prev_rank) > cfg.rank_change_threshold:
                reason = "rank_jump"
            else:
                reason = "weight_gap"
        else:
            continue
        records.append(
            {
                "code": code,
                "trade_reason": reason,
                "previous_weight": prev_weight,
                "target_weight": float(new_weight),
                "previous_rank": prev_rank if prev_rank != 10_000 else pd.NA,
                "target_rank": new_rank,
                "score_advantage": score_advantage,
            }
        )

    for code, prev_weight in previous_weights.items():
        if code in selected_codes:
            continue
        reason = "forced_exit" if code in eligible_codes else "veto_filter"
        records.append(
            {
                "code": code,
                "trade_reason": reason,
                "previous_weight": float(prev_weight),
                "target_weight": 0.0,
                "previous_rank": int(previous_ranks.get(code, 10_000)) if previous_ranks else pd.NA,
                "target_rank": pd.NA,
                "score_advantage": pd.NA,
            }
        )

    if not records:
        return pd.DataFrame(columns=["code", "trade_reason", "previous_weight", "target_weight", "previous_rank", "target_rank", "score_advantage"])
    return pd.DataFrame(records)
