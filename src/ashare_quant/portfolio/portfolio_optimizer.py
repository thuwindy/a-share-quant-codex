from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class PortfolioConfig:
    top_n: int = 8
    weighting_method: str = "equal"
    score_threshold: float = 0.0
    softmax_temperature: float = 1.0
    max_weight: float = 0.2
    industry_cap: float = 0.4
    min_holdings: int = 5
    weight_change_threshold: float = 0.0
    rank_change_threshold: int = 0
    entry_score_advantage_threshold: float = 0.0


def _normalize(weights: pd.Series) -> pd.Series:
    total = float(weights.sum())
    if total <= 0:
        return weights * 0.0
    return weights / total


def _base_weights(selected: pd.DataFrame, cfg: PortfolioConfig, score_col: str) -> pd.Series:
    n = len(selected)
    if n == 0:
        return pd.Series(dtype=float)
    if cfg.weighting_method == "equal":
        return pd.Series(1.0 / n, index=selected.index, dtype=float)
    if cfg.weighting_method == "rank":
        raw = pd.Series(np.arange(n, 0, -1, dtype=float), index=selected.index)
        return _normalize(raw)
    if cfg.weighting_method == "softmax":
        centered = selected[score_col].astype(float) - float(selected[score_col].astype(float).max())
        raw = np.exp(centered / max(cfg.softmax_temperature, 1e-6))
        return _normalize(pd.Series(raw, index=selected.index, dtype=float))
    if cfg.weighting_method == "linear":
        raw = (selected[score_col].astype(float) - cfg.score_threshold).clip(lower=0.0)
        if float(raw.sum()) <= 0:
            raw = pd.Series(1.0, index=selected.index, dtype=float)
        return _normalize(raw)
    raise ValueError(f"Unsupported weighting_method: {cfg.weighting_method}")


def _apply_max_weight(weights: pd.Series, max_weight: float) -> pd.Series:
    out = weights.copy()
    if max_weight <= 0:
        return out
    for _ in range(5):
        capped = out.clip(upper=max_weight)
        residual = 1.0 - float(capped.sum())
        if residual <= 1e-10:
            out = capped
            break
        eligible = capped[capped < max_weight - 1e-10]
        if eligible.empty:
            out = capped / max(float(capped.sum()), 1e-12)
            break
        add = residual * eligible / max(float(eligible.sum()), 1e-12)
        out = capped
        out.loc[eligible.index] = capped.loc[eligible.index] + add
    return _normalize(out)


def _apply_industry_cap(selected: pd.DataFrame, weights: pd.Series, industry_cap: float) -> pd.Series:
    if industry_cap <= 0 or "industry" not in selected.columns:
        return weights
    out = weights.copy().astype(float)
    industry_labels = selected["industry"].fillna("Unknown")

    for _ in range(10):
        industry_weight = selected.assign(_w=out, _industry=industry_labels).groupby("_industry")["_w"].sum()
        capped = industry_weight[industry_weight > industry_cap + 1e-12]
        if capped.empty:
            break
        for industry, current in capped.items():
            idx = selected.index[industry_labels == industry]
            if len(idx) == 0 or current <= 0:
                continue
            out.loc[idx] = out.loc[idx] * (industry_cap / current)

    for _ in range(10):
        total_weight = float(out.sum())
        residual = max(1.0 - total_weight, 0.0)
        if residual <= 1e-10:
            break
        industry_weight = selected.assign(_w=out, _industry=industry_labels).groupby("_industry")["_w"].sum()
        industry_room = (industry_cap - industry_weight).clip(lower=0.0)
        eligible_idx = []
        eligible_room = []
        for idx, industry in zip(selected.index, industry_labels):
            room = float(industry_room.get(industry, 0.0))
            if room <= 1e-12:
                continue
            eligible_idx.append(idx)
            eligible_room.append(room)
        if not eligible_idx:
            break
        base = out.loc[eligible_idx].clip(lower=0.0)
        if float(base.sum()) <= 1e-12:
            allocation_key = pd.Series(1.0, index=eligible_idx, dtype=float)
        else:
            allocation_key = base
        allocation_key = allocation_key / max(float(allocation_key.sum()), 1e-12)
        proposed_add = allocation_key * residual
        current_industry_weight = selected.assign(_w=out, _industry=industry_labels).groupby("_industry")["_w"].sum()
        for idx in eligible_idx:
            industry = industry_labels.loc[idx]
            room = max(industry_cap - float(current_industry_weight.get(industry, 0.0)), 0.0)
            add = min(float(proposed_add.loc[idx]), room)
            if add <= 0:
                continue
            out.loc[idx] += add
            current_industry_weight.loc[industry] = float(current_industry_weight.get(industry, 0.0)) + add
    return out


def _apply_no_trade_band(
    new_weights: pd.Series,
    previous_weights: dict[str, float] | None,
    selected: pd.DataFrame,
    previous_ranks: dict[str, int] | None,
    cfg: PortfolioConfig,
) -> pd.Series:
    if not previous_weights:
        return new_weights

    out = new_weights.copy()
    current_ranks = dict(zip(selected["code"], range(1, len(selected) + 1)))
    previous_ranks = previous_ranks or {}
    for idx, row in selected.iterrows():
        code = str(row["code"])
        prev_weight = float(previous_weights.get(code, 0.0))
        if prev_weight <= 1e-12:
            # No-trade bands should stabilize existing holdings, not block every
            # new entry whose target weight is naturally smaller than the band.
            continue
        prev_rank = int(previous_ranks.get(code, 10_000))
        rank_change = abs(current_ranks.get(code, prev_rank) - prev_rank)
        if abs(float(out.loc[idx]) - prev_weight) < cfg.weight_change_threshold:
            out.loc[idx] = prev_weight
        elif rank_change <= cfg.rank_change_threshold:
            out.loc[idx] = prev_weight
    out = out.clip(lower=0.0)
    total = float(out.sum())
    if total > 1.0 + 1e-12:
        out = out / total
    return out


def _stabilize_selection(
    ranked: pd.DataFrame,
    selected: pd.DataFrame,
    previous_weights: dict[str, float] | None,
    cfg: PortfolioConfig,
    score_col: str,
) -> pd.DataFrame:
    if selected.empty or not previous_weights or cfg.entry_score_advantage_threshold <= 0:
        return selected

    candidate_map = ranked.set_index("code")
    current = selected.copy()
    held_codes = [code for code, weight in previous_weights.items() if float(weight) > 1e-12]
    newcomer_mask = ~current["code"].isin(held_codes)

    for held_code in held_codes:
        if held_code in set(current["code"]) or held_code not in candidate_map.index:
            continue
        held_row = candidate_map.loc[held_code]
        held_score = float(held_row[score_col])
        newcomer_slice = current.loc[newcomer_mask].sort_values(score_col)
        if newcomer_slice.empty:
            continue
        replace_idx = newcomer_slice.index[0]
        newcomer_score = float(current.loc[replace_idx, score_col])
        if newcomer_score - held_score >= cfg.entry_score_advantage_threshold:
            continue
        replacement_row = held_row.to_dict()
        replacement_row["code"] = held_code
        replacement_frame = pd.DataFrame([replacement_row], columns=current.columns)
        current = current.drop(index=replace_idx)
        current = pd.concat([current, replacement_frame], ignore_index=True)
        newcomer_mask = ~current["code"].isin(held_codes)

    return current.sort_values(score_col, ascending=False).head(len(selected)).reset_index(drop=True)


def optimize_portfolio_weights(
    candidates: pd.DataFrame,
    cfg: PortfolioConfig,
    score_col: str = "score",
    previous_weights: dict[str, float] | None = None,
    previous_ranks: dict[str, int] | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["code", "target_weight"])

    ranked = candidates.sort_values(score_col, ascending=False).copy()
    threshold_filtered = ranked
    if cfg.weighting_method == "linear":
        threshold_filtered = ranked.loc[ranked[score_col].astype(float) >= cfg.score_threshold].copy()
    selected = threshold_filtered.head(max(cfg.top_n, cfg.min_holdings)).copy()
    if len(selected) < cfg.min_holdings:
        selected = ranked.head(max(cfg.min_holdings, cfg.top_n)).copy()
    selected = _stabilize_selection(
        ranked=ranked,
        selected=selected,
        previous_weights=previous_weights,
        cfg=cfg,
        score_col=score_col,
    )
    weights = _base_weights(selected, cfg=cfg, score_col=score_col)
    weights = _apply_max_weight(weights, max_weight=cfg.max_weight)
    weights = _apply_industry_cap(selected, weights=weights, industry_cap=cfg.industry_cap)
    weights = _apply_no_trade_band(
        weights,
        previous_weights=previous_weights,
        selected=selected,
        previous_ranks=previous_ranks,
        cfg=cfg,
    )
    selected["target_weight"] = weights.values
    selected["rank"] = range(1, len(selected) + 1)
    return selected
