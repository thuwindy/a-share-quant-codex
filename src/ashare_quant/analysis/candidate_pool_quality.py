from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_PATH_HORIZONS = (1, 5, 10, 20)
DEFAULT_FAILURE_SPIKE_THRESHOLD = 0.05
DEFAULT_REALIZED_RETURN_THRESHOLD = 0.02
DEFAULT_TREND_LOOKBACK = 5


def add_candidate_path_columns(
    frame: pd.DataFrame,
    *,
    horizons: Iterable[int] = DEFAULT_PATH_HORIZONS,
) -> pd.DataFrame:
    out = frame.sort_values(["code", "date"]).copy()
    close_col = "research_close" if "research_close" in out.columns else "close"
    high_col = "research_high" if "research_high" in out.columns else "high"
    low_col = "research_low" if "research_low" in out.columns else "low"
    g = out.groupby("code", group_keys=False)
    base_close = pd.to_numeric(out[close_col], errors="coerce").replace(0.0, np.nan)

    for horizon in sorted({int(h) for h in horizons if int(h) > 0}):
        future_close = pd.to_numeric(g[close_col].shift(-horizon), errors="coerce")
        out[f"path_return_{horizon}d"] = future_close / base_close - 1.0
        future_highs = [pd.to_numeric(g[high_col].shift(-step), errors="coerce") for step in range(1, horizon + 1)]
        future_lows = [pd.to_numeric(g[low_col].shift(-step), errors="coerce") for step in range(1, horizon + 1)]
        out[f"path_max_up_{horizon}d"] = pd.concat(future_highs, axis=1).max(axis=1) / base_close - 1.0
        out[f"path_max_drawdown_{horizon}d"] = pd.concat(future_lows, axis=1).min(axis=1) / base_close - 1.0
    return out


def _payoff_ratio(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    pos = clean.loc[clean > 0]
    neg = clean.loc[clean < 0]
    if pos.empty or neg.empty:
        return 0.0
    return float(pos.mean() / abs(neg.mean()))


def summarize_candidate_pool_quality(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    eligibility_col: str = "strategy_tradeable",
    top_ns: Iterable[int] = (20, 10, 5),
    holding_horizon: int = 5,
) -> pd.DataFrame:
    eligible = frame.loc[
        frame[eligibility_col].fillna(False).astype(bool)
        & frame[score_col].notna()
        & frame[f"path_return_{holding_horizon}d"].notna()
    ].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "pool_name",
                "top_n",
                "sample_count",
                "avg_return",
                "hit_rate",
                "win_rate",
                "avg_drawdown",
                "avg_max_up",
                "payoff_ratio",
            ]
        )

    eligible["future_return"] = pd.to_numeric(eligible[f"path_return_{holding_horizon}d"], errors="coerce")
    eligible["future_drawdown"] = pd.to_numeric(eligible[f"path_max_drawdown_{holding_horizon}d"], errors="coerce")
    eligible["future_max_up"] = pd.to_numeric(eligible[f"path_max_up_{holding_horizon}d"], errors="coerce")
    if "daily_median_future_return" in eligible.columns:
        eligible["daily_median_future_return"] = pd.to_numeric(
            eligible["daily_median_future_return"],
            errors="coerce",
        )
    else:
        eligible["daily_median_future_return"] = eligible.groupby("date")["future_return"].transform("median")
    eligible["daily_rank"] = eligible.groupby("date")[score_col].rank(method="first", ascending=False)

    rows: list[dict[str, Any]] = []
    for top_n in sorted({int(v) for v in top_ns if int(v) > 0}, reverse=True):
        subset = eligible.loc[eligible["daily_rank"] <= top_n].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "pool_name": f"top{top_n}",
                "top_n": top_n,
                "sample_count": int(len(subset)),
                "avg_return": float(subset["future_return"].mean()),
                "hit_rate": float((subset["future_return"] > subset["daily_median_future_return"]).mean()),
                "win_rate": float((subset["future_return"] > 0).mean()),
                "avg_drawdown": float(subset["future_drawdown"].mean()),
                "avg_max_up": float(subset["future_max_up"].mean()),
                "payoff_ratio": _payoff_ratio(subset["future_return"]),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_pool_quality_timeseries(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    eligibility_col: str = "strategy_tradeable",
    top_ns: Iterable[int] = (20, 10, 5),
    holding_horizon: int = 5,
    rolling_window: int = 60,
) -> pd.DataFrame:
    eligible = frame.loc[
        frame[eligibility_col].fillna(False).astype(bool)
        & frame[score_col].notna()
        & frame[f"path_return_{holding_horizon}d"].notna()
    ].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["future_return"] = pd.to_numeric(eligible[f"path_return_{holding_horizon}d"], errors="coerce")
    eligible["future_drawdown"] = pd.to_numeric(eligible[f"path_max_drawdown_{holding_horizon}d"], errors="coerce")
    eligible["future_max_up"] = pd.to_numeric(eligible[f"path_max_up_{holding_horizon}d"], errors="coerce")
    if "daily_median_future_return" in eligible.columns:
        eligible["daily_median_future_return"] = pd.to_numeric(
            eligible["daily_median_future_return"],
            errors="coerce",
        )
    else:
        eligible["daily_median_future_return"] = eligible.groupby("date")["future_return"].transform("median")
    eligible["daily_rank"] = eligible.groupby("date")[score_col].rank(method="first", ascending=False)

    rows: list[dict[str, Any]] = []
    for top_n in sorted({int(v) for v in top_ns if int(v) > 0}, reverse=True):
        subset = eligible.loc[eligible["daily_rank"] <= top_n].copy()
        if subset.empty:
            continue
        for date, daily in subset.groupby("date", sort=True):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "pool_name": f"top{top_n}",
                    "top_n": top_n,
                    "sample_count": int(len(daily)),
                    "avg_return": float(daily["future_return"].mean()),
                    "hit_rate": float((daily["future_return"] > daily["daily_median_future_return"]).mean()),
                    "win_rate": float((daily["future_return"] > 0).mean()),
                    "avg_drawdown": float(daily["future_drawdown"].mean()),
                    "avg_max_up": float(daily["future_max_up"].mean()),
                    "payoff_ratio": _payoff_ratio(daily["future_return"]),
                }
            )
    out = pd.DataFrame(rows).sort_values(["pool_name", "date"]).reset_index(drop=True)
    if out.empty:
        return out
    for metric in ("avg_return", "hit_rate", "win_rate", "avg_drawdown", "avg_max_up", "payoff_ratio"):
        out[f"rolling_{rolling_window}_{metric}"] = (
            out.groupby("pool_name")[metric]
            .transform(lambda series: series.rolling(window=rolling_window, min_periods=min(20, rolling_window)).mean())
        )
    return out


def summarize_observation_tag_paths(observation_history: pd.DataFrame) -> pd.DataFrame:
    frame = observation_history.copy()
    required = {"path_return_5d", "path_return_10d", "path_max_up_5d", "path_max_drawdown_5d"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    if "industry_leader_follow_tag" not in frame.columns:
        frame["industry_leader_follow_tag"] = "暂不可判定"
    if "overhead_density_tag" not in frame.columns:
        frame["overhead_density_tag"] = "暂不可判定"
    frame["combined_observation_tag"] = (
        frame["industry_leader_follow_tag"].fillna("无龙头扩散标签").astype(str)
        + " / "
        + frame["overhead_density_tag"].fillna("无兑现压力标签").astype(str)
    )
    rows: list[dict[str, Any]] = []
    for group_type, column in (
        ("industry_leader_follow_tag", "industry_leader_follow_tag"),
        ("overhead_density_tag", "overhead_density_tag"),
        ("combined_observation_tag", "combined_observation_tag"),
    ):
        for tag_value, subset in frame.groupby(column, sort=True):
            clean = subset.loc[pd.to_numeric(subset["path_return_5d"], errors="coerce").notna()].copy()
            if clean.empty:
                continue
            avg_return_5d = float(pd.to_numeric(clean["path_return_5d"], errors="coerce").mean())
            avg_return_10d = float(pd.to_numeric(clean["path_return_10d"], errors="coerce").mean())
            avg_max_up_5d = float(pd.to_numeric(clean["path_max_up_5d"], errors="coerce").mean())
            avg_drawdown_5d = float(pd.to_numeric(clean["path_max_drawdown_5d"], errors="coerce").mean())
            realization_gap = avg_max_up_5d - avg_return_5d
            if realization_gap >= 0.04 and avg_return_5d <= avg_max_up_5d * 0.4:
                path_quality = "会冲一下"
                comment = "上冲空间存在，但利润兑现偏弱。"
            elif avg_return_10d > avg_return_5d and realization_gap <= 0.03:
                path_quality = "可兑现收益"
                comment = "持有时间拉长后收益延续更好，兑现质量更高。"
            else:
                path_quality = "中性路径"
                comment = "路径没有明显偏向，需要继续观察。"
            rows.append(
                {
                    "group_type": group_type,
                    "group_value": str(tag_value),
                    "sample_count": int(len(clean)),
                    "avg_return_5d": avg_return_5d,
                    "avg_return_10d": avg_return_10d,
                    "win_rate_5d": float((pd.to_numeric(clean["path_return_5d"], errors="coerce") > 0).mean()),
                    "avg_max_up_5d": avg_max_up_5d,
                    "avg_drawdown_5d": avg_drawdown_5d,
                    "realization_gap_5d": realization_gap,
                    "path_quality": path_quality,
                    "path_comment": comment,
                }
            )
    return pd.DataFrame(rows).sort_values(["group_type", "avg_return_5d"], ascending=[True, False]).reset_index(drop=True)


def summarize_failure_attribution(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    eligibility_col: str = "strategy_tradeable",
    top_ns: Iterable[int] = (20, 10, 5),
    evaluation_horizon: int = 5,
    spike_threshold: float = DEFAULT_FAILURE_SPIKE_THRESHOLD,
    realized_return_threshold: float = DEFAULT_REALIZED_RETURN_THRESHOLD,
) -> pd.DataFrame:
    future_return_col = f"path_return_{evaluation_horizon}d"
    future_max_up_col = f"path_max_up_{evaluation_horizon}d"
    eligible = frame.loc[
        frame[eligibility_col].fillna(False).astype(bool)
        & frame[score_col].notna()
        & frame[future_return_col].notna()
        & frame[future_max_up_col].notna()
    ].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "pool_name",
                "top_n",
                "sample_count",
                "realized_ratio",
                "ranking_error_ratio",
                "path_error_ratio",
                "failure_comment",
            ]
        )
    eligible["future_return"] = pd.to_numeric(eligible[future_return_col], errors="coerce")
    eligible["future_max_up"] = pd.to_numeric(eligible[future_max_up_col], errors="coerce")
    eligible["daily_rank"] = eligible.groupby("date")[score_col].rank(method="first", ascending=False)

    rows: list[dict[str, Any]] = []
    for top_n in sorted({int(v) for v in top_ns if int(v) > 0}, reverse=True):
        subset = eligible.loc[eligible["daily_rank"] <= top_n].copy()
        if subset.empty:
            continue
        realized_mask = subset["future_return"] >= realized_return_threshold
        ranking_error_mask = subset["future_max_up"] < spike_threshold
        path_error_mask = (~realized_mask) & (subset["future_max_up"] >= spike_threshold)
        realized_ratio = float(realized_mask.mean())
        ranking_error_ratio = float(ranking_error_mask.mean())
        path_error_ratio = float(path_error_mask.mean())
        if path_error_ratio >= max(0.30, ranking_error_ratio + 0.05):
            comment = "兑现路径错更明显：很多票会冲，但利润保留差。"
        elif ranking_error_ratio >= max(0.30, path_error_ratio + 0.05):
            comment = "排序错更明显：不少高分票连有效上冲都没有形成。"
        else:
            comment = "排序错和兑现路径错都存在，暂时没有单一主导问题。"
        rows.append(
            {
                "pool_name": f"top{top_n}",
                "top_n": top_n,
                "sample_count": int(len(subset)),
                "realized_ratio": realized_ratio,
                "ranking_error_ratio": ranking_error_ratio,
                "path_error_ratio": path_error_ratio,
                "failure_comment": comment,
            }
        )
    return pd.DataFrame(rows)


def summarize_pool_quality_trends(
    rolling_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    lookback_points: int = DEFAULT_TREND_LOOKBACK,
) -> pd.DataFrame:
    if rolling_df.empty or summary_df.empty:
        return pd.DataFrame(
            columns=[
                "pool_name",
                "top_n",
                "trend_status",
                "trend_comment",
                "recent_avg_return",
                "prior_avg_return",
                "recent_hit_rate",
                "prior_hit_rate",
                "recent_avg_drawdown",
                "prior_avg_drawdown",
            ]
        )
    rows: list[dict[str, Any]] = []
    for pool_name, pool_df in rolling_df.sort_values("date").groupby("pool_name", sort=True):
        recent = pool_df.tail(lookback_points)
        prior = pool_df.iloc[max(0, len(pool_df) - 2 * lookback_points): max(0, len(pool_df) - lookback_points)]
        if recent.empty:
            continue
        recent_avg_return = float(recent["avg_return"].mean())
        recent_hit_rate = float(recent["hit_rate"].mean())
        recent_avg_drawdown = float(recent["avg_drawdown"].mean())
        prior_avg_return = float(prior["avg_return"].mean()) if not prior.empty else recent_avg_return
        prior_hit_rate = float(prior["hit_rate"].mean()) if not prior.empty else recent_hit_rate
        prior_avg_drawdown = float(prior["avg_drawdown"].mean()) if not prior.empty else recent_avg_drawdown

        drawdown_not_worse = recent_avg_drawdown >= (prior_avg_drawdown - 0.005)
        drawdown_worse = recent_avg_drawdown < (prior_avg_drawdown - 0.005)
        return_improved = recent_avg_return >= (prior_avg_return + 0.01)
        return_worse = recent_avg_return <= (prior_avg_return - 0.01)
        hit_improved = recent_hit_rate >= (prior_hit_rate + 0.03)
        hit_worse = recent_hit_rate <= (prior_hit_rate - 0.03)

        if return_improved and hit_improved and drawdown_not_worse:
            trend_status = "improving"
            trend_comment = "最近候选池质量在改善，收益和命中率同步抬升。"
        elif return_worse and hit_worse and drawdown_worse:
            trend_status = "deteriorating"
            trend_comment = "最近候选池质量在恶化，收益、命中率和回撤同时变差。"
        else:
            trend_status = "flat"
            trend_comment = "最近候选池质量大体持平，尚未看到明确拐点。"
        top_n = int(summary_df.loc[summary_df["pool_name"] == pool_name, "top_n"].iloc[0]) if (summary_df["pool_name"] == pool_name).any() else 0
        rows.append(
            {
                "pool_name": pool_name,
                "top_n": top_n,
                "trend_status": trend_status,
                "trend_comment": trend_comment,
                "recent_avg_return": recent_avg_return,
                "prior_avg_return": prior_avg_return,
                "recent_hit_rate": recent_hit_rate,
                "prior_hit_rate": prior_hit_rate,
                "recent_avg_drawdown": recent_avg_drawdown,
                "prior_avg_drawdown": prior_avg_drawdown,
            }
        )
    return pd.DataFrame(rows).sort_values("top_n", ascending=False).reset_index(drop=True)


def build_stable_observation_cycle_verdict(
    summary_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    tag_df: pd.DataFrame,
    *,
    lookback_points: int = DEFAULT_TREND_LOOKBACK,
) -> dict[str, Any]:
    trend_df = summarize_pool_quality_trends(rolling_df, summary_df, lookback_points=lookback_points)
    top10_trend = trend_df.loc[trend_df["pool_name"] == "top10"]
    top5_trend = trend_df.loc[trend_df["pool_name"] == "top5"]
    improving_signal = bool(
        ((top10_trend["trend_status"] == "improving").any() or (top5_trend["trend_status"] == "improving").any())
    )
    tag_support = False
    if not tag_df.empty:
        tag_support = bool(
            (
                (tag_df["sample_count"] >= 60)
                & (tag_df["path_quality"] == "可兑现收益")
                & (tag_df["avg_return_10d"] > tag_df["avg_return_5d"])
            ).any()
        )
    keep_frozen = not (improving_signal and tag_support)
    if keep_frozen:
        why = "当前还没有跨时间稳定、且动作含义清楚的新证据，继续保持冻结。"
    else:
        why = "候选池质量与观察标签路径都出现了稳定改善迹象，可讨论是否进入下一阶段实验。"
    return {
        "keep_frozen": keep_frozen,
        "why": why,
        "trend_rows": trend_df.to_dict(orient="records"),
        "new_evidence_clear": not keep_frozen,
    }
