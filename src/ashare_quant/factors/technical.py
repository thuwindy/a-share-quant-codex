from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_COLUMNS = [
    "momentum_20",
    "momentum_60",
    "reversal_5",
    "quality_20",
    "liquidity_20",
    "volatility_20",
    "downside_vol_20",
    "intraday_range_10",
    "gap_5",
    "trend_gap_20",
    "turnover_20",
    "volume_shock_20",
    "smart_money_inflow_20",
    "overhead_resistance",
    "analyst_revision_score",
    "rsi_14",
    "macd_line_12_26_9",
    "macd_hist_12_26_9",
    "ma_gap_5_20",
    "ma_gap_20_60",
    "bollinger_z_20",
    "bb_width_20",
    "atr_14",
    "stoch_k_14",
    "stoch_d_14",
    "cci_20",
    "rule_ma_trend",
    "rule_rsi_rebound",
    "rule_macd_trend",
    "rule_breakout_20",
]

RESEARCH_ONLY_FACTOR_COLUMNS = [
    "log_mkt_cap",
    "log_float_mkt_cap",
    "momentum_5",
    "momentum_10",
    "momentum_120",
    "price_vs_ma20",
    "price_vs_ma60",
    "trend_slope_20",
    "volatility_60",
    "drawdown_20",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "roe",
    "gross_margin",
    "debt_to_assets",
    "operating_cashflow_ratio",
]

PROTOTYPE_FACTOR_COLUMNS = [
    "industry_breadth_5",
    "industry_leader_follow_5",
    "overhead_density_20",
    "turnover_burst_rank_10",
]

ALL_FACTOR_COLUMNS = FACTOR_COLUMNS + RESEARCH_ONLY_FACTOR_COLUMNS + PROTOTYPE_FACTOR_COLUMNS

FACTOR_SETS = {
    "all12": FACTOR_COLUMNS.copy(),
    "stable6": [
        "reversal_5",
        "intraday_range_10",
        "gap_5",
        "volatility_20",
        "turnover_20",
        "momentum_20",
    ],
    "stable2": [
        "reversal_5",
        "intraday_range_10",
    ],
    "baseline5": [
        "momentum_20",
        "reversal_5",
        "quality_20",
        "liquidity_20",
        "volatility_20",
    ],
    "mid4": [
        "momentum_20",
        "momentum_60",
        "volatility_20",
        "turnover_20",
    ],
    "premium3": [
        "smart_money_inflow_20",
        "overhead_resistance",
        "analyst_revision_score",
    ],
    "indicator_core": [
        "rsi_14",
        "macd_hist_12_26_9",
        "ma_gap_5_20",
        "ma_gap_20_60",
        "bollinger_z_20",
        "atr_14",
    ],
    "rule_core": [
        "rule_ma_trend",
        "rule_rsi_rebound",
        "rule_macd_trend",
        "rule_breakout_20",
    ],
    "ta_full": [
        "rsi_14",
        "macd_line_12_26_9",
        "macd_hist_12_26_9",
        "ma_gap_5_20",
        "ma_gap_20_60",
        "bollinger_z_20",
        "bb_width_20",
        "atr_14",
        "stoch_k_14",
        "stoch_d_14",
        "cci_20",
        "rule_ma_trend",
        "rule_rsi_rebound",
        "rule_macd_trend",
        "rule_breakout_20",
    ],
    "hybrid_ta_ml": [
        "momentum_20",
        "momentum_60",
        "volatility_20",
        "turnover_20",
        "rsi_14",
        "macd_hist_12_26_9",
        "ma_gap_5_20",
        "rule_breakout_20",
    ],
}


FACTOR_GROUPS = {
    "trend": ["momentum_20", "momentum_60", "trend_gap_20"],
    "reversal": ["reversal_5", "gap_5"],
    "volatility": ["volatility_20", "downside_vol_20", "intraday_range_10"],
    "liquidity": ["liquidity_20", "turnover_20", "volume_shock_20", "turnover_burst_rank_10"],
    "quality": ["quality_20"],
    "flow": ["smart_money_inflow_20", "overhead_resistance", "overhead_density_20"],
    "analyst": ["analyst_revision_score"],
    "ta": [
        "rsi_14",
        "macd_line_12_26_9",
        "macd_hist_12_26_9",
        "ma_gap_5_20",
        "ma_gap_20_60",
        "bollinger_z_20",
        "bb_width_20",
        "atr_14",
        "stoch_k_14",
        "stoch_d_14",
        "cci_20",
    ],
    "rules": ["rule_ma_trend", "rule_rsi_rebound", "rule_macd_trend", "rule_breakout_20"],
    "breadth": ["industry_breadth_5", "industry_leader_follow_5"],
}


SHORT_HORIZON_FACTORS = [
    "reversal_5",
    "gap_5",
    "intraday_range_10",
    "turnover_20",
    "volume_shock_20",
    "industry_breadth_5",
    "industry_leader_follow_5",
    "turnover_burst_rank_10",
]
MEDIUM_HORIZON_FACTORS = ["momentum_20", "trend_gap_20", "quality_20", "volatility_20"]
LONG_HORIZON_FACTORS = ["momentum_60", "momentum_20", "trend_gap_20", "quality_20"]


def _pick_numeric_feature(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _scale_percent_ratio(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    valid = out.dropna()
    if valid.empty:
        return out
    if float(valid.abs().median()) > 2.0:
        return out / 100.0
    return out


def _rolling_slope_ratio(window: np.ndarray) -> float:
    if len(window) == 0 or np.isnan(window).any():
        return np.nan
    x = np.arange(1, len(window) + 1, dtype=float)
    y = np.asarray(window, dtype=float)
    denom = len(x) * np.square(x).sum() - np.square(x.sum())
    if abs(denom) <= 1e-12:
        return np.nan
    slope = (len(x) * np.dot(x, y) - x.sum() * y.sum()) / denom
    mean_level = float(np.mean(np.abs(y)))
    if mean_level <= 1e-12:
        return np.nan
    return float(slope / mean_level)


def normalize_requested_factor_columns(requested: list[str] | None = None) -> list[str]:
    if not requested:
        return [f"{name}_neu" for name in FACTOR_COLUMNS]

    normalized: list[str] = []
    for item in requested:
        factor = str(item).strip()
        if not factor:
            continue
        raw_name = factor[:-4] if factor.endswith("_neu") else factor
        if raw_name not in ALL_FACTOR_COLUMNS:
            raise ValueError(f"Unknown factor column requested: {factor}")
        normalized.append(f"{raw_name}_neu")

    if not normalized:
        raise ValueError("At least one valid factor column is required.")
    return normalized


def factor_set_columns(factor_set: str | None) -> list[str]:
    if not factor_set:
        return FACTOR_SETS["all12"].copy()
    key = str(factor_set).strip().lower()
    if key not in FACTOR_SETS:
        raise ValueError(f"Unknown factor set requested: {factor_set}")
    return FACTOR_SETS[key].copy()


def requested_raw_factor_columns(requested: list[str] | None = None) -> list[str]:
    return [name[:-4] for name in normalize_requested_factor_columns(requested)]


def default_horizon_factor_map() -> dict[int, list[str]]:
    return {
        3: SHORT_HORIZON_FACTORS,
        5: SHORT_HORIZON_FACTORS + MEDIUM_HORIZON_FACTORS,
        10: LONG_HORIZON_FACTORS,
        20: LONG_HORIZON_FACTORS,
    }


def _price_col(df: pd.DataFrame, base: str) -> str:
    research_col = f"research_{base}"
    return research_col if research_col in df.columns else base


def _latest_rank_pct(window: np.ndarray) -> float:
    if len(window) == 0:
        return np.nan
    last = window[-1]
    if np.isnan(last):
        return np.nan
    valid = window[~np.isnan(window)]
    if len(valid) == 0:
        return np.nan
    return float((valid <= last).sum()) / float(len(valid))


def _bounded_center_score(series: pd.Series, *, center: float, width: float) -> pd.Series:
    width = max(float(width), 1e-6)
    score = 1.0 - (series.sub(center).abs() / width)
    return score.clip(lower=0.0, upper=1.0)


def add_technical_factors(
    df: pd.DataFrame,
    include_technical: bool = True,
    include_fundamental: bool = True,
    include_flow: bool = True,
    include_regime: bool = True,
) -> pd.DataFrame:
    """Add simple cross-sectional factors from OHLCV-like data."""

    out = df.sort_values(["code", "date"]).copy()
    if not include_technical:
        return out

    g = out.groupby("code", group_keys=False)
    open_col = _price_col(out, "open")
    high_col = _price_col(out, "high")
    low_col = _price_col(out, "low")
    close_col = _price_col(out, "close")

    out["prev_close"] = g[close_col].shift(1)
    out["ret_1"] = g[close_col].pct_change()
    out["momentum_5"] = g[close_col].pct_change(5)
    out["momentum_10"] = g[close_col].pct_change(10)
    out["momentum_20"] = g[close_col].pct_change(20)
    out["momentum_60"] = g[close_col].pct_change(60)
    out["momentum_120"] = g[close_col].pct_change(120)
    out["reversal_5"] = -g[close_col].pct_change(5)
    out["volatility_20"] = g["ret_1"].transform(lambda s: s.rolling(20, min_periods=10).std())
    out["volatility_60"] = g["ret_1"].transform(lambda s: s.rolling(60, min_periods=20).std())
    out["downside_ret_1"] = out["ret_1"].where(out["ret_1"] < 0.0, 0.0)
    out["downside_vol_20"] = g["downside_ret_1"].transform(lambda s: s.rolling(20, min_periods=10).std())

    avg_price = (out[open_col] + out[high_col] + out[low_col] + out[close_col]) / 4.0
    out["quality_20"] = g.apply(
        lambda x: ((x[close_col] - x[open_col]) / (avg_price.loc[x.index] + 1e-12))
        .rolling(20, min_periods=10)
        .mean()
    ).reset_index(level=0, drop=True)

    out["liquidity_20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    out["range_ratio"] = (out[high_col] - out[low_col]) / (out[close_col].abs() + 1e-12)
    out["intraday_range_10"] = -g["range_ratio"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    out["gap_1"] = out[open_col] / (out["prev_close"] + 1e-12) - 1.0
    out["gap_5"] = g["gap_1"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    out["trend_gap_20"] = out[close_col] / (
        g[close_col].transform(lambda s: s.rolling(20, min_periods=10).mean()) + 1e-12
    ) - 1.0
    out["turnover_1"] = out["amount"] / out["market_cap"].replace(0.0, np.nan)
    out["turnover_20"] = g["turnover_1"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    out["turnover_burst_rank_10"] = g["turnover_1"].transform(
        lambda s: s.rolling(10, min_periods=5).apply(_latest_rank_pct, raw=True)
    )
    out["volume_shock_20"] = out["amount"] / (
        g["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean()) + 1e-12
    ) - 1.0

    out["sma_5"] = g[close_col].transform(lambda s: s.rolling(5, min_periods=3).mean())
    out["sma_10"] = g[close_col].transform(lambda s: s.rolling(10, min_periods=5).mean())
    out["sma_20"] = g[close_col].transform(lambda s: s.rolling(20, min_periods=10).mean())
    out["sma_60"] = g[close_col].transform(lambda s: s.rolling(60, min_periods=20).mean())
    out["rolling_high_20"] = g[close_col].transform(lambda s: s.rolling(20, min_periods=10).max())
    out["ema_12"] = g[close_col].transform(lambda s: s.ewm(span=12, adjust=False, min_periods=12).mean())
    out["ema_26"] = g[close_col].transform(lambda s: s.ewm(span=26, adjust=False, min_periods=26).mean())
    out["macd_line_12_26_9"] = out["ema_12"] - out["ema_26"]
    out["macd_signal_12_26_9"] = g["macd_line_12_26_9"].transform(
        lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    out["macd_hist_12_26_9"] = out["macd_line_12_26_9"] - out["macd_signal_12_26_9"]

    delta = g[close_col].diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = -delta.where(delta < 0.0, 0.0)
    avg_gain = gain.groupby(out["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    avg_loss = loss.groupby(out["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    rs = avg_gain / (avg_loss + 1e-12)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    rolling_std_20 = g[close_col].transform(lambda s: s.rolling(20, min_periods=10).std())
    out["bollinger_z_20"] = (out[close_col] - out["sma_20"]) / (rolling_std_20 + 1e-12)
    out["bb_width_20"] = (4.0 * rolling_std_20) / (out["sma_20"].abs() + 1e-12)
    out["ma_gap_5_20"] = out["sma_5"] / (out["sma_20"] + 1e-12) - 1.0
    out["ma_gap_20_60"] = out["sma_20"] / (out["sma_60"] + 1e-12) - 1.0
    out["price_vs_ma20"] = out[close_col] / (out["sma_20"] + 1e-12) - 1.0
    out["price_vs_ma60"] = out[close_col] / (out["sma_60"] + 1e-12) - 1.0
    out["trend_slope_20"] = g[close_col].transform(
        lambda s: s.rolling(20, min_periods=10).apply(_rolling_slope_ratio, raw=True)
    )
    out["drawdown_20"] = out[close_col] / (out["rolling_high_20"] + 1e-12) - 1.0

    prev_close_ref = out["prev_close"].fillna(out[close_col])
    true_range = pd.concat(
        [
            (out[high_col] - out[low_col]).abs(),
            (out[high_col] - prev_close_ref).abs(),
            (out[low_col] - prev_close_ref).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = true_range.groupby(out["code"]).transform(lambda s: s.rolling(14, min_periods=7).mean()) / (
        out[close_col].abs() + 1e-12
    )

    rolling_high_14 = g[high_col].transform(lambda s: s.rolling(14, min_periods=7).max())
    rolling_low_14 = g[low_col].transform(lambda s: s.rolling(14, min_periods=7).min())
    out["stoch_k_14"] = (out[close_col] - rolling_low_14) / (rolling_high_14 - rolling_low_14 + 1e-12)
    out["stoch_d_14"] = g["stoch_k_14"].transform(lambda s: s.rolling(3, min_periods=2).mean())

    typical_price = (out[high_col] + out[low_col] + out[close_col]) / 3.0
    tp_ma_20 = typical_price.groupby(out["code"]).transform(lambda s: s.rolling(20, min_periods=10).mean())
    tp_md_20 = (
        (typical_price - tp_ma_20)
        .abs()
        .groupby(out["code"])
        .transform(lambda s: s.rolling(20, min_periods=10).mean())
    )
    out["cci_20"] = (typical_price - tp_ma_20) / (0.015 * tp_md_20 + 1e-12)

    rolling_high_20_prev = g[high_col].transform(lambda s: s.rolling(20, min_periods=10).max()).shift(1)
    out["rule_breakout_20"] = out[close_col] / (rolling_high_20_prev + 1e-12) - 1.0
    out["rule_ma_trend"] = 0.6 * np.tanh(out["ma_gap_5_20"].fillna(0.0) * 20.0) + 0.4 * np.tanh(
        out["ma_gap_20_60"].fillna(0.0) * 15.0
    )
    rsi_strength = (out["rsi_14"].fillna(50.0) - 50.0) / 50.0
    rsi_turn = g["rsi_14"].transform(lambda s: s.diff(3)).fillna(0.0) / 10.0
    out["rule_rsi_rebound"] = 0.7 * np.tanh(rsi_strength * 2.0) + 0.3 * np.tanh(rsi_turn)
    out["rule_macd_trend"] = 0.5 * np.tanh(out["macd_line_12_26_9"].fillna(0.0) * 8.0) + 0.5 * np.tanh(
        out["macd_hist_12_26_9"].fillna(0.0) * 12.0
    )

    industry_series = out["industry"].astype(str).str.strip() if "industry" in out.columns else pd.Series("", index=out.index)
    valid_industry = industry_series.ne("") & industry_series.ne("nan")
    momentum_5_short = g[close_col].pct_change(5)
    breadth_signal = 0.5 * momentum_5_short.gt(0.0).astype(float) + 0.5 * out[close_col].gt(out["sma_20"]).astype(float)
    industry_groups = [out["date"], industry_series]
    out["industry_breadth_5"] = breadth_signal.groupby(industry_groups).transform("mean").where(valid_industry, np.nan)

    industry_count = out.groupby(industry_groups)["code"].transform("count").where(valid_industry, np.nan)
    liquid_threshold = out["amount"].groupby(industry_groups).transform("median")
    leader_ret_3 = g[close_col].pct_change(3)
    leader_ret_5 = out["momentum_5"]
    leader_volume_shock = out["volume_shock_20"].fillna(0.0)
    leader_eligible = (
        valid_industry
        & industry_count.fillna(0).ge(3)
        & out["amount"].ge(liquid_threshold.fillna(0.0))
        & out[close_col].gt(out["sma_20"])
        & leader_ret_5.fillna(-1.0).gt(0.0)
    )
    leader_candidate_score = leader_ret_5.fillna(-99.0) + 1e-9 * np.log(out["amount"].clip(lower=1.0))
    eligible_score = leader_candidate_score.where(leader_eligible, -999.0)
    best_eligible_score = eligible_score.groupby(industry_groups).transform("max")
    best_any_score = leader_candidate_score.groupby(industry_groups).transform("max")
    leader_score = np.where(best_eligible_score > -998.0, best_eligible_score, best_any_score)
    leader_score = pd.Series(leader_score, index=out.index, dtype=float)
    is_leader = valid_industry & leader_candidate_score.eq(leader_score)

    group_size = out.groupby(industry_groups)["code"].transform("size").where(valid_industry, np.nan)
    leader_strength_5 = leader_ret_5.where(is_leader).groupby(industry_groups).transform("max")
    leader_strength_3 = leader_ret_3.where(is_leader).groupby(industry_groups).transform("max")
    leader_flow_confirm = leader_volume_shock.where(is_leader).groupby(industry_groups).transform("max")
    leader_above_ma20 = out[close_col].gt(out["sma_20"]).where(is_leader).groupby(industry_groups).transform("max")

    follower_participant = (~is_leader) & out[close_col].gt(out["sma_20"]) & leader_ret_3.fillna(0.0).gt(0.0)
    follower_count = follower_participant.astype(float).groupby(industry_groups).transform("sum")
    non_leader_count = (group_size - is_leader.astype(float).groupby(industry_groups).transform("sum")).clip(lower=1.0)
    follow_share = (follower_count / (non_leader_count + 1e-12)).where(valid_industry, np.nan)

    leader_confirmed = (
        group_size.fillna(0).ge(3)
        & leader_strength_5.fillna(0.0).ge(0.08)
        & leader_strength_3.fillna(0.0).ge(0.03)
        & leader_flow_confirm.fillna(-1.0).gt(0.0)
        & leader_above_ma20.fillna(False).astype(bool)
    )

    follow_ratio = (leader_ret_5 / (leader_strength_5.abs() + 1e-12)).replace([np.inf, -np.inf], np.nan)
    gap_score = _bounded_center_score(follow_ratio.fillna(0.0), center=0.45, width=0.35)
    own_follow_signal = (
        0.5 * out[close_col].gt(out["sma_20"]).astype(float)
        + 0.5 * leader_ret_3.fillna(0.0).gt(0.0).astype(float)
    )
    out["industry_leader_follow_5"] = (
        leader_confirmed.astype(float)
        * follow_share.fillna(0.0).clip(lower=0.0, upper=1.0)
        * gap_score.fillna(0.0)
        * own_follow_signal.fillna(0.0)
        * (~is_leader).astype(float)
    ).where(valid_industry, np.nan)

    chip_cost_95 = _pick_numeric_feature(out, ["chip_cost_95pct", "chip_cost95pct"])
    chip_weight_avg = _pick_numeric_feature(out, ["chip_weight_avg", "chip_avg_cost"])
    chip_winner = _scale_percent_ratio(
        _pick_numeric_feature(out, ["chip_winner_rate", "chip_winner_rate_pct", "chip_profit_ratio"])
    ).clip(lower=0.0, upper=1.0)
    close_price = pd.to_numeric(out[close_col], errors="coerce").replace(0.0, np.nan)
    upper_gap = ((chip_cost_95 - close_price) / close_price).clip(lower=0.0)
    avg_gap = ((chip_weight_avg - close_price) / close_price).clip(lower=0.0)
    winner_deficit = (1.0 - chip_winner).clip(lower=0.0, upper=1.0)
    scaled_upper_gap = (upper_gap / 0.15).clip(lower=0.0, upper=1.0)
    scaled_avg_gap = (avg_gap / 0.10).clip(lower=0.0, upper=1.0)
    pressure_snapshot = (
        0.50 * winner_deficit.fillna(0.0)
        + 0.30 * scaled_upper_gap.fillna(0.0)
        + 0.20 * scaled_avg_gap.fillna(0.0)
    ).clip(lower=0.0, upper=1.0)
    pressure_turnover_weight = out["turnover_1"].clip(lower=0.0).fillna(0.0)
    weighted_pressure = pressure_snapshot * pressure_turnover_weight
    out["overhead_density_20"] = (
        weighted_pressure.groupby(out["code"]).transform(lambda s: s.rolling(20, min_periods=10).sum())
        / (
            pressure_turnover_weight.groupby(out["code"]).transform(lambda s: s.rolling(20, min_periods=10).sum())
            + 1e-12
        )
    ).clip(lower=0.0, upper=1.0)

    out["log_mkt_cap"] = np.log(pd.to_numeric(out.get("market_cap"), errors="coerce").clip(lower=1.0))
    float_market_cap = _pick_numeric_feature(
        out,
        [
            "float_market_cap",
            "float_mv",
            "circ_mv",
            "free_mv",
            "float_mkt_cap",
            "market_cap",
        ],
    ).clip(lower=1.0)
    out["log_float_mkt_cap"] = np.log(float_market_cap)

    out["pe_ttm"] = _pick_numeric_feature(out, ["pe_ttm", "pe"])
    out["pb"] = _pick_numeric_feature(out, ["pb", "pb_mrq"])
    out["ps_ttm"] = _pick_numeric_feature(out, ["ps_ttm", "ps"])
    out["roe"] = _scale_percent_ratio(_pick_numeric_feature(out, ["roe", "roe_dt", "roe_factor"]))
    out["gross_margin"] = _scale_percent_ratio(
        _pick_numeric_feature(out, ["gross_margin", "grossprofit_margin", "gross_margin_rate"])
    )
    out["debt_to_assets"] = _scale_percent_ratio(_pick_numeric_feature(out, ["debt_to_assets", "debt_to_asset"]))
    out["operating_cashflow_ratio"] = _scale_percent_ratio(
        _pick_numeric_feature(out, ["operating_cashflow_ratio", "q_ocf_to_sales", "cashflow_quality_factor"])
    )

    if include_fundamental:
        for col in ("valuation_factor", "profit_quality_factor", "roe_factor", "earnings_growth_factor", "cashflow_quality_factor"):
            if col not in out.columns:
                out[col] = np.nan
        if "analyst_revision_score" not in out.columns:
            out["analyst_revision_score"] = np.nan
    if include_flow:
        for col in ("turnover_change_factor", "amount_surge_factor", "money_flow_factor", "crowding_factor"):
            if col not in out.columns:
                out[col] = np.nan
        if "smart_money_inflow_20" not in out.columns:
            out["smart_money_inflow_20"] = np.nan
        if "overhead_resistance" not in out.columns:
            out["overhead_resistance"] = np.nan
    if include_regime:
        for col in ("index_trend_factor", "market_breadth_factor", "style_rotation_factor", "volatility_regime_factor"):
            if col not in out.columns:
                out[col] = np.nan

    out = out.drop(
        columns=[
            "prev_close",
            "downside_ret_1",
            "range_ratio",
            "gap_1",
            "turnover_1",
            "sma_5",
            "sma_10",
            "sma_20",
            "sma_60",
            "rolling_high_20",
            "ema_12",
            "ema_26",
            "macd_signal_12_26_9",
        ]
    )
    return out


def add_group_scores(df: pd.DataFrame, factor_cols: list[str], use_neutralized: bool = True) -> pd.DataFrame:
    out = df.copy()
    suffix = "_neu" if use_neutralized else ""
    for group_name, members in FACTOR_GROUPS.items():
        cols = [f"{member}{suffix}" for member in members if f"{member}{suffix}" in out.columns]
        if not cols:
            continue
        out[f"group_{group_name}_score"] = out[cols].mean(axis=1, skipna=True)
    return out
