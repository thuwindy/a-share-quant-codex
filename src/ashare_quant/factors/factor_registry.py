from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorDefinition:
    family: str
    description: str
    is_low_freq_experimental: bool = False
    allowed_in_production: bool = True

    @property
    def low_freq_experimental(self) -> bool:
        return self.is_low_freq_experimental


FACTOR_REGISTRY: dict[str, FactorDefinition] = {
    "momentum_5": FactorDefinition("momentum", "5日价格动量"),
    "momentum_10": FactorDefinition("momentum", "10日价格动量"),
    "momentum_20": FactorDefinition("momentum", "20日价格动量"),
    "momentum_60": FactorDefinition("momentum", "60日价格动量"),
    "momentum_120": FactorDefinition("momentum", "120日价格动量"),
    "price_vs_ma20": FactorDefinition("momentum", "收盘价相对20日均线偏离"),
    "price_vs_ma60": FactorDefinition("momentum", "收盘价相对60日均线偏离"),
    "trend_slope_20": FactorDefinition("momentum", "20日趋势斜率"),
    "trend_gap_20": FactorDefinition("momentum", "收盘价相对20日均线偏离"),
    "ma_gap_5_20": FactorDefinition("momentum", "5日均线相对20日均线偏离"),
    "ma_gap_20_60": FactorDefinition("momentum", "20日均线相对60日均线偏离"),
    "industry_breadth_5": FactorDefinition("momentum", "行业内5日强势扩散度"),
    "industry_leader_follow_5": FactorDefinition("momentum", "行业龙头确认后的5日跟随扩散"),
    "overhead_density_20": FactorDefinition("money_flow", "20日换手加权上方兑现压力密度"),
    "reversal_5": FactorDefinition("reversal", "5日反转"),
    "gap_5": FactorDefinition("reversal", "近5日跳空均值"),
    "volatility_20": FactorDefinition("volatility", "20日波动率"),
    "volatility_60": FactorDefinition("volatility", "60日波动率"),
    "downside_vol_20": FactorDefinition("volatility", "20日下行波动率"),
    "intraday_range_10": FactorDefinition("volatility", "10日平均日内振幅"),
    "atr_14": FactorDefinition("volatility", "14日ATR"),
    "drawdown_20": FactorDefinition("volatility", "20日滚动回撤"),
    "bb_width_20": FactorDefinition("volatility", "布林带宽度"),
    "turnover_20": FactorDefinition("turnover_liquidity", "20日换手"),
    "turnover_burst_rank_10": FactorDefinition("turnover_liquidity", "10日换手爆发分位"),
    "liquidity_20": FactorDefinition("turnover_liquidity", "20日成交额均值"),
    "volume_shock_20": FactorDefinition("turnover_liquidity", "20日量能冲击"),
    "smart_money_inflow_20": FactorDefinition("money_flow", "20日聪明钱流入"),
    "overhead_resistance": FactorDefinition("money_flow", "筹码上方压力"),
    "analyst_revision_score": FactorDefinition("money_flow", "研报修正分"),
    "log_mkt_cap": FactorDefinition("size", "总市值对数"),
    "log_float_mkt_cap": FactorDefinition("size", "流通市值对数"),
    "pe_ttm": FactorDefinition("value_low_freq", "滚动市盈率", is_low_freq_experimental=True, allowed_in_production=False),
    "pb": FactorDefinition("value_low_freq", "市净率", is_low_freq_experimental=True, allowed_in_production=False),
    "ps_ttm": FactorDefinition("value_low_freq", "滚动市销率", is_low_freq_experimental=True, allowed_in_production=False),
    "roe": FactorDefinition("quality_low_freq", "净资产收益率", is_low_freq_experimental=True, allowed_in_production=False),
    "gross_margin": FactorDefinition("quality_low_freq", "毛利率", is_low_freq_experimental=True, allowed_in_production=False),
    "debt_to_assets": FactorDefinition("quality_low_freq", "资产负债率", is_low_freq_experimental=True, allowed_in_production=False),
    "operating_cashflow_ratio": FactorDefinition("quality_low_freq", "经营现金流质量", is_low_freq_experimental=True, allowed_in_production=False),
    "quality_20": FactorDefinition("quality_behavioral", "20日行为质量"),
    "rsi_14": FactorDefinition("rule_based", "RSI14"),
    "macd_line_12_26_9": FactorDefinition("rule_based", "MACD快线"),
    "macd_hist_12_26_9": FactorDefinition("rule_based", "MACD柱状图"),
    "bollinger_z_20": FactorDefinition("rule_based", "布林带Z分数"),
    "stoch_k_14": FactorDefinition("rule_based", "KDJ-K"),
    "stoch_d_14": FactorDefinition("rule_based", "KDJ-D"),
    "cci_20": FactorDefinition("rule_based", "CCI20"),
    "rule_ma_trend": FactorDefinition("rule_based", "均线趋势规则"),
    "rule_rsi_rebound": FactorDefinition("rule_based", "RSI反弹规则"),
    "rule_macd_trend": FactorDefinition("rule_based", "MACD趋势规则"),
    "rule_breakout_20": FactorDefinition("rule_based", "20日突破规则"),
}


DEFAULT_FACTOR_FAMILIES = [
    "momentum",
    "volatility",
    "turnover_liquidity",
    "money_flow",
    "size",
    "value_low_freq",
    "quality_low_freq",
    "rule_based",
]


def list_factor_families() -> list[str]:
    return sorted({item.family for item in FACTOR_REGISTRY.values()})


def factors_for_family(family: str) -> list[str]:
    key = str(family).strip().lower()
    return [name for name, meta in FACTOR_REGISTRY.items() if meta.family == key]


def factors_for_families(families: list[str] | tuple[str, ...] | None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for family in families or []:
        for factor in factors_for_family(str(family)):
            if factor in seen:
                continue
            seen.add(factor)
            ordered.append(factor)
    return ordered


def family_membership(factors: list[str] | tuple[str, ...]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for factor in factors:
        raw_name = str(factor).removesuffix("_neu")
        meta = FACTOR_REGISTRY.get(raw_name)
        family = meta.family if meta else "unknown"
        grouped.setdefault(family, []).append(raw_name)
    return grouped


def summarize_factor_families(factors: list[str] | tuple[str, ...]) -> dict[str, object]:
    raw_factors = [str(name).removesuffix("_neu") for name in factors]
    grouped = family_membership(raw_factors)
    total = len(raw_factors) or 1
    ratios = {family: len(items) / total for family, items in grouped.items()}
    low_freq = [
        name
        for name in raw_factors
        if FACTOR_REGISTRY.get(name) and FACTOR_REGISTRY[name].low_freq_experimental
    ]
    return {
        "families": grouped,
        "family_ratio": ratios,
        "contains_low_freq_experimental": bool(low_freq),
        "low_freq_experimental_factors": low_freq,
    }


def blocked_production_factors(factors: list[str] | tuple[str, ...]) -> list[str]:
    blocked: list[str] = []
    for factor in factors:
        raw_name = str(factor).removesuffix("_neu")
        meta = FACTOR_REGISTRY.get(raw_name)
        if meta and not meta.allowed_in_production:
            blocked.append(raw_name)
    return blocked


def resolve_family_factor_request(
    *,
    factor_families: list[str] | tuple[str, ...] | None = None,
    factor_columns: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for factor in factors_for_families(list(factor_families or [])):
        if factor not in seen:
            ordered.append(factor)
            seen.add(factor)
    for factor in factor_columns or []:
        raw_name = str(factor).strip().removesuffix("_neu")
        if not raw_name:
            continue
        if raw_name not in seen:
            ordered.append(raw_name)
            seen.add(raw_name)
    return ordered
