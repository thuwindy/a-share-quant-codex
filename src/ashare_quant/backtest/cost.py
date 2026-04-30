from __future__ import annotations

import math


def transaction_cost(
    buy_turnover: float,
    sell_turnover: float,
    commission: float = 0.0003,
    slippage: float = 0.0005,
    sell_tax: float = 0.001,
) -> float:
    return (buy_turnover + sell_turnover) * (commission + slippage) + sell_turnover * sell_tax


def liquidity_aware_transaction_cost(
    trade_deltas: dict[str, float],
    adv20_map: dict[str, float],
    commission: float = 0.0003,
    base_slippage: float = 0.0005,
    sell_tax: float = 0.001,
    portfolio_notional: float = 10_000_000.0,
    slippage_adv_coef: float = 0.1,
    min_adv20: float = 1_000_000.0,
    max_participation: float = 0.25,
    max_slippage: float = 0.02,
) -> tuple[float, float, float]:
    buy_turnover = 0.0
    sell_turnover = 0.0
    total_cost = 0.0

    for code, delta in trade_deltas.items():
        turnover = abs(float(delta))
        if turnover <= 0:
            continue
        adv20_raw = float(adv20_map.get(code, 0.0) or 0.0)
        if not math.isfinite(adv20_raw) or adv20_raw <= 0.0:
            adv20 = float(min_adv20)
        else:
            adv20 = max(float(adv20_raw), float(min_adv20))

        participation = max((turnover * portfolio_notional) / adv20, 0.0)
        participation = min(participation, float(max_participation))
        slippage = min(base_slippage + slippage_adv_coef * participation, float(max_slippage))
        if delta > 0:
            buy_turnover += turnover
            total_cost += turnover * (commission + slippage)
        else:
            sell_turnover += turnover
            total_cost += turnover * (commission + slippage + sell_tax)
    return total_cost, buy_turnover, sell_turnover
