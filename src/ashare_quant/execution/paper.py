from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from ashare_quant.backtest.cost import transaction_cost
from ashare_quant.execution.base import ExecutionAdapter, Fill, Order


@dataclass
class PaperExecutionConfig:
    commission: float = 0.0003
    slippage: float = 0.0005
    sell_tax: float = 0.001


class PaperExecutionAdapter(ExecutionAdapter):
    """Immediate-fill paper adapter that records orders, fills, cash, and positions."""

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        config: PaperExecutionConfig | None = None,
    ) -> None:
        self.config = config or PaperExecutionConfig()
        self.cash_ = float(initial_cash)
        self.positions_: dict[str, float] = {}
        self.avg_prices_: dict[str, float] = {}
        self.fills_: list[Fill] = []

    def get_positions(self) -> dict[str, float]:
        return dict(self.positions_)

    def get_cash(self) -> float:
        return float(self.cash_)

    def submit_orders(self, orders: Sequence[Order]) -> list[Fill]:
        fills = [self._execute_order(order) for order in orders]
        self.fills_.extend(fills)
        return fills

    def get_fills(self, trading_day: str | pd.Timestamp | None = None) -> list[Fill]:
        if trading_day is None:
            return list(self.fills_)
        normalized_day = pd.Timestamp(trading_day)
        return [fill for fill in self.fills_ if fill.trading_day == normalized_day]

    def _execute_order(self, order: Order) -> Fill:
        side = order.side.upper()
        quantity = float(order.quantity)
        price = float(order.reference_price)
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported order side: {order.side}")
        if quantity <= 0:
            raise ValueError("Order quantity must be positive.")
        if price <= 0:
            raise ValueError("Reference price must be positive.")

        trading_day = pd.Timestamp(order.trading_day)
        # Keep paper-execution cost semantics consistent with backtest fixed-cost model:
        # cost = turnover * (commission + slippage) + sell_turnover * sell_tax.
        fill_price = price
        notional = quantity * fill_price
        if side == "BUY":
            cost = transaction_cost(
                buy_turnover=notional,
                sell_turnover=0.0,
                commission=self.config.commission,
                slippage=self.config.slippage,
                sell_tax=self.config.sell_tax,
            )
        else:
            cost = transaction_cost(
                buy_turnover=0.0,
                sell_turnover=notional,
                commission=self.config.commission,
                slippage=self.config.slippage,
                sell_tax=self.config.sell_tax,
            )

        if side == "BUY":
            cash_needed = notional + cost
            if cash_needed > self.cash_ + 1e-9:
                raise ValueError("Insufficient cash for buy order.")
            current_qty = self.positions_.get(order.code, 0.0)
            current_notional = current_qty * self.avg_prices_.get(order.code, 0.0)
            new_qty = current_qty + quantity
            self.positions_[order.code] = new_qty
            self.avg_prices_[order.code] = (current_notional + notional) / new_qty
            self.cash_ -= cash_needed
        else:
            current_qty = self.positions_.get(order.code, 0.0)
            if quantity > current_qty + 1e-9:
                raise ValueError("Sell quantity exceeds current position.")
            remaining_qty = current_qty - quantity
            if remaining_qty <= 1e-9:
                self.positions_.pop(order.code, None)
                self.avg_prices_.pop(order.code, None)
            else:
                self.positions_[order.code] = remaining_qty
            self.cash_ += notional - cost

        return Fill(
            trading_day=trading_day,
            code=order.code,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            notional=notional,
            cost=cost,
        )
