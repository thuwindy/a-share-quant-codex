from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Sequence

import pandas as pd


OrderSide = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class Order:
    """Minimal paper-order contract for execution adapter tests and demos."""

    trading_day: str | pd.Timestamp
    code: str
    side: OrderSide
    quantity: float
    reference_price: float


@dataclass(frozen=True)
class Fill:
    trading_day: pd.Timestamp
    code: str
    side: OrderSide
    quantity: float
    fill_price: float
    notional: float
    cost: float


class ExecutionAdapter(ABC):
    """Abstract execution adapter interface for paper and live implementations."""

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        """Return current position quantities keyed by security code."""

    @abstractmethod
    def get_cash(self) -> float:
        """Return current available cash."""

    @abstractmethod
    def submit_orders(self, orders: Sequence[Order]) -> list[Fill]:
        """Submit orders and return any generated fills."""

    @abstractmethod
    def get_fills(self, trading_day: str | pd.Timestamp | None = None) -> list[Fill]:
        """Return all fills or filter them to a single trading day."""
