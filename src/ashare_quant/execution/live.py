from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ashare_quant.execution.base import ExecutionAdapter, Fill, Order


@dataclass
class LiveExecutionConfig:
    broker_name: str = "unconfigured"
    account_id: str = ""
    endpoint: str = ""


class LiveExecutionAdapter(ExecutionAdapter):
    """Production-facing adapter shell. Concrete broker implementations should subclass this."""

    def __init__(self, config: LiveExecutionConfig | None = None) -> None:
        self.config = config or LiveExecutionConfig()

    def get_positions(self) -> dict[str, float]:
        raise NotImplementedError("Broker position query is not implemented yet.")

    def get_cash(self) -> float:
        raise NotImplementedError("Broker cash query is not implemented yet.")

    def submit_orders(self, orders: Sequence[Order]) -> list[Fill]:
        raise NotImplementedError("Broker order submission is not implemented yet.")

    def get_fills(self, trading_day=None) -> list[Fill]:
        raise NotImplementedError("Broker fill query is not implemented yet.")
