from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.execution.base import Order
from ashare_quant.execution.paper import PaperExecutionAdapter, PaperExecutionConfig


class PaperExecutionTest(unittest.TestCase):
    def test_buy_and_sell_update_cash_positions_and_fills(self) -> None:
        adapter = PaperExecutionAdapter(
            initial_cash=1_000.0,
            config=PaperExecutionConfig(commission=0.0, slippage=0.0, sell_tax=0.0),
        )

        buy_fill = adapter.submit_orders(
            [Order(trading_day="2024-01-02", code="000001.SZ", side="BUY", quantity=10.0, reference_price=10.0)]
        )[0]
        self.assertEqual(buy_fill.notional, 100.0)
        self.assertEqual(adapter.get_positions()["000001.SZ"], 10.0)
        self.assertAlmostEqual(adapter.get_cash(), 900.0)

        sell_fill = adapter.submit_orders(
            [Order(trading_day="2024-01-03", code="000001.SZ", side="SELL", quantity=4.0, reference_price=12.0)]
        )[0]
        self.assertEqual(sell_fill.notional, 48.0)
        self.assertEqual(adapter.get_positions()["000001.SZ"], 6.0)
        self.assertAlmostEqual(adapter.get_cash(), 948.0)

        fills = adapter.get_fills(pd.Timestamp("2024-01-03"))
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].side, "SELL")

    def test_sell_more_than_position_raises(self) -> None:
        adapter = PaperExecutionAdapter(
            initial_cash=1_000.0,
            config=PaperExecutionConfig(commission=0.0, slippage=0.0, sell_tax=0.0),
        )
        with self.assertRaises(ValueError):
            adapter.submit_orders(
                [Order(trading_day="2024-01-02", code="000001.SZ", side="SELL", quantity=1.0, reference_price=10.0)]
            )

    def test_cost_formula_matches_backtest_semantics(self) -> None:
        adapter = PaperExecutionAdapter(
            initial_cash=2_000.0,
            config=PaperExecutionConfig(commission=0.001, slippage=0.002, sell_tax=0.003),
        )

        buy_fill = adapter.submit_orders(
            [Order(trading_day="2024-01-02", code="000001.SZ", side="BUY", quantity=10.0, reference_price=10.0)]
        )[0]
        expected_buy_notional = 100.0
        expected_buy_cost = expected_buy_notional * (0.001 + 0.002)
        self.assertAlmostEqual(buy_fill.fill_price, 10.0)
        self.assertAlmostEqual(buy_fill.notional, expected_buy_notional)
        self.assertAlmostEqual(buy_fill.cost, expected_buy_cost)
        self.assertAlmostEqual(adapter.get_cash(), 2_000.0 - expected_buy_notional - expected_buy_cost)

        sell_fill = adapter.submit_orders(
            [Order(trading_day="2024-01-03", code="000001.SZ", side="SELL", quantity=4.0, reference_price=10.0)]
        )[0]
        expected_sell_notional = 40.0
        expected_sell_cost = expected_sell_notional * (0.001 + 0.002 + 0.003)
        self.assertAlmostEqual(sell_fill.fill_price, 10.0)
        self.assertAlmostEqual(sell_fill.notional, expected_sell_notional)
        self.assertAlmostEqual(sell_fill.cost, expected_sell_cost)
        expected_cash = (
            2_000.0
            - expected_buy_notional
            - expected_buy_cost
            + expected_sell_notional
            - expected_sell_cost
        )
        self.assertAlmostEqual(adapter.get_cash(), expected_cash)


if __name__ == "__main__":
    unittest.main()
