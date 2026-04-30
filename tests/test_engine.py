from __future__ import annotations

import unittest

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.backtest.cost import liquidity_aware_transaction_cost
from ashare_quant.backtest.engine import BacktestConfig, DailyBacktester


class EngineTest(unittest.TestCase):
    def test_liquidity_aware_cost_is_capped_when_adv_is_missing(self) -> None:
        cost, buy_turnover, sell_turnover = liquidity_aware_transaction_cost(
            {"000001.SZ": 0.5},
            adv20_map={"000001.SZ": 0.0},
            commission=0.0003,
            base_slippage=0.0005,
            portfolio_notional=10_000_000.0,
            slippage_adv_coef=0.1,
            min_adv20=1_000_000.0,
            max_participation=0.25,
            max_slippage=0.02,
        )
        self.assertAlmostEqual(buy_turnover, 0.5)
        self.assertAlmostEqual(sell_turnover, 0.0)
        self.assertLess(cost, 0.02)

    def test_execution_day_buy_block_prevents_position_and_extreme_cost(self) -> None:
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "code": ["000001.SZ", "000001.SZ"],
                "research_close": [10.0, 10.2],
                "amount": [100_000_000.0, 0.0],
                "adv20": [100_000_000.0, 0.0],
                "can_buy": [True, False],
                "can_sell": [True, False],
            }
        )
        targets = pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2024-01-02")],
                "execution_date": [pd.Timestamp("2024-01-03")],
                "sleeve": [0],
                "code": ["000001.SZ"],
                "target_weight": [1.0],
            }
        )
        result, metrics = DailyBacktester(
            config=BacktestConfig(use_liquidity_aware_cost=True, sleeve_count=1)
        ).run(panel=panel, targets=targets)
        last = result.iloc[-1]
        self.assertEqual(float(last["cost"]), 0.0)
        self.assertEqual(int(last["n_holdings"]), 0)
        self.assertGreaterEqual(metrics["annual_return"], -1.0)
        self.assertGreaterEqual(metrics["max_drawdown"], -1.0)

    def test_open_execution_buys_at_open_not_close(self) -> None:
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "code": ["000001.SZ", "000001.SZ"],
                "research_open": [10.0, 12.0],
                "research_close": [10.0, 18.0],
                "amount": [100_000_000.0, 100_000_000.0],
                "can_buy": [True, True],
                "can_sell": [True, True],
            }
        )
        targets = pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2024-01-02")],
                "execution_date": [pd.Timestamp("2024-01-03")],
                "sleeve": [0],
                "code": ["000001.SZ"],
                "target_weight": [1.0],
            }
        )
        result, _ = DailyBacktester(
            config=BacktestConfig(
                execution_price="open",
                commission=0.0,
                slippage=0.0,
                sell_tax=0.0,
                use_liquidity_aware_cost=False,
                sleeve_count=1,
            )
        ).run(panel=panel, targets=targets)
        last = result.iloc[-1]
        self.assertAlmostEqual(float(last["gross_return"]), 0.5, places=6)
        self.assertAlmostEqual(float(last["net_return"]), 0.5, places=6)

    def test_close_execution_enters_after_close_and_realizes_next_day(self) -> None:
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "research_close": [10.0, 12.0, 15.0],
                "amount": [100_000_000.0, 100_000_000.0, 100_000_000.0],
                "can_buy": [True, True, True],
                "can_sell": [True, True, True],
            }
        )
        targets = pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2024-01-02")],
                "execution_date": [pd.Timestamp("2024-01-03")],
                "sleeve": [0],
                "code": ["000001.SZ"],
                "target_weight": [1.0],
            }
        )
        result, _ = DailyBacktester(
            config=BacktestConfig(
                execution_price="close",
                commission=0.0,
                slippage=0.0,
                sell_tax=0.0,
                use_liquidity_aware_cost=False,
                sleeve_count=1,
            )
        ).run(panel=panel, targets=targets)
        exec_day = result.loc[result["date"] == pd.Timestamp("2024-01-03")].iloc[0]
        next_day = result.loc[result["date"] == pd.Timestamp("2024-01-04")].iloc[0]
        self.assertAlmostEqual(float(exec_day["gross_return"]), 0.0, places=8)
        self.assertAlmostEqual(float(exec_day["net_return"]), 0.0, places=8)
        self.assertAlmostEqual(float(next_day["gross_return"]), 15.0 / 12.0 - 1.0, places=8)
        self.assertAlmostEqual(float(next_day["net_return"]), 15.0 / 12.0 - 1.0, places=8)

    def test_vwap_execution_buys_at_vwap_not_close(self) -> None:
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "code": ["000001.SZ", "000001.SZ"],
                "research_open": [10.0, 12.0],
                "research_high": [10.0, 20.0],
                "research_low": [10.0, 12.0],
                "research_close": [10.0, 18.0],
                "research_vwap": [10.0, 15.5],
                "amount": [100_000_000.0, 100_000_000.0],
                "can_buy": [True, True],
                "can_sell": [True, True],
            }
        )
        targets = pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2024-01-02")],
                "execution_date": [pd.Timestamp("2024-01-03")],
                "sleeve": [0],
                "code": ["000001.SZ"],
                "target_weight": [1.0],
            }
        )
        result, _ = DailyBacktester(
            config=BacktestConfig(
                execution_price="vwap",
                commission=0.0,
                slippage=0.0,
                sell_tax=0.0,
                use_liquidity_aware_cost=False,
                sleeve_count=1,
            )
        ).run(panel=panel, targets=targets)
        last = result.iloc[-1]
        self.assertAlmostEqual(float(last["gross_return"]), 18.0 / 15.5 - 1.0, places=6)
        self.assertAlmostEqual(float(last["net_return"]), 18.0 / 15.5 - 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
