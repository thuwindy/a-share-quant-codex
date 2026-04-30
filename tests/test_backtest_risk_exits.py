from __future__ import annotations

import unittest

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.backtest.engine import BacktestConfig, DailyBacktester


class BacktestRiskExitTest(unittest.TestCase):
    def test_stop_loss_forces_exit(self) -> None:
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
                "code": ["AAA", "AAA", "AAA"],
                "research_open": [100.0, 100.0, 85.0],
                "research_high": [101.0, 101.0, 86.0],
                "research_low": [99.0, 98.0, 84.0],
                "research_close": [100.0, 88.0, 85.0],
                "research_vwap": [100.0, 89.0, 85.0],
                "amount": [1e8, 1e8, 1e8],
                "can_buy": [True, True, True],
                "can_sell": [True, True, True],
            }
        )
        targets = pd.DataFrame(
            {
                "execution_date": pd.to_datetime(["2025-01-02"]),
                "sleeve": [0],
                "code": ["AAA"],
                "target_weight": [1.0],
            }
        )
        backtester = DailyBacktester(
            BacktestConfig(
                execution_price="vwap",
                execution_lag=1,
                holding_window=20,
                sleeve_count=1,
                stop_loss_pct=0.10,
            )
        )
        result, metrics = backtester.run(panel, targets)
        self.assertGreater(metrics["avg_forced_exit_count"], 0.0)
        self.assertIn("forced_exit_count", result.columns)


if __name__ == "__main__":
    unittest.main()
