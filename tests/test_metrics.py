from __future__ import annotations

import unittest

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.backtest.metrics import max_drawdown, payoff_ratio, summarize_returns


class MetricsTest(unittest.TestCase):
    def test_max_drawdown(self) -> None:
        equity = pd.Series([1.0, 1.1, 1.2, 0.9, 1.0])
        self.assertAlmostEqual(max_drawdown(equity), -0.25)

    def test_summary_keys(self) -> None:
        rets = pd.Series([0.01, -0.005, 0.002, 0.003])
        summary = summarize_returns(rets)
        self.assertIn("sharpe", summary)
        self.assertIn("max_drawdown", summary)
        self.assertIn("annual_return", summary)

    def test_summary_handles_negative_terminal_equity_without_nan(self) -> None:
        rets = pd.Series([-1.2, 0.1, 0.2])
        summary = summarize_returns(rets)
        self.assertEqual(summary["annual_return"], -1.0)
        self.assertEqual(summary["max_drawdown"], -1.0)

    def test_payoff_ratio(self) -> None:
        rets = pd.Series([0.03, 0.01, -0.02, -0.01])
        self.assertAlmostEqual(payoff_ratio(rets), 4.0 / 3.0)
        summary = summarize_returns(rets)
        self.assertAlmostEqual(summary["payoff_ratio"], 4.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
