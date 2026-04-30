from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.vectorized_strategy_sandbox import run_sandbox


class VectorizedStrategySandboxTest(unittest.TestCase):
    def test_sandbox_aligns_position_and_returns(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=220)
        rows = []
        for code_idx, code in enumerate(["000001.SZ", "000002.SZ", "000003.SZ"]):
            for i, date in enumerate(dates):
                close = 10 + code_idx + i * 0.03
                rows.append(
                    {
                        "date": date,
                        "stock_code": code,
                        "open": close * 0.99,
                        "high": close * 1.01,
                        "low": close * 0.98,
                        "close": close,
                        "volume": 100000,
                        "amount": 50000000,
                    }
                )
        detail, portfolio, report = run_sandbox(pd.DataFrame(rows), template="ma_trend")
        self.assertEqual(len(detail), len(rows))
        self.assertEqual(len(portfolio), len(dates))
        self.assertIn("net_nav", portfolio.columns)
        self.assertIn(report["overall_status"], {"PASS", "WARN"})


if __name__ == "__main__":
    unittest.main()
