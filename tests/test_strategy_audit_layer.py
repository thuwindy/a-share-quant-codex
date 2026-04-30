from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.strategy_audit import check_future_leakage, check_signal_alignment


class StrategyAuditLayerTest(unittest.TestCase):
    def test_future_leakage_warns_when_position_matches_same_day_signal(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-02", periods=4),
                "stock_code": ["000001.SZ"] * 4,
                "signal_target": [0.0, 1.0, 1.0, 0.0],
                "position": [0.0, 1.0, 1.0, 0.0],
                "ret_1": [0.0, 0.01, 0.02, -0.01],
            }
        )
        report = check_future_leakage(df, "signal_target", "position", "ret_1")
        self.assertEqual(report["status"], "WARN")

    def test_signal_alignment_passes_when_position_is_shifted(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-02", periods=4),
                "stock_code": ["000001.SZ"] * 4,
                "signal_target": [0.0, 1.0, 1.0, 0.0],
                "position": [0.0, 0.0, 1.0, 1.0],
            }
        )
        report = check_signal_alignment(df, "signal_target", "position")
        self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
