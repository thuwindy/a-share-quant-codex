from __future__ import annotations

import unittest

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.performance_eval import build_calendar_metrics, build_equity_curve_frame


class PerformanceEvalTest(unittest.TestCase):
    def test_build_equity_curve_frame_adds_drawdown(self) -> None:
        result = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-06"]),
                "gross_return": [0.01, -0.02, 0.01],
                "net_return": [0.009, -0.021, 0.009],
                "gross_equity": [1.01, 0.9898, 0.999698],
                "equity": [1.009, 0.987811, 0.996701299],
                "turnover": [0.1, 0.2, 0.1],
                "cost": [0.001, 0.002, 0.001],
                "max_position_weight": [0.08, 0.08, 0.08],
            }
        )
        curve = build_equity_curve_frame(result)
        self.assertIn("drawdown", curve.columns)
        self.assertLessEqual(float(curve["drawdown"].min()), 0.0)

    def test_build_calendar_metrics_year_and_month(self) -> None:
        result = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-05", "2026-01-06"]),
                "gross_return": [0.01, -0.01, 0.02, -0.01],
                "net_return": [0.009, -0.011, 0.019, -0.011],
                "gross_equity": [1.01, 0.9999, 1.019898, 1.00969902],
                "equity": [1.009, 0.997901, 1.016861119, 1.005675647691],
                "turnover": [0.1, 0.1, 0.2, 0.1],
                "cost": [0.001, 0.001, 0.002, 0.001],
                "max_position_weight": [0.09, 0.09, 0.10, 0.10],
            }
        )
        yearly = build_calendar_metrics(result, freq="Y", annual_days=252)
        monthly = build_calendar_metrics(result, freq="M", annual_days=252)
        self.assertEqual(set(yearly["period"]), {"2025", "2026"})
        self.assertIn("payoff_ratio", monthly.columns)


if __name__ == "__main__":
    unittest.main()
