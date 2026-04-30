from __future__ import annotations

import unittest

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.audit_report import build_leakage_bias_checklist, build_strategy_change_log


class AuditReportTest(unittest.TestCase):
    def test_build_strategy_change_log_generates_baseline_and_deltas(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "scenario": "baseline",
                    "description": "baseline config",
                    "annual_return": 0.10,
                    "sharpe": 0.50,
                    "max_drawdown": -0.20,
                    "avg_turnover": 0.10,
                    "after_cost_return_drag": 0.03,
                },
                {
                    "scenario": "improved",
                    "description": "turn on better no-trade band",
                    "annual_return": 0.12,
                    "sharpe": 0.60,
                    "max_drawdown": -0.18,
                    "avg_turnover": 0.08,
                    "after_cost_return_drag": 0.02,
                },
            ]
        )
        out = build_strategy_change_log(df)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.iloc[0]["impact_summary"], "baseline")
        self.assertAlmostEqual(float(out.iloc[1]["delta_sharpe"]), 0.10, places=6)

    def test_build_leakage_bias_checklist_flags_missing_execution_lag(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "scenario": "warn_case",
                    "label_type": "raw",
                    "execution_time": "t+0 close",
                    "avg_turnover": 0.10,
                    "after_cost_return_drag": 0.02,
                },
                {
                    "scenario": "pass_case",
                    "label_type": "neutralized_residual",
                    "execution_lag": 1,
                    "execution_time": "t+1 close",
                    "avg_turnover": 0.10,
                    "after_cost_return_drag": 0.02,
                    "tradeable_ratio": 0.95,
                },
            ]
        )
        out = build_leakage_bias_checklist(df)
        self.assertEqual(len(out), 2)
        warn_row = out.loc[out["scenario"] == "warn_case"].iloc[0]
        pass_row = out.loc[out["scenario"] == "pass_case"].iloc[0]
        self.assertEqual(warn_row["overall_status"], "WARN")
        self.assertIn("execution_lag_lt_1_or_missing", str(warn_row["warnings"]))
        self.assertEqual(pass_row["overall_status"], "PASS")


if __name__ == "__main__":
    unittest.main()
