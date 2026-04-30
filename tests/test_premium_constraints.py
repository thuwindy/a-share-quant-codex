from __future__ import annotations

import unittest

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.premium_constraints import PremiumConstraintConfig, apply_premium_dynamic_constraints


class PremiumConstraintTest(unittest.TestCase):
    def test_smart_money_and_warning_flags_block_new_entries(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"] * 3),
                "code": ["A", "B", "C"],
                "smart_money_inflow_20": [0.10, -0.05, 0.20],
                "profit_warning_flag": [False, False, True],
                "distress_risk_flag": [False, False, False],
            }
        )
        out, summary = apply_premium_dynamic_constraints(
            df,
            config=PremiumConstraintConfig(
                enabled=True,
                entry_filter_smart_money_enabled=True,
                min_smart_money_inflow=0.0,
                entry_filter_profit_warning_enabled=True,
            ),
        )
        self.assertFalse(bool(out.loc[out["code"] == "A", "premium_entry_filter"].iloc[0]))
        self.assertTrue(bool(out.loc[out["code"] == "B", "premium_entry_filter"].iloc[0]))
        self.assertTrue(bool(out.loc[out["code"] == "C", "premium_entry_filter"].iloc[0]))
        self.assertIn("weak_smart_money", summary.reason_counts)
        self.assertIn("profit_warning", summary.reason_counts)

    def test_limit_sentiment_gate_combines_with_existing_risk(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "code": ["A", "A"],
                "risk_state": ["full_risk", "half_risk"],
                "risk_multiplier": [1.0, 0.5],
                "market_limit_sentiment_score": [-0.30, 0.10],
            }
        )
        out, summary = apply_premium_dynamic_constraints(
            df,
            config=PremiumConstraintConfig(
                enabled=True,
                market_gate_limit_sentiment_enabled=True,
                limit_sentiment_half_threshold=0.0,
                limit_sentiment_low_threshold=-0.20,
                half_risk_multiplier=0.5,
                low_risk_multiplier=0.0,
            ),
        )
        self.assertEqual(str(out.loc[0, "premium_risk_state"]), "low_risk")
        self.assertAlmostEqual(float(out.loc[0, "risk_multiplier"]), 0.0, places=8)
        self.assertEqual(str(out.loc[1, "risk_state"]), "half_risk")
        self.assertAlmostEqual(float(out.loc[1, "risk_multiplier"]), 0.5, places=8)
        self.assertEqual(summary.low_risk_days, 1)


if __name__ == "__main__":
    unittest.main()
