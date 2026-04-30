from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.run_factor_family_research import run_factor_family_research


class FactorFamilyResearchTest(unittest.TestCase):
    def test_research_script_outputs_summary_tables(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=160)
        codes = [f"{i:06d}.SZ" for i in range(1, 11)]
        industries = ["Bank", "Broker", "Battery", "Consumer", "Software", "Chem", "Auto", "Media", "Power", "Defense"]
        rows = []
        for date_idx, date in enumerate(dates):
            for code_idx, code in enumerate(codes):
                close = 10 + code_idx * 0.5 + date_idx * (0.02 + code_idx * 0.002)
                rows.append(
                    {
                        "date": date,
                        "code": code,
                        "open": close * 0.995,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "volume": 100000 + code_idx * 2000,
                        "amount": 50000000 + code_idx * 1000000,
                        "market_cap": 5_000_000_000 + code_idx * 300_000_000,
                        "industry": industries[code_idx],
                        "is_st": False,
                        "is_suspended": False,
                        "can_buy": True,
                        "can_sell": True,
                        "adj_factor": 1.0,
                        "list_date": "2020-01-01",
                    }
                )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path = tmp / "panel.csv"
            config_path = tmp / "research.json"
            output_dir = tmp / "outputs"
            pd.DataFrame(rows).to_csv(data_path, index=False)
            config_path.write_text(
                json.dumps(
                    {
                        "factor_set": "custom",
                        "factor_families": ["momentum"],
                        "factor_columns": [],
                        "label_type": "future_return_5d",
                        "label_horizons": [5],
                        "min_listing_days": 0,
                        "min_avg_amount_20": 0,
                        "min_market_cap": 0,
                        "min_price": 0,
                        "include_fundamental": False,
                        "include_flow": False,
                        "include_regime": False
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            run_factor_family_research(
                data_path=data_path,
                config_path=config_path,
                output_dir=output_dir,
            )
            self.assertTrue((output_dir / "factor_summary.csv").exists())
            self.assertTrue((output_dir / "candidate_pool_quality.csv").exists())
            self.assertTrue((output_dir / "factor_family_research.md").exists())
            summary = pd.read_csv(output_dir / "factor_summary.csv")
            self.assertIn("ic_mean", summary.columns)
            self.assertIn("ir", summary.columns)
            self.assertIn("research_status", summary.columns)
            self.assertIn("research_comment", summary.columns)
            self.assertIn("role_assignment", summary.columns)


if __name__ == "__main__":
    unittest.main()
