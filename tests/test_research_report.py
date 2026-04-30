from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.research_report import build_research_summary_markdown
from ashare_quant.data.research_slice import ResearchSliceProfile


class ResearchReportTest(unittest.TestCase):
    def test_build_research_summary_markdown_contains_core_sections(self) -> None:
        profile = ResearchSliceProfile(
            rows=1000,
            unique_codes=20,
            start_date="2020-01-01",
            end_date="2024-01-01",
            avg_rows_per_code=50.0,
            suspended_ratio=0.01,
            st_ratio=0.02,
            unknown_industry_ratio=1.0,
        )
        markdown = build_research_summary_markdown(
            profile=profile,
            metrics={"sharpe": 1.2, "annual_return": 0.3, "annual_volatility": 0.25, "max_drawdown": -0.1, "hit_rate": 0.55, "avg_turnover": 0.2},
            factor_weights={"momentum_20_neu": 0.5, "reversal_5_neu": -0.2},
            slice_path=Path("data/research_slice.csv"),
            adjust_mode="qfq",
        )
        self.assertIn("Dataset", markdown)
        self.assertIn("Backtest Metrics", markdown)
        self.assertIn("Prompt Seed", markdown)


if __name__ == "__main__":
    unittest.main()
