from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.factors.neutralize import neutralize_by_size_and_industry


class NeutralizeTest(unittest.TestCase):
    def test_neutralized_column_exists(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"] * 8),
                "market_cap": [10, 20, 30, 40, 50, 60, 70, 80],
                "industry": ["A", "A", "B", "B", "C", "C", "D", "D"],
                "factor_x": np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float),
            }
        )
        out = neutralize_by_size_and_industry(df, ["factor_x"])
        self.assertIn("factor_x_neu", out.columns)
        self.assertEqual(len(out), len(df))


if __name__ == "__main__":
    unittest.main()
