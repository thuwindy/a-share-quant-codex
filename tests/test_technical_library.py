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

from ashare_quant.factors.technical import add_technical_factors, factor_set_columns


class TechnicalLibraryTest(unittest.TestCase):
    def test_indicator_and_rule_columns_present(self) -> None:
        dates = pd.date_range("2025-01-01", periods=90, freq="B")
        base = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "code": ["000001.SZ"] * len(dates) + ["000002.SZ"] * len(dates),
                "open": np.linspace(10, 15, len(dates)).tolist() + np.linspace(20, 24, len(dates)).tolist(),
                "high": np.linspace(10.2, 15.2, len(dates)).tolist() + np.linspace(20.4, 24.4, len(dates)).tolist(),
                "low": np.linspace(9.8, 14.8, len(dates)).tolist() + np.linspace(19.6, 23.6, len(dates)).tolist(),
                "close": np.linspace(10.1, 15.1, len(dates)).tolist() + np.linspace(20.1, 24.1, len(dates)).tolist(),
                "amount": [1e8] * (len(dates) * 2),
                "market_cap": [5e9] * (len(dates) * 2),
            }
        )
        enriched = add_technical_factors(base)
        for col in [
            "rsi_14",
            "macd_hist_12_26_9",
            "ma_gap_5_20",
            "bollinger_z_20",
            "atr_14",
            "rule_ma_trend",
            "rule_macd_trend",
            "rule_breakout_20",
        ]:
            self.assertIn(col, enriched.columns)

    def test_factor_sets_include_rule_and_ta_lanes(self) -> None:
        self.assertIn("rule_ma_trend", factor_set_columns("rule_core"))
        self.assertIn("rsi_14", factor_set_columns("indicator_core"))


if __name__ == "__main__":
    unittest.main()
