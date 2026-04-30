from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.csv_adapter import CSVDataSource


class CSVAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "bars.csv"
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "code": ["000001.SZ", "000001.SZ"],
                "open": [10.0, 10.2],
                "high": [10.5, 10.4],
                "low": [9.9, 10.1],
                "close": [10.3, 10.35],
                "volume": [1000.0, 1200.0],
                "amount": [10300.0, 12420.0],
                "market_cap": [1e9, 1.01e9],
                "industry": ["Bank", "Bank"],
                "is_st": [False, False],
                "is_suspended": [False, False],
                "can_buy": [True, True],
                "can_sell": [True, True],
            }
        )
        df.to_csv(self.path, index=False)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_load_daily_bars_filters_date_range(self) -> None:
        out = CSVDataSource(self.path).load_daily_bars(start="2024-01-03", end="2024-01-03")
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["date"], pd.Timestamp("2024-01-03"))

    def test_qfq_adjustment_uses_adj_factor(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "code": ["000001.SZ", "000001.SZ"],
                "open": [10.0, 20.0],
                "high": [11.0, 21.0],
                "low": [9.0, 19.0],
                "close": [10.0, 20.0],
                "volume": [1000.0, 1200.0],
                "amount": [10300.0, 12420.0],
                "market_cap": [1e9, 1.01e9],
                "industry": ["Bank", "Bank"],
                "is_st": [False, False],
                "is_suspended": [False, False],
                "can_buy": [True, True],
                "can_sell": [True, True],
                "adj_factor": [1.0, 2.0],
            }
        )
        df.to_csv(self.path, index=False)

        out = CSVDataSource(self.path, adjust="qfq").load_daily_bars()
        self.assertAlmostEqual(float(out.iloc[0]["close"]), 5.0)
        self.assertAlmostEqual(float(out.iloc[1]["close"]), 20.0)

    def test_price_views_keep_raw_trade_prices_and_adjusted_research_prices(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "code": ["000001.SZ", "000001.SZ"],
                "open": [10.0, 20.0],
                "high": [11.0, 21.0],
                "low": [9.0, 19.0],
                "close": [10.0, 20.0],
                "volume": [1000.0, 1200.0],
                "amount": [10300.0, 12420.0],
                "market_cap": [1e9, 1.01e9],
                "industry": ["Bank", "Bank"],
                "is_st": [False, False],
                "is_suspended": [False, False],
                "can_buy": [True, True],
                "can_sell": [True, True],
                "adj_factor": [1.0, 2.0],
            }
        )
        df.to_csv(self.path, index=False)

        out = CSVDataSource(self.path, adjust="qfq").load_daily_bars()
        self.assertAlmostEqual(float(out.iloc[0]["trade_close"]), 10.0)
        self.assertAlmostEqual(float(out.iloc[0]["research_close"]), 5.0)
        self.assertAlmostEqual(float(out.iloc[1]["trade_close"]), 20.0)
        self.assertAlmostEqual(float(out.iloc[1]["research_close"]), 20.0)


if __name__ == "__main__":
    unittest.main()
