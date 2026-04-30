from __future__ import annotations

import unittest

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.universe import UniverseFilterConfig, apply_basic_universe_filters


class UniverseFilterTest(unittest.TestCase):
    def test_filters_block_new_low_liquidity_and_near_limit_names(self) -> None:
        dates = pd.bdate_range("2024-05-20", periods=12)
        rows = []
        for current_date in dates:
            rows.extend(
                [
                    {
                        "date": current_date,
                        "code": "000001.SZ",
                        "trade_open": 10.0,
                        "trade_high": 10.2,
                        "trade_low": 9.8,
                        "trade_close": 10.1,
                        "open": 10.0,
                        "high": 10.2,
                        "low": 9.8,
                        "close": 10.1,
                        "amount": 50_000_000.0,
                        "market_cap": 5_000_000_000.0,
                        "industry": "Bank",
                        "is_st": False,
                        "is_suspended": False,
                        "can_buy": True,
                        "can_sell": True,
                        "list_date": "2020-01-01",
                        "up_limit": 12.0,
                        "down_limit": 8.0,
                    },
                    {
                        "date": current_date,
                        "code": "000002.SZ",
                        "trade_open": 10.0,
                        "trade_high": 10.1,
                        "trade_low": 9.9,
                        "trade_close": 10.0,
                        "open": 10.0,
                        "high": 10.1,
                        "low": 9.9,
                        "close": 10.0,
                        "amount": 1_000_000.0,
                        "market_cap": 5_000_000_000.0,
                        "industry": "Bank",
                        "is_st": False,
                        "is_suspended": False,
                        "can_buy": True,
                        "can_sell": True,
                        "list_date": "2020-01-01",
                        "up_limit": 12.0,
                        "down_limit": 8.0,
                    },
                    {
                        "date": current_date,
                        "code": "000003.SZ",
                        "trade_open": 11.9,
                        "trade_high": 12.0,
                        "trade_low": 11.8,
                        "trade_close": 11.99,
                        "open": 11.9,
                        "high": 12.0,
                        "low": 11.8,
                        "close": 11.99,
                        "amount": 50_000_000.0,
                        "market_cap": 5_000_000_000.0,
                        "industry": "Tech",
                        "is_st": False,
                        "is_suspended": False,
                        "can_buy": True,
                        "can_sell": True,
                        "list_date": "2020-01-01",
                        "up_limit": 12.0,
                        "down_limit": 8.0,
                    },
                ]
            )
        df = pd.DataFrame(rows)
        out, summary = apply_basic_universe_filters(
            df,
            config=UniverseFilterConfig(
                min_listing_days=60,
                min_avg_amount_20=20_000_000.0,
                min_market_cap=1_000_000_000.0,
                min_price=3.0,
                near_limit_buffer=0.005,
            ),
            return_summary=True,
        )

        latest = out.loc[out["date"] == dates[-1]].copy()
        self.assertTrue(bool(latest.loc[latest["code"] == "000001.SZ", "tradeable"].iloc[0]))
        self.assertFalse(bool(latest.loc[latest["code"] == "000002.SZ", "tradeable"].iloc[0]))
        self.assertFalse(bool(latest.loc[latest["code"] == "000003.SZ", "can_buy"].iloc[0]))
        self.assertFalse(bool(latest.loc[latest["code"] == "000003.SZ", "tradeable"].iloc[0]))
        self.assertEqual(summary.rows_before, 36)
        self.assertEqual(summary.rows_after, 3)
        self.assertGreaterEqual(summary.near_limit_blocked, 1)

    def test_max_price_filter_blocks_high_price_names(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "date": "2024-05-31",
                    "code": "000001.SZ",
                    "trade_open": 12.0,
                    "trade_high": 12.5,
                    "trade_low": 11.8,
                    "trade_close": 12.2,
                    "open": 12.0,
                    "high": 12.5,
                    "low": 11.8,
                    "close": 12.2,
                    "amount": 50_000_000.0,
                    "market_cap": 5_000_000_000.0,
                    "industry": "Bank",
                    "is_st": False,
                    "is_suspended": False,
                    "can_buy": True,
                    "can_sell": True,
                    "list_date": "2020-01-01",
                },
                {
                    "date": "2024-05-31",
                    "code": "000002.SZ",
                    "trade_open": 28.0,
                    "trade_high": 29.0,
                    "trade_low": 27.5,
                    "trade_close": 28.8,
                    "open": 28.0,
                    "high": 29.0,
                    "low": 27.5,
                    "close": 28.8,
                    "amount": 50_000_000.0,
                    "market_cap": 5_000_000_000.0,
                    "industry": "Tech",
                    "is_st": False,
                    "is_suspended": False,
                    "can_buy": True,
                    "can_sell": True,
                    "list_date": "2020-01-01",
                },
            ]
        )
        out, summary = apply_basic_universe_filters(
            df,
            config=UniverseFilterConfig(
                min_listing_days=60,
                min_avg_amount_20=0.0,
                min_market_cap=0.0,
                min_price=3.0,
                max_price=20.0,
                near_limit_buffer=0.005,
            ),
            return_summary=True,
        )
        self.assertTrue(bool(out.loc[out["code"] == "000001.SZ", "tradeable"].iloc[0]))
        self.assertFalse(bool(out.loc[out["code"] == "000002.SZ", "tradeable"].iloc[0]))
        self.assertGreaterEqual(summary.price_blocked, 1)


if __name__ == "__main__":
    unittest.main()
