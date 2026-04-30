from __future__ import annotations

import unittest

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.tushare_sync import apply_fundamental_events_to_frame
from ashare_quant.data.tushare_sync import merge_raw_sync_table


class TushareFundamentalMergeTest(unittest.TestCase):
    def test_apply_fundamental_events_to_frame_uses_latest_announced_values(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-01", "2024-04-15", "2024-05-01"]),
                "code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "open": [10.0, 10.0, 10.0],
                "high": [11.0, 11.0, 11.0],
                "low": [9.0, 9.0, 9.0],
                "close": [10.5, 10.7, 10.8],
                "volume": [1000, 1000, 1000],
                "amount": [10000, 10000, 10000],
                "market_cap": [1e9, 1e9, 1e9],
                "industry": ["Bank", "Bank", "Bank"],
                "is_st": [False, False, False],
                "is_suspended": [False, False, False],
                "can_buy": [True, True, True],
                "can_sell": [True, True, True],
            }
        )
        events = pd.DataFrame(
            {
                "code": ["000001.SZ", "000001.SZ"],
                "ann_date": pd.to_datetime(["2024-04-10", "2024-04-20"]),
                "end_date": pd.to_datetime(["2024-03-31", "2024-03-31"]),
                "profit_quality_factor": [0.2, 0.1],
                "roe_factor": [0.08, 0.06],
                "earnings_growth_factor": [0.15, -0.05],
                "cashflow_quality_factor": [0.12, 0.11],
                "debt_to_asset": [0.45, 0.46],
                "distress_risk_flag": [False, True],
                "profit_warning_flag": [False, True],
                "forecast_type": ["", "预减"],
                "p_change_min": [pd.NA, -20.0],
                "p_change_max": [pd.NA, -10.0],
            }
        )
        enriched = apply_fundamental_events_to_frame(frame, events)
        self.assertTrue(pd.isna(enriched.loc[0, "profit_quality_factor"]))
        self.assertAlmostEqual(float(enriched.loc[1, "profit_quality_factor"]), 0.2)
        self.assertAlmostEqual(float(enriched.loc[2, "profit_quality_factor"]), 0.1)
        self.assertTrue(bool(enriched.loc[2, "profit_warning_flag"]))

    def test_merge_raw_sync_table_keeps_ann_date_key_without_forward_fill(self) -> None:
        existing = pd.DataFrame(
            {
                "code": ["000001.SZ"],
                "ann_date": pd.to_datetime(["2024-04-10"]),
                "end_date": pd.to_datetime(["2024-03-31"]),
                "source_endpoint": ["forecast_vip"],
                "forecast_type": ["预增"],
                "p_change_max": [30.0],
            }
        )
        updates = pd.DataFrame(
            {
                "code": ["000001.SZ", "000001.SZ"],
                "ann_date": pd.to_datetime(["2024-04-20", "2024-04-20"]),
                "end_date": pd.to_datetime(["2024-03-31", "2024-03-31"]),
                "source_endpoint": ["forecast_vip", "forecast_vip"],
                "forecast_type": ["预减", "预减"],
                "p_change_max": [-10.0, -10.0],
            }
        )
        merged = merge_raw_sync_table(
            existing,
            updates,
            key_columns=["code", "ann_date", "end_date", "source_endpoint"],
            sort_columns=["code", "ann_date", "end_date", "source_endpoint"],
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(
            merged["ann_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-04-10", "2024-04-20"],
        )
        self.assertEqual(
            merged["forecast_type"].tolist(),
            ["预增", "预减"],
        )


if __name__ == "__main__":
    unittest.main()
