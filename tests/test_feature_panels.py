from __future__ import annotations

import unittest

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.feature_panels import attach_premium_feature_panels


class FeaturePanelAlignmentTest(unittest.TestCase):
    def test_same_day_daily_features_stay_on_same_date(self) -> None:
        base = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "code": ["000001.SZ", "000001.SZ"],
                "close": [10.0, 10.2],
                "amount": [1_000_000.0, 1_100_000.0],
            }
        )
        flow = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "code": ["000001.SZ", "000001.SZ"],
                "buy_lg_amount_rate": [10.0, 20.0],
                "net_amount": [5.0, 8.0],
                "net_d5_amount": [10.0, 15.0],
            }
        )
        aligned = attach_premium_feature_panels(base, moneyflow_df=flow)
        self.assertEqual(float(aligned.loc[0, "flow_buy_lg_amount_rate"]), 10.0)
        self.assertEqual(float(aligned.loc[1, "flow_buy_lg_amount_rate"]), 20.0)

    def test_report_events_override_by_latest_report_date(self) -> None:
        base = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-01", "2024-04-15", "2024-05-01"]),
                "code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "close": [10.0, 10.5, 10.8],
                "amount": [1_000_000.0, 1_000_000.0, 1_000_000.0],
            }
        )
        report_rc = pd.DataFrame(
            {
                "code": ["000001.SZ", "000001.SZ"],
                "report_date": pd.to_datetime(["2024-04-10", "2024-04-20"]),
                "quarter": ["2024Q1", "2024Q1"],
                "org_name": ["机构A", "机构A"],
                "eps": [1.0, 1.2],
                "np": [100.0, 120.0],
                "roe": [10.0, 11.0],
                "rating": ["买入", "买入"],
                "max_price": [20.0, 22.0],
                "min_price": [18.0, 20.0],
            }
        )
        aligned = attach_premium_feature_panels(base, report_rc_df=report_rc)
        self.assertTrue(pd.isna(aligned.loc[0, "analyst_revision_score"]))
        self.assertAlmostEqual(float(aligned.loc[1, "analyst_revision_score"]), 0.2, places=6)
        self.assertGreater(float(aligned.loc[2, "analyst_revision_score"]), float(aligned.loc[1, "analyst_revision_score"]))

    def test_limit_sentiment_daily_features_align_on_same_date(self) -> None:
        base = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
                "code": ["000001.SZ", "000002.SZ", "000001.SZ"],
                "close": [10.0, 11.0, 10.5],
                "amount": [1_000_000.0, 1_200_000.0, 1_100_000.0],
            }
        )
        limit_sentiment = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
                "code": ["A", "B", "A", "B"],
                "limit": ["U", "U", "U", "D"],
                "open_times": [0, 1, 0, 0],
            }
        )
        aligned = attach_premium_feature_panels(base, limit_sentiment_df=limit_sentiment)
        self.assertEqual(float(aligned.loc[0, "market_limit_up_count"]), 2.0)
        self.assertEqual(float(aligned.loc[0, "market_limit_down_count"]), 0.0)
        self.assertGreater(float(aligned.loc[0, "market_limit_sentiment_score"]), 0.0)
        self.assertEqual(float(aligned.loc[2, "market_limit_up_count"]), 1.0)
        self.assertEqual(float(aligned.loc[2, "market_limit_down_count"]), 1.0)
        self.assertLess(float(aligned.loc[2, "market_limit_sentiment_score"]), float(aligned.loc[0, "market_limit_sentiment_score"]))


if __name__ == "__main__":
    unittest.main()
