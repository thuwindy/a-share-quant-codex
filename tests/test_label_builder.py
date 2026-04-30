from __future__ import annotations

import unittest

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.labels.label_builder import ExecutionSpec, add_label_columns


class LabelBuilderTest(unittest.TestCase):
    def test_close_execution_alignment_matches_t_plus_one_close_entry(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=8)
        df = pd.DataFrame(
            {
                "date": dates.tolist() * 2,
                "code": ["000001.SZ"] * len(dates) + ["000002.SZ"] * len(dates),
                "research_open": [9, 10, 11, 12, 13, 14, 15, 16] * 2,
                "research_close": [10, 11, 12, 13, 14, 15, 16, 17] * 2,
                "market_cap": [1e9] * (len(dates) * 2),
                "industry": ["Bank"] * len(dates) + ["Broker"] * len(dates),
            }
        )
        out = add_label_columns(
            df,
            horizons=[5],
            spec=ExecutionSpec(signal_time="close", execution_price="close", execution_lag=1, holding_window=5),
        )

        first = out.loc[out["code"] == "000001.SZ"].sort_values("date").iloc[0]
        expected = 16.0 / 11.0 - 1.0
        self.assertAlmostEqual(float(first["forward_return_5d"]), expected)
        self.assertEqual(pd.Timestamp(first["execution_date_5d"]), pd.Timestamp(dates[1]))
        self.assertEqual(pd.Timestamp(first["label_available_date_5d"]), pd.Timestamp(dates[6]))

    def test_open_execution_alignment_matches_t_plus_one_open_entry(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=8)
        df = pd.DataFrame(
            {
                "date": dates.tolist() * 2,
                "code": ["000001.SZ"] * len(dates) + ["000002.SZ"] * len(dates),
                "research_open": [9, 10, 11, 12, 13, 14, 15, 16] * 2,
                "research_close": [10, 11, 12, 13, 14, 15, 16, 17] * 2,
                "market_cap": [1e9] * (len(dates) * 2),
                "industry": ["Bank"] * len(dates) + ["Broker"] * len(dates),
            }
        )
        out = add_label_columns(
            df,
            horizons=[5],
            spec=ExecutionSpec(signal_time="close", execution_price="open", execution_lag=1, holding_window=5),
        )
        first = out.loc[out["code"] == "000001.SZ"].sort_values("date").iloc[0]
        expected = 15.0 / 10.0 - 1.0
        self.assertAlmostEqual(float(first["forward_return_5d"]), expected)
        self.assertEqual(pd.Timestamp(first["execution_date_5d"]), pd.Timestamp(dates[1]))
        self.assertEqual(pd.Timestamp(first["label_available_date_5d"]), pd.Timestamp(dates[5]))

    def test_vwap_execution_alignment_matches_t_plus_one_vwap_entry(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=8)
        df = pd.DataFrame(
            {
                "date": dates.tolist() * 2,
                "code": ["000001.SZ"] * len(dates) + ["000002.SZ"] * len(dates),
                "research_open": [9, 10, 11, 12, 13, 14, 15, 16] * 2,
                "research_vwap": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5] * 2,
                "research_close": [10, 11, 12, 13, 14, 15, 16, 17] * 2,
                "market_cap": [1e9] * (len(dates) * 2),
                "industry": ["Bank"] * len(dates) + ["Broker"] * len(dates),
            }
        )
        out = add_label_columns(
            df,
            horizons=[5],
            spec=ExecutionSpec(signal_time="close", execution_price="vwap", execution_lag=1, holding_window=5),
        )
        first = out.loc[out["code"] == "000001.SZ"].sort_values("date").iloc[0]
        expected = 15.0 / 10.5 - 1.0
        self.assertAlmostEqual(float(first["forward_return_5d"]), expected)
        self.assertEqual(pd.Timestamp(first["execution_date_5d"]), pd.Timestamp(dates[1]))
        self.assertEqual(pd.Timestamp(first["label_available_date_5d"]), pd.Timestamp(dates[5]))

    def test_industry_excess_label_is_cross_sectionally_centered(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=8)
        rows = []
        for code, industry, scale in [
            ("000001.SZ", "Bank", 1.0),
            ("000002.SZ", "Bank", 1.1),
            ("000003.SZ", "Broker", 1.2),
            ("000004.SZ", "Broker", 1.3),
        ]:
            for idx, current_date in enumerate(dates):
                rows.append(
                    {
                        "date": current_date,
                        "code": code,
                        "research_open": 10.0 + idx * scale,
                        "research_close": 10.2 + idx * scale,
                        "market_cap": 1e9 + idx * 1e7,
                        "industry": industry,
                    }
                )
        out = add_label_columns(df=pd.DataFrame(rows), horizons=[3], spec=ExecutionSpec(execution_lag=1))
        valid = out.dropna(subset=["forward_return_3d_industry_excess"]).copy()
        industry_means = valid.groupby(["date", "industry"])["forward_return_3d_industry_excess"].mean().round(10)
        self.assertTrue((industry_means == 0.0).all())


if __name__ == "__main__":
    unittest.main()
