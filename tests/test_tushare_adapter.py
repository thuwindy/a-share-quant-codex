from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.tushare_adapter import TushareDataSource
from ashare_quant.data.tushare_sync import (
    apply_stock_metadata_to_frame,
    merge_local_history_with_updates,
    normalize_local_history_frame,
)


class FakeTusharePro:
    def trade_cal(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({"cal_date": ["20260327"], "is_open": ["1"]})

    def stock_basic(self, **kwargs) -> pd.DataFrame:
        if kwargs.get("list_status") == "L":
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "symbol": ["000001"],
                    "name": ["平安银行"],
                    "area": ["深圳"],
                    "industry": ["Bank"],
                    "market": ["主板"],
                    "list_status": ["L"],
                    "list_date": ["19910403"],
                    "delist_date": [""],
                }
            )
        return pd.DataFrame()

    def daily(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260327"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.4],
                "vol": [100.0],
                "amount": [250.0],
            }
        )

    def daily_basic(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260327"],
                "total_mv": [123.0],
            }
        )

    def adj_factor(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260327"],
                "adj_factor": [1.5],
            }
        )

    def stk_limit(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260327"],
                "up_limit": [10.4],
                "down_limit": [9.0],
            }
        )

    def stock_st(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260327"], "name": ["ST样本"]})


class TushareAdapterTest(unittest.TestCase):
    def test_invoke_retries_retryable_errors(self) -> None:
        class RetryablePro:
            def __init__(self) -> None:
                self.calls = 0

            def daily(self, **kwargs) -> pd.DataFrame:
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("抱歉，您每分钟最多调用该接口500次")
                return pd.DataFrame({"ok": [1]})

        pro = RetryablePro()
        source = TushareDataSource(
            pro_client=pro,
            max_retries=3,
            retry_backoff_seconds=0.0,
            retry_backoff_cap_seconds=0.0,
        )
        out = source._call_required("daily")
        self.assertEqual(pro.calls, 3)
        self.assertEqual(int(out.iloc[0]["ok"]), 1)

    def test_build_pro_client_respects_custom_http_url(self) -> None:
        fake_ts = SimpleNamespace()
        fake_ts.pro_api_calls = []

        def pro_api(token: str = "") -> object:
            fake_ts.pro_api_calls.append(token)
            return SimpleNamespace()

        fake_ts.pro_api = pro_api

        with mock.patch.dict(sys.modules, {"tushare": fake_ts}):
            with mock.patch.dict(
                os.environ,
                {"TUSHARE_TOKEN": "demo-token", "TUSHARE_HTTP_URL": "http://example.test:8010"},
                clear=False,
            ):
                pro = TushareDataSource._build_pro_client()

        self.assertEqual(fake_ts.pro_api_calls, ["demo-token"])
        self.assertEqual(getattr(pro, "_DataApi__http_url"), "http://example.test:8010/")

    def test_build_pro_client_applies_proxy_environment(self) -> None:
        fake_ts = SimpleNamespace()

        def pro_api(token: str = "") -> object:
            return SimpleNamespace()

        fake_ts.pro_api = pro_api

        with mock.patch.dict(sys.modules, {"tushare": fake_ts}):
            with mock.patch.dict(os.environ, {"TUSHARE_TOKEN": "demo-token"}, clear=True):
                TushareDataSource._build_pro_client(proxy_url="http://127.0.0.1:7890")
                self.assertEqual(os.environ["HTTP_PROXY"], "http://127.0.0.1:7890")
                self.assertEqual(os.environ["HTTPS_PROXY"], "http://127.0.0.1:7890")
                self.assertEqual(os.environ["http_proxy"], "http://127.0.0.1:7890")
                self.assertEqual(os.environ["https_proxy"], "http://127.0.0.1:7890")

    def test_build_pro_client_can_bypass_system_proxy(self) -> None:
        fake_ts = SimpleNamespace()

        def pro_api(token: str = "") -> object:
            return SimpleNamespace()

        fake_ts.pro_api = pro_api

        with mock.patch.dict(sys.modules, {"tushare": fake_ts}):
            with mock.patch.dict(
                os.environ,
                {"TUSHARE_TOKEN": "demo-token", "HTTP_PROXY": "http://127.0.0.1:7890"},
                clear=True,
            ):
                TushareDataSource._build_pro_client(bypass_system_proxy=True)
                self.assertEqual(os.environ["HTTP_PROXY"], "")
                self.assertEqual(os.environ["HTTPS_PROXY"], "")
                self.assertEqual(os.environ["NO_PROXY"], "*")
                self.assertEqual(os.environ["no_proxy"], "*")

    def test_normalizes_daily_frame_to_repo_schema(self) -> None:
        source = TushareDataSource(pro_client=FakeTusharePro())
        out = source.load_daily_bars(start="2026-03-27", end="2026-03-27")
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["code"], "000001.SZ")
        self.assertEqual(row["industry"], "Bank")
        self.assertEqual(float(row["volume"]), 10000.0)
        self.assertEqual(float(row["amount"]), 250000.0)
        self.assertEqual(float(row["market_cap"]), 1_230_000.0)
        self.assertEqual(float(row["adj_factor"]), 1.5)
        self.assertFalse(bool(row["can_buy"]))
        self.assertTrue(bool(row["can_sell"]))
        self.assertTrue(bool(row["is_st"]))

    def test_merge_local_history_preserves_existing_rows_and_fills_metadata(self) -> None:
        existing = normalize_local_history_frame(
            pd.DataFrame(
                {
                    "date": ["2026-03-20"],
                    "code": ["000001.SZ"],
                    "open": [10.0],
                    "high": [10.2],
                    "low": [9.9],
                    "close": [10.1],
                    "volume": [1000.0],
                    "amount": [10100.0],
                    "market_cap": [1_200_000.0],
                    "industry": ["Bank"],
                    "is_st": [False],
                    "is_suspended": [False],
                    "can_buy": [True],
                    "can_sell": [True],
                    "adj_factor": [1.4],
                    "name": ["平安银行"],
                }
            )
        )
        updates = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-03-27")],
                "code": ["000001.SZ"],
                "open": [10.2],
                "high": [10.5],
                "low": [10.1],
                "close": [10.4],
                "volume": [10000.0],
                "amount": [250000.0],
                "market_cap": [pd.NA],
                "industry": [pd.NA],
                "is_st": [False],
                "is_suspended": [False],
                "can_buy": [True],
                "can_sell": [True],
                "adj_factor": [1.5],
                "name": [pd.NA],
                "list_status": [pd.NA],
                "list_date": [pd.NA],
                "delist_date": [pd.NA],
            }
        )
        merged = merge_local_history_with_updates(existing, updates)
        self.assertEqual(len(merged), 2)
        last_row = merged.iloc[-1]
        self.assertEqual(last_row["industry"], "Bank")
        self.assertEqual(last_row["name"], "平安银行")
        self.assertEqual(float(last_row["market_cap"]), 1_200_000.0)

    def test_apply_stock_metadata_fills_unknown_industry(self) -> None:
        frame = normalize_local_history_frame(
            pd.DataFrame(
                {
                    "date": ["2026-03-20"],
                    "code": ["000001.SZ"],
                    "open": [10.0],
                    "high": [10.2],
                    "low": [9.9],
                    "close": [10.1],
                    "volume": [1000.0],
                    "amount": [10100.0],
                    "market_cap": [1_200_000.0],
                    "industry": ["Unknown"],
                    "is_st": [False],
                    "is_suspended": [False],
                    "can_buy": [True],
                    "can_sell": [True],
                    "name": ["平安银行"],
                }
            )
        )
        metadata = pd.DataFrame(
            {
                "code": ["000001.SZ"],
                "industry": ["Bank"],
                "name": ["平安银行"],
                "list_status": ["L"],
                "list_date": ["19910403"],
                "delist_date": [""],
            }
        )
        enriched = apply_stock_metadata_to_frame(frame, metadata)
        self.assertEqual(enriched.iloc[0]["industry"], "Bank")
        self.assertEqual(enriched.iloc[0]["list_status"], "L")


if __name__ == "__main__":
    unittest.main()
