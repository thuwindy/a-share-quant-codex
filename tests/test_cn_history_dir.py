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

from ashare_quant.data.cn_history_dir import normalize_cn_history_file


class CNHistoryDirTest(unittest.TestCase):
    def test_normalize_cn_history_file_maps_columns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001.SZ.csv"
            pd.DataFrame(
                {
                    "股票代码": ["000001.SZ", "000001.SZ"],
                    "股票名称": ["平安银行", "*ST平安"],
                    "交易日": ["20260320", "20260321"],
                    "开盘价": [10.0, None],
                    "最高价": [10.5, None],
                    "最低价": [9.8, None],
                    "收盘价": [10.4, None],
                    "成交量（手）": [100.0, None],
                    "成交额（千元）": [250.0, None],
                    "总市值（万元）": [123.0, 124.0],
                    "复权因子": [1.5, 1.6],
                    "当日涨停价": [10.4, 11.0],
                    "当日跌停价": [9.0, 10.0],
                }
            ).to_csv(path, index=False)

            out = normalize_cn_history_file(path)
            self.assertEqual(list(out["code"].unique()), ["000001.SZ"])
            self.assertEqual(float(out.iloc[0]["amount"]), 250000.0)
            self.assertEqual(float(out.iloc[0]["market_cap"]), 1230000.0)
            self.assertFalse(bool(out.iloc[0]["can_buy"]))
            self.assertTrue(bool(out.iloc[0]["can_sell"]))
            self.assertTrue(bool(out.iloc[1]["is_st"]))
            self.assertTrue(bool(out.iloc[1]["is_suspended"]))
            self.assertFalse(bool(out.iloc[1]["can_buy"]))


if __name__ == "__main__":
    unittest.main()
