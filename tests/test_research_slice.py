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

from ashare_quant.data.research_slice import build_research_slice


class ResearchSliceTest(unittest.TestCase):
    def test_build_research_slice_selects_top_codes_by_amount(self) -> None:
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            output_path = Path(tmpdir) / "slice.csv"
            df = pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
                    "code": ["000001.SZ", "000001.SZ", "600000.SH", "600000.SH", "430001.BJ", "430001.BJ"],
                    "open": [10, 10, 20, 20, 30, 30],
                    "high": [10, 10, 20, 20, 30, 30],
                    "low": [10, 10, 20, 20, 30, 30],
                    "close": [10, 11, 20, 21, 30, 31],
                    "volume": [1, 1, 1, 1, 1, 1],
                    "amount": [100, 100, 500, 500, 1000, 1000],
                    "market_cap": [1, 1, 2, 2, 3, 3],
                    "industry": ["Unknown"] * 6,
                    "is_st": [False] * 6,
                    "is_suspended": [False] * 6,
                    "can_buy": [True] * 6,
                    "can_sell": [True] * 6,
                    "adj_factor": [1.0] * 6,
                    "name": ["a", "a", "b", "b", "c", "c"],
                    "list_status": [pd.NA] * 6,
                    "list_date": [pd.NA] * 6,
                    "delist_date": [pd.NA] * 6,
                }
            )
            df.to_csv(input_path, index=False)

            profile = build_research_slice(
                input_path=input_path,
                output_path=output_path,
                start_date="2024-01-01",
                end_date="2024-01-31",
                include_suffixes=("SH", "SZ"),
                max_codes=1,
                chunk_size=2,
            )

            out = pd.read_csv(output_path)
            self.assertEqual(profile.unique_codes, 1)
            self.assertEqual(set(out["code"]), {"600000.SH"})

    def test_build_research_slice_supports_full_universe_when_max_codes_is_zero(self) -> None:
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            output_path = Path(tmpdir) / "slice.csv"
            df = pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
                    "code": ["000001.SZ", "000001.SZ", "600000.SH", "600000.SH", "430001.BJ", "430001.BJ"],
                    "open": [10, 10, 20, 20, 30, 30],
                    "high": [10, 10, 20, 20, 30, 30],
                    "low": [10, 10, 20, 20, 30, 30],
                    "close": [10, 11, 20, 21, 30, 31],
                    "volume": [1, 1, 1, 1, 1, 1],
                    "amount": [100, 100, 500, 500, 1000, 1000],
                    "market_cap": [1, 1, 2, 2, 3, 3],
                    "industry": ["Unknown"] * 6,
                    "is_st": [False] * 6,
                    "is_suspended": [False] * 6,
                    "can_buy": [True] * 6,
                    "can_sell": [True] * 6,
                    "adj_factor": [1.0] * 6,
                    "name": ["a", "a", "b", "b", "c", "c"],
                    "list_status": [pd.NA] * 6,
                    "list_date": [pd.NA] * 6,
                    "delist_date": [pd.NA] * 6,
                }
            )
            df.to_csv(input_path, index=False)

            profile = build_research_slice(
                input_path=input_path,
                output_path=output_path,
                start_date="2024-01-01",
                end_date="2024-01-31",
                include_suffixes=("SH", "SZ"),
                max_codes=0,
                chunk_size=2,
            )

            out = pd.read_csv(output_path)
            self.assertEqual(profile.unique_codes, 2)
            self.assertEqual(set(out["code"]), {"000001.SZ", "600000.SH"})


if __name__ == "__main__":
    unittest.main()
