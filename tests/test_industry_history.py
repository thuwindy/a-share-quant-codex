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

from ashare_quant.data.industry_history import (
    apply_industry_events_to_frame,
    build_industry_events_from_frames,
    normalize_cninfo_change_frame,
)


class IndustryHistoryTest(unittest.TestCase):
    def test_normalize_cninfo_change_frame_keeps_effective_dates(self) -> None:
        change_df = pd.DataFrame(
            {
                "变更日期": ["2020-01-01", "2021-06-01"],
                "分类标准": ["巨潮行业分类标准", "巨潮行业分类标准"],
                "行业门类": ["制造业", "制造业"],
                "行业大类": ["电子", "计算机"],
                "行业中类": ["半导体", "软件开发"],
            }
        )
        out = normalize_cninfo_change_frame(change_df, canonical_code="000001.SZ", industry_level="大类")
        self.assertEqual(list(out["industry"]), ["电子", "计算机"])
        self.assertEqual(out["code"].unique().tolist(), ["000001.SZ"])

    def test_apply_industry_events_to_frame_forward_fills_by_date(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-06-01", "2021-07-01"],
                "code": ["000001.SZ"] * 3,
                "open": [10, 10, 10],
                "high": [11, 11, 11],
                "low": [9, 9, 9],
                "close": [10, 10, 10],
                "volume": [1, 1, 1],
                "amount": [10, 10, 10],
                "market_cap": [100, 100, 100],
                "industry": ["Unknown"] * 3,
                "is_st": [False] * 3,
                "is_suspended": [False] * 3,
                "can_buy": [True] * 3,
                "can_sell": [True] * 3,
                "adj_factor": [1.0] * 3,
                "name": ["A"] * 3,
                "list_status": [""] * 3,
                "list_date": [""] * 3,
                "delist_date": [""] * 3,
            }
        )
        events = pd.DataFrame(
            {
                "code": ["000001.SZ", "000001.SZ"],
                "effective_date": ["2020-03-01", "2021-06-01"],
                "industry": ["电子", "计算机"],
                "standard": ["巨潮"] * 2,
                "industry_level": ["大类"] * 2,
            }
        )
        out = apply_industry_events_to_frame(frame, industry_events=events)
        self.assertEqual(list(out["industry"]), ["Unknown", "电子", "计算机"])
        self.assertEqual(list(out["code"]), ["000001.SZ", "000001.SZ", "000001.SZ"])

    def test_build_industry_events_from_frames_aggregates_codes(self) -> None:
        frames = [
            (
                "000001.SZ",
                pd.DataFrame(
                    {
                        "变更日期": ["2020-01-01"],
                        "分类标准": ["巨潮行业分类标准"],
                        "行业大类": ["电子"],
                    }
                ),
            ),
            ("000002.SZ", pd.DataFrame()),
        ]
        out, summary = build_industry_events_from_frames(frames=frames, industry_level="大类")
        self.assertEqual(summary.codes_total, 2)
        self.assertEqual(summary.codes_with_events, 1)
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
