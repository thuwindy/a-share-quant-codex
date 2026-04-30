from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.candidate_pool_quality import (
    add_candidate_path_columns,
    build_candidate_pool_quality_timeseries,
    build_stable_observation_cycle_verdict,
    summarize_failure_attribution,
    summarize_candidate_pool_quality,
    summarize_observation_tag_paths,
)


class CandidatePoolQualityTest(unittest.TestCase):
    def test_candidate_pool_summary_and_tag_paths(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=12)
        rows = []
        for idx, current_date in enumerate(dates):
            for code_idx, code in enumerate(["000001.SZ", "000002.SZ", "000003.SZ"]):
                close = 10.0 + code_idx + idx * (0.2 - code_idx * 0.03)
                rows.append(
                    {
                        "date": current_date,
                        "code": code,
                        "close": close,
                        "high": close * 1.03,
                        "low": close * 0.98,
                        "score": 0.9 - code_idx * 0.3,
                        "strategy_tradeable": True,
                        "industry_leader_follow_tag": "龙头扩散强" if code_idx == 0 else "龙头扩散弱",
                        "overhead_density_tag": "兑现压力轻" if code_idx != 1 else "兑现压力大",
                    }
                )
        frame = pd.DataFrame(rows)
        frame = add_candidate_path_columns(frame, horizons={5, 10})
        summary = summarize_candidate_pool_quality(frame, top_ns=[2, 1], holding_horizon=5)
        rolling = build_candidate_pool_quality_timeseries(frame, top_ns=[2, 1], holding_horizon=5, rolling_window=3)
        failure = summarize_failure_attribution(frame, top_ns=[2, 1], evaluation_horizon=5)
        tag_paths = summarize_observation_tag_paths(frame)
        verdict = build_stable_observation_cycle_verdict(summary, rolling, tag_paths, lookback_points=2)

        self.assertIn("pool_name", summary.columns)
        self.assertIn("avg_return", summary.columns)
        self.assertFalse(summary.empty)
        self.assertIn("ranking_error_ratio", failure.columns)
        self.assertIn("path_quality", tag_paths.columns)
        self.assertIn("path_comment", tag_paths.columns)
        self.assertIn("keep_frozen", verdict)


if __name__ == "__main__":
    unittest.main()
