from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.run_observation_tag_combo_research import (
    assign_combo_decisions,
    build_combo_columns,
    summarize_combo,
    summarize_combo_yearly,
)


class ObservationTagComboResearchTest(unittest.TestCase):
    def test_combo_summary_has_status_and_comment(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2025-01-10",
                        "2025-01-10",
                        "2026-01-10",
                        "2026-01-10",
                    ]
                ),
                "code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
                "rank": [1, 8, 2, 12],
                "industry_leader_follow_tag": ["龙头扩散强", "龙头扩散弱", "龙头扩散强", "龙头扩散弱"],
                "overhead_density_tag": ["兑现压力轻", "兑现压力大", "兑现压力轻", "兑现压力轻"],
                "holding_return_1d": [0.01, -0.01, 0.02, 0.0],
                "holding_return_3d": [0.02, -0.02, 0.03, 0.01],
                "holding_return_5d": [0.03, -0.03, 0.04, 0.01],
                "holding_return_10d": [0.05, -0.01, 0.06, 0.02],
                "holding_max_up_5d": [0.06, 0.01, 0.07, 0.02],
                "holding_max_drawdown_5d": [-0.01, -0.05, -0.01, -0.02],
            }
        )
        frame = build_combo_columns(frame)
        summary = pd.DataFrame(
            [
                summarize_combo(frame, combo_name="high_score", combo_label="高分"),
                summarize_combo(
                    frame,
                    combo_name="high_score_spread_strong_pressure_low",
                    combo_label="高分 + 扩散强 + 压力低",
                ),
            ]
        )
        yearly = summarize_combo_yearly(frame)
        summary = assign_combo_decisions(summary, yearly)

        self.assertIn("combo_status", summary.columns)
        self.assertIn("combo_comment", summary.columns)
        self.assertIn("role_assignment", summary.columns)
        self.assertTrue(summary["combo_status"].notna().all())
        self.assertTrue(summary["combo_comment"].astype(str).str.len().gt(0).all())


if __name__ == "__main__":
    unittest.main()
