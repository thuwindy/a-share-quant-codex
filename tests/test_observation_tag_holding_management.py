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

from analysis.run_observation_tag_holding_management_experiment import (
    assign_policy_decisions,
    define_policy_horizon,
    summarize_policy,
)


class ObservationTagHoldingManagementTest(unittest.TestCase):
    def test_define_policy_horizon_respects_tag_meaning(self) -> None:
        frame = pd.DataFrame(
            {
                "rank": [1, 2, 11],
                "industry_leader_follow_tag": ["龙头扩散强", "龙头扩散弱", "龙头扩散强"],
                "overhead_density_tag": ["兑现压力轻", "兑现压力大", "兑现压力轻"],
            }
        )
        policy = define_policy_horizon(frame, "mixed_tag_path")
        self.assertEqual(list(policy.astype(int)), [10, 3, 5])

    def test_policy_summary_has_status_and_comment(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-10", "2026-01-10"]),
                "rank": [1, 1],
                "industry_leader_follow_tag": ["龙头扩散强", "龙头扩散强"],
                "overhead_density_tag": ["兑现压力轻", "兑现压力轻"],
                "holding_return_3d": [0.01, 0.01],
                "holding_return_5d": [0.02, 0.02],
                "holding_return_10d": [0.03, 0.03],
                "holding_return_15d": [0.03, 0.03],
                "holding_return_20d": [0.03, 0.03],
                "holding_drawdown_3d": [-0.01, -0.01],
                "holding_drawdown_5d": [-0.01, -0.01],
                "holding_drawdown_10d": [-0.01, -0.01],
                "holding_drawdown_15d": [-0.01, -0.01],
                "holding_drawdown_20d": [-0.01, -0.01],
            }
        )
        baseline = summarize_policy(frame, policy_name="fixed_5d", policy_label="固定持有5天")
        variant = summarize_policy(frame, policy_name="mixed_tag_path", policy_label="标签驱动持有")
        summary = pd.DataFrame([baseline, variant])
        yearly = pd.DataFrame(
            [
                {"year": 2025, "policy_name": "fixed_5d", "policy_label": "固定持有5天", "sample_count": 1, "avg_holding_days": 5.0, "avg_return": 0.02, "win_rate": 1.0, "avg_drawdown": -0.01},
                {"year": 2026, "policy_name": "fixed_5d", "policy_label": "固定持有5天", "sample_count": 1, "avg_holding_days": 5.0, "avg_return": 0.02, "win_rate": 1.0, "avg_drawdown": -0.01},
                {"year": 2025, "policy_name": "mixed_tag_path", "policy_label": "标签驱动持有", "sample_count": 1, "avg_holding_days": 10.0, "avg_return": 0.03, "win_rate": 1.0, "avg_drawdown": -0.01},
                {"year": 2026, "policy_name": "mixed_tag_path", "policy_label": "标签驱动持有", "sample_count": 1, "avg_holding_days": 10.0, "avg_return": 0.03, "win_rate": 1.0, "avg_drawdown": -0.01},
            ]
        )
        decided = assign_policy_decisions(summary, yearly)
        self.assertIn("policy_status", decided.columns)
        self.assertIn("policy_comment", decided.columns)
        self.assertIn("role_assignment", decided.columns)
        self.assertTrue(decided["policy_comment"].astype(str).str.len().gt(0).all())


if __name__ == "__main__":
    unittest.main()
