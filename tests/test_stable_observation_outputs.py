from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


run_daily_monitor = _load_module(ROOT / "scripts" / "run_daily_monitor.py", "run_daily_monitor_test")
candidate_dashboard = _load_module(
    ROOT / "analysis" / "run_candidate_pool_quality_dashboard.py",
    "candidate_pool_quality_dashboard_test",
)


class StableObservationOutputTest(unittest.TestCase):
    def test_daily_monitor_markdown_contains_quality_and_review_sections(self) -> None:
        summary = type(
            "Summary",
            (),
            {
                "layer": "monitor / observation pool",
                "selection_date": "2026-04-16",
                "selected_count": 2,
                "candidate_count": 20,
                "label_type": "industry_excess",
                "signal_time": "close",
                "execution_time": "t+1 vwap",
                "system_mode": "stable_observation_cycle",
                "main_score_frozen": True,
                "execution_layer_frozen": True,
                "observation_layer_frozen": True,
                "new_research_gate_passed": False,
                "new_research_gate_reason": "does not improve candidate pool quality or action clarity",
            },
        )()
        observation = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "code": "000001.SZ",
                    "name": "A",
                    "target_weight": 0.5,
                    "score": 0.2,
                    "industry": "Bank",
                    "industry_leader_follow_score": 0.35,
                    "industry_leader_follow_rank": 1,
                    "industry_leader_follow_tag": "龙头扩散强",
                    "industry_leader_follow_detail": "龙头带动明显",
                    "overhead_density_score": 0.2,
                    "overhead_density_rank": 1,
                    "overhead_density_tag": "兑现压力轻",
                    "overhead_density_detail": "上方兑现压力轻",
                }
            ]
        )
        markdown = run_daily_monitor.build_daily_monitor_markdown(
            summary=summary,
            picks_json_path=Path("/tmp/picks.json"),
            observation_table=observation,
            strategy_table=observation.assign(trade_reason="hold", risk_state="full_risk"),
            factor_weights={"momentum_20": 0.1},
            metrics={
                "annual_return": 0.1,
                "sharpe": 1.0,
                "max_drawdown": -0.1,
                "avg_turnover": 0.02,
                "candidate_pool_quality": {
                    "headline": "候选池质量先看 top10",
                    "summary_rows": [{"pool_name": "top10", "avg_return": 0.05, "hit_rate": 0.6, "win_rate": 0.55, "avg_drawdown": -0.03, "payoff_ratio": 1.2}],
                    "failure_attribution_rows": [{"pool_name": "top10", "ranking_error_ratio": 0.2, "path_error_ratio": 0.3, "realized_ratio": 0.5, "failure_comment": "兑现路径错更明显"}],
                },
                "research_summary": {
                    "system_mode": "stable_observation_cycle",
                    "main_score_frozen": True,
                    "execution_layer_frozen": True,
                    "observation_layer_frozen": True,
                    "new_research_gate_passed": False,
                    "new_research_gate_reason": "does not improve candidate pool quality or action clarity",
                },
            },
            update_info=None,
            strategy_assessment={},
            research_summary={
                "system_mode": "stable_observation_cycle",
                "main_score_frozen": True,
                "execution_layer_frozen": True,
                "observation_layer_frozen": True,
                "new_research_gate_passed": False,
                "new_research_gate_reason": "does not improve candidate pool quality or action clarity",
            },
            performance_artifacts={},
        )
        self.assertIn("## Candidate Pool Quality", markdown)
        self.assertIn("## Observation Layer Detail", markdown)
        self.assertIn("## 复盘三问", markdown)

    def test_weekly_dashboard_markdown_contains_keep_frozen(self) -> None:
        markdown = candidate_dashboard.build_markdown(
            summary_df=pd.DataFrame([{"pool_name": "top10", "top_n": 10, "sample_count": 20, "avg_return": 0.03, "hit_rate": 0.6, "win_rate": 0.55, "avg_drawdown": -0.03, "avg_max_up": 0.07, "payoff_ratio": 1.2}]),
            rolling_df=pd.DataFrame(),
            tag_df=pd.DataFrame([{"group_type": "combined_observation_tag", "group_value": "龙头扩散强 / 兑现压力轻", "sample_count": 50, "avg_return_5d": 0.04, "avg_return_10d": 0.06, "win_rate_5d": 0.62, "avg_max_up_5d": 0.08, "avg_drawdown_5d": -0.03, "realization_gap_5d": 0.02, "path_quality": "可兑现收益", "path_comment": "持有时间拉长后收益延续更好"}]),
            failure_df=pd.DataFrame([{"pool_name": "top10", "top_n": 10, "sample_count": 20, "realized_ratio": 0.5, "ranking_error_ratio": 0.2, "path_error_ratio": 0.3, "failure_comment": "兑现路径错更明显"}]),
            payload={
                "research_config_path": "configs/research_production_default.json",
                "data_path": "data/sample.csv",
                "evaluation_start_date": "2026-01-01",
                "evaluation_end_date": "2026-04-16",
                "holding_horizon": 5,
                "system_mode": "stable_observation_cycle",
                "keep_frozen": True,
                "keep_frozen_reason": "当前还没有跨时间稳定的新证据",
                "new_research_gate_passed": False,
                "new_research_gate_reason": "does not improve candidate pool quality or action clarity",
                "headline": "候选池质量先看 top10",
                "trend_rows": [],
            },
        )
        self.assertIn("keep_frozen", markdown)
        self.assertIn("新增研究门槛", markdown)


if __name__ == "__main__":
    unittest.main()
