from __future__ import annotations

import unittest
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.latest_picks import LatestPickSummary
from scripts.build_dual_board_report import _build_daily_checklist, _load_matching_risk_summary, _status_badge
from scripts.run_daily_monitor import build_daily_monitor_markdown
from scripts.send_pushplus_evening_brief import _render_within_pushplus_limit


class DailyReportOutputsTest(unittest.TestCase):
    def test_daily_monitor_markdown_includes_risk_governor_section(self) -> None:
        summary = LatestPickSummary(
            layer="monitor / observation pool",
            selection_date="2026-04-01",
            train_start_date="2024-04-01",
            train_end_date="2026-03-31",
            train_rows=100,
            train_dates=50,
            candidate_count=40,
            selected_count=12,
            top_n=12,
            label_horizon=5,
            label_type="neutralized_residual",
            adjust_mode="qfq",
            data_path="data/daily_monitor_slice.csv",
            signal_time="close",
            execution_time="t+1 close",
            system_mode="stable_observation_cycle",
            main_score_frozen=True,
            execution_layer_frozen=True,
            observation_layer_frozen=True,
            new_research_gate_passed=False,
            new_research_gate_reason="does not improve candidate pool quality or action clarity",
        )
        observation = pd.DataFrame(
            [{"rank": 1, "code": "000001.SZ", "name": "A", "target_weight": 0.1, "score": 0.9, "industry": "Bank"}]
        )
        strategy = pd.DataFrame(
            [{
                "rank": 1,
                "code": "000001.SZ",
                "name": "A",
                "target_weight": 0.1,
                "score": 0.9,
                "industry": "Bank",
                "trade_reason": "new_entry_stronger",
                "risk_state": "full_risk",
            }]
        )
        markdown = build_daily_monitor_markdown(
            summary=summary,
            picks_json_path=Path("outputs/daily_monitor_auto/daily_monitor_20260401_picks.json"),
            observation_table=observation,
            strategy_table=strategy,
            factor_weights={"5": 0.4},
            metrics={"annual_return": 0.1, "gross_annual_return": 0.12, "sharpe": 0.5, "max_drawdown": -0.2, "avg_turnover": 0.1, "after_cost_return_drag": 0.02, "signal_time": "close", "execution_time": "t+1 close"},
            update_info=None,
            strategy_assessment={"classification": "tradable prototype", "analyst_view": "ok", "trader_view": "ok", "exposure_warning": "none", "key_risks": []},
            research_summary={"system_mode": "stable_observation_cycle", "new_research_gate_passed": False, "new_research_gate_reason": "does not improve candidate pool quality or action clarity"},
            risk_summary={"status": "PASS", "annual_return": 0.1, "sharpe": 0.5, "max_drawdown": -0.2, "avg_turnover": 0.1, "selected_count": 12, "top_industry_weight": 0.15, "max_single_weight": 0.08},
        )
        self.assertIn("## Risk Governor", markdown)
        self.assertIn("status: `PASS`", markdown)
        self.assertIn("top_industry_weight", markdown)

    def test_dual_board_checklist_flags_date_mismatch(self) -> None:
        checklist = _build_daily_checklist(
            main_summary={"selection_date": "2026-04-01", "selected_count": 12},
            elastic_summary={"selection_date": "2026-03-31", "selected_count": 9},
            main_payload={"strategy_snapshot": [{"code": "000001.SZ"}]},
            elastic_payload={"observation_pool": [{"code": "000002.SZ"}]},
            compare_csv=ROOT / "outputs" / "tmp_compare.csv",
        )
        status_map = {item["check"]: item["status"] for item in checklist}
        self.assertEqual(status_map["same_selection_date"], "WARN")
        self.assertEqual(status_map["main_selected_nonzero"], "PASS")
        self.assertEqual(status_map["elastic_selected_nonzero"], "PASS")

    def test_dual_board_risk_loader_falls_back_to_empty_when_missing(self) -> None:
        summary = _load_matching_risk_summary(ROOT / "outputs" / "not_exists_for_test", "2026-04-01")
        self.assertEqual(summary, {})

    def test_status_badge_mapping(self) -> None:
        self.assertEqual(_status_badge("PASS"), "GREEN")
        self.assertEqual(_status_badge("WARN"), "YELLOW")
        self.assertEqual(_status_badge("BLOCK"), "RED")
        self.assertEqual(_status_badge("other"), "GRAY")

    def test_evening_brief_auto_compacts_when_content_is_too_large(self) -> None:
        long_detail = "这是为了测试 PushPlus 超长降级而构造的说明。" * 40
        main_payload = {
            "summary": {"selection_date": "2026-04-23", "selected_count": 20},
            "factor_weights": {"momentum_20_neu": 0.55, "momentum_60_neu": 0.45},
            "observation_pool": [
                {
                    "rank": idx + 1,
                    "code": f"00000{idx + 1}.SZ",
                    "name": f"Main{idx + 1}",
                    "industry": "软件服务",
                    "close": 10.0 + idx,
                    "score": 0.9 - idx * 0.05,
                    "target_weight": 0.02,
                    "market_cap": 10_000_000_000.0,
                    "momentum_20_neu": 0.4,
                    "momentum_60_neu": -0.2,
                    "ml_observation_tag": "原排序高 + ml_score低",
                    "industry_leader_follow_tag": "龙头扩散强",
                    "industry_leader_follow_detail": long_detail,
                    "overhead_density_tag": "兑现压力轻",
                    "overhead_density_detail": long_detail,
                    "moneyflow_confirmation_tag": "资金确认强",
                    "moneyflow_confirmation_detail": long_detail,
                    "fundamental_context_tag": "公告/预警偏正面",
                    "fundamental_context_detail": long_detail,
                }
                for idx in range(5)
            ],
        }
        elastic_payload = {
            "summary": {"selection_date": "2026-04-23", "selected_count": 20},
            "observation_pool": [
                {"rank": idx + 1, "code": f"30000{idx + 1}.SZ", "name": f"Elastic{idx + 1}", "industry": "电子", "close": 8.0 + idx, "score": 0.6 - idx * 0.03, "amount": 2_000_000_000.0, "market_cap": 8_000_000_000.0}
                for idx in range(5)
            ],
        }
        shortline_payload = {
            "summary": {"selection_date": "2026-04-23", "selected_count": 5},
            "cards": [
                {"rank": idx + 1, "code": f"60000{idx + 1}.SH", "name": f"Short{idx + 1}", "industry": "题材", "close": 12.0 + idx, "shortline_score": 0.8 - idx * 0.04, "current_streak": idx + 1, "buy_range": "10.0-10.5", "target_price": "11.2", "stop_loss": "9.8"}
                for idx in range(5)
            ],
        }
        risk_payload = {"status": "PASS", "action_hint": "normal sizing", "reasons": ["主池结构正常"]}
        title, content, profile, attempts = _render_within_pushplus_limit(
            main_payload,
            elastic_payload,
            shortline_payload,
            risk_payload,
            content_limit=5000,
        )
        self.assertEqual(title, "量化三合一晚报 2026-04-23")
        self.assertLessEqual(len(content), 5000)
        self.assertNotEqual(profile, "full")
        self.assertTrue(attempts)


if __name__ == "__main__":
    unittest.main()
