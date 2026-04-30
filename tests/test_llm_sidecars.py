from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llm_daily_ops_review import collect_ops_context
from scripts.llm_report_renderer import build_llm_prompt_facts, build_structured_facts, render_deterministic_report, resolve_artifacts, validate_numbers_grounded_in_json, validate_report_context


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class LLMSidecarTest(unittest.TestCase):
    def test_report_renderer_rejects_date_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = root / "data.csv"
            master.write_text("date,code\n2026-04-24,000001.SZ\n", encoding="utf-8")
            _write_json(root / "main" / "daily_monitor_20260424_picks.json", {"summary": {"selection_date": "2026-04-24"}, "observation_pool": []})
            _write_json(root / "elastic" / "under20_elastic_20260424_picks.json", {"summary": {"selection_date": "2026-04-23"}, "observation_pool": []})
            _write_json(root / "short" / "shortline_opportunity_20260424_cards.json", {"summary": {"selection_date": "2026-04-24"}, "cards": []})
            _write_json(root / "risk" / "risk_gate_20260424.json", {"selection_date": "2026-04-24", "status": "PASS"})
            args = argparse.Namespace(
                master_data_path=str(master),
                main_monitor_dir=str(root / "main"),
                elastic_monitor_dir=str(root / "elastic"),
                shortline_dir=str(root / "short"),
                risk_dir=str(root / "risk"),
                candidate_quality_root=str(root),
                date_key="20260424",
            )
            context = resolve_artifacts(args)
            errors = validate_report_context(context)
            self.assertIn("elastic_date_mismatch:20260423!=20260424", errors)

    def test_report_renderer_fallback_uses_structured_json(self) -> None:
        context = {
            "date_key": "20260424",
            "paths": {"main": Path("m"), "elastic": Path("e"), "shortline": Path("s"), "risk": Path("r")},
            "payloads": {
                "main": {"summary": {"selection_date": "2026-04-24"}, "observation_pool": [{"rank": 1, "code": "000001.SZ", "name": "测试A", "score": 0.8, "industry_leader_follow_tag": "龙头扩散强", "overhead_density_tag": "兑现压力轻"}]},
                "elastic": {"summary": {"selection_date": "2026-04-24"}, "observation_pool": [{"rank": 1, "code": "000002.SZ", "name": "测试B", "score": 0.6}]},
                "shortline": {"summary": {"selection_date": "2026-04-24"}, "cards": [{"rank": 1, "code": "000003.SZ", "name": "测试C", "shortline_score": 0.7}]},
                "risk": {"selection_date": "2026-04-24", "status": "PASS", "action_hint": "normal sizing", "max_drawdown": -0.1},
            },
            "candidate_quality": {"headline": "top5 稳定", "stable_observation_cycle_verdict": {"keep_frozen": True}},
        }
        args = argparse.Namespace(top_main=5, top_elastic=3, top_shortline=3)
        facts = build_structured_facts(context, args)
        markdown = render_deterministic_report(facts)
        self.assertIn("测试A", markdown)
        self.assertIn("龙头扩散强", markdown)
        self.assertIn("keep_frozen", markdown)

    def test_ops_review_marks_missing_artifact_as_high_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = root / "data.csv"
            master.write_text("date,code\n2026-04-24,000001.SZ\n", encoding="utf-8")
            ops = root / "ops"
            ops.mkdir(parents=True)
            (ops / "cron_preflight_evening.log").write_text('{"target_date":"2026-04-24","ok":false,"missing_after":["shortline"]}', encoding="utf-8")
            args = argparse.Namespace(
                master_data_path=str(master),
                main_monitor_dir=str(root / "main"),
                elastic_monitor_dir=str(root / "elastic"),
                shortline_dir=str(root / "short"),
                risk_dir=str(root / "risk"),
                ops_log_dir=str(ops),
                date_key="20260424",
            )
            context = collect_ops_context(args)
            self.assertFalse(context["system_ok"])
            self.assertEqual(context["risk_level"], "high")
            self.assertTrue(context["suggested_commands"])

    def test_report_renderer_rejects_ungrounded_numbers(self) -> None:
        facts = {"selection_date": "2026-04-24", "main_strategy": {"top_rows": [{"name": "A", "score": 0.8}]}}
        errors = validate_numbers_grounded_in_json("A 的分数是 0.8，但预计上涨 99%。", facts)
        self.assertIn("99", errors)
        self.assertNotIn("0.8", errors)

    def test_llm_prompt_facts_are_compact_and_keep_observation_tags(self) -> None:
        facts = {
            "selection_date": "2026-04-24",
            "validation": {"date_consistent": True},
            "main_strategy": {
                "top_rows": [
                    {
                        "rank": 1,
                        "code": "000001.SZ",
                        "name": "A",
                        "score": 0.8,
                        "industry_leader_follow_tag": "龙头扩散强",
                        "industry_leader_follow_detail": "x" * 200,
                        "overhead_density_tag": "兑现压力轻",
                    }
                ]
            },
            "elastic_pool": {"top_rows": []},
            "shortline": {"cards": []},
            "risk": {"status": "PASS"},
            "candidate_pool_quality": {"headline": "ok"},
        }
        prompt_facts = build_llm_prompt_facts(facts)
        self.assertEqual(prompt_facts["main_top"][0]["industry_leader_follow_tag"], "龙头扩散强")
        self.assertLess(len(prompt_facts["main_top"][0]["industry_leader_follow_detail"]), 100)


if __name__ == "__main__":
    unittest.main()
