from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.execution.paper_gate import evaluate_paper_gate, latest_risk_gate_json


class PaperGateTest(unittest.TestCase):
    def test_block_status_disables_paper_live_and_collects_block_reasons(self) -> None:
        summary = {
            "selection_date": "2026-04-01",
            "status": "BLOCK",
            "checks": [
                {"name": "max_drawdown", "ok": False, "severity": "block"},
                {"name": "sharpe", "ok": True, "severity": "warn"},
            ],
        }
        decision = evaluate_paper_gate(summary, risk_json_path="outputs/risk_governor/risk_gate_20260401.json")
        self.assertFalse(decision.allow_paper_live)
        self.assertEqual(decision.risk_status, "BLOCK")
        self.assertIn("max_drawdown", decision.block_reasons)
        self.assertEqual(decision.note, "paper_live_blocked_by_risk_governor")

    def test_non_block_status_allows_paper_live(self) -> None:
        summary = {
            "selection_date": "2026-04-01",
            "status": "WARN",
            "checks": [
                {"name": "max_drawdown", "ok": False, "severity": "warn"},
            ],
        }
        decision = evaluate_paper_gate(summary, risk_json_path="outputs/risk_governor/risk_gate_20260401.json")
        self.assertTrue(decision.allow_paper_live)
        self.assertEqual(decision.block_reasons, [])
        self.assertEqual(decision.note, "paper_live_allowed")

    def test_latest_risk_gate_json_uses_latest_date_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            (tmp / "risk_gate_20260331.json").write_text("{}", encoding="utf-8")
            (tmp / "risk_gate_20260401.json").write_text("{}", encoding="utf-8")
            latest = latest_risk_gate_json(tmp)
            self.assertEqual(latest.name, "risk_gate_20260401.json")


if __name__ == "__main__":
    unittest.main()
