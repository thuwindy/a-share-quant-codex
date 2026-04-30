from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from ashare_quant.pipeline import (
    DEFAULT_RESEARCH,
    build_candidate_pool_quality_payload,
    build_research_summary,
    resolve_factor_columns,
)


class PipelineGuardrailsTest(unittest.TestCase):
    def test_low_freq_experimental_factor_rejected_in_production(self) -> None:
        cfg = {
            **DEFAULT_RESEARCH,
            "factor_set": "custom",
            "factor_columns": ["pe_ttm"],
            "ranker_identity": "production_primary_ranker",
        }
        with self.assertRaisesRegex(ValueError, "low-frequency experimental factors are blocked"):
            resolve_factor_columns(cfg)

    def test_low_freq_experimental_factor_allowed_with_override(self) -> None:
        cfg = {
            **DEFAULT_RESEARCH,
            "factor_set": "custom",
            "factor_columns": ["pe_ttm"],
            "ranker_identity": "production_primary_ranker",
            "allow_low_freq_experimental_in_production": True,
        }
        raw_factor_cols, factor_cols = resolve_factor_columns(cfg)
        self.assertIn("pe_ttm", raw_factor_cols)
        self.assertIn("pe_ttm_neu", factor_cols)

    def test_research_audit_missing_is_explicit(self) -> None:
        cfg = {
            **DEFAULT_RESEARCH,
            "factor_columns": ["momentum_20_neu"],
            "research_audit_required": True,
            "ranker_identity": "research_ranker",
        }
        summary = build_research_summary(research_cfg=cfg, raw_factor_cols=["momentum_20"], metadata={"label_horizons": [5]})
        self.assertTrue(summary["audit_required"])
        self.assertEqual(summary["audit_status"], "missing")
        self.assertGreaterEqual(summary["audit_warnings_count"], 1)
        self.assertEqual(summary["role_assignment"], "research_factor")
        self.assertEqual(summary["system_mode"], "stable_observation_cycle")
        self.assertFalse(summary["new_research_gate_passed"])

    def test_observation_ranker_identity_maps_to_observation_label(self) -> None:
        cfg = {
            **DEFAULT_RESEARCH,
            "factor_columns": ["momentum_20_neu"],
            "ranker_identity": "observation_score",
        }
        summary = build_research_summary(research_cfg=cfg, raw_factor_cols=["momentum_20"], metadata={"label_horizons": [5]})
        self.assertEqual(summary["role_assignment"], "observation_label")

    def test_candidate_pool_quality_payload_adds_failure_horizon_columns(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=30)
        rows = []
        for idx, current_date in enumerate(dates):
            for code_idx, code in enumerate(["000001.SZ", "000002.SZ"]):
                close = 10.0 + idx * 0.2 + code_idx * 0.3
                rows.append(
                    {
                        "date": current_date,
                        "code": code,
                        "close": close,
                        "high": close * 1.04,
                        "low": close * 0.98,
                        "score": 1.0 - code_idx * 0.2,
                        "strategy_tradeable": True,
                    }
                )
        payload = build_candidate_pool_quality_payload(
            pd.DataFrame(rows),
            research_cfg={"holding_period": 20, "candidate_pool_top_ns": [2, 1]},
        )
        self.assertIn("summary_rows", payload)
        self.assertIn("failure_attribution_rows", payload)
        self.assertTrue(payload["failure_attribution_rows"])


if __name__ == "__main__":
    unittest.main()
