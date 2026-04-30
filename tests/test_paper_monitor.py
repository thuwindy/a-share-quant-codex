from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.execution.paper import PaperExecutionConfig
from ashare_quant.execution.paper_monitor import (
    PaperRiskConfig,
    apply_paper_risk_overlay,
    assess_paper_risk,
    execute_due_orders,
    mark_to_market,
    stage_orders_for_next_execution,
)
from scripts.run_paper_monitor import _latest_monitor_json


class PaperMonitorTest(unittest.TestCase):
    def test_observation_pool_can_feed_paper_monitor_shape(self) -> None:
        payload = {
            "observation_pool": [
                {
                    "code": "000001.SZ",
                    "name": "PingAn",
                    "industry": "Bank",
                    "target_weight": 0.2,
                    "score": 0.8,
                }
            ]
        }
        strategy_snapshot = pd.DataFrame(payload.get("strategy_snapshot", []) or payload.get("observation_pool", []))
        self.assertFalse(strategy_snapshot.empty)
        self.assertEqual(strategy_snapshot.loc[0, "code"], "000001.SZ")

    def test_latest_monitor_json_prefers_latest_date_over_fast_suffix(self) -> None:
        with TemporaryDirectory() as tmpdir:
            monitor_dir = Path(tmpdir)
            (monitor_dir / "under20_elastic_20260424_picks.json").write_text("{}", encoding="utf-8")
            (monitor_dir / "under20_elastic_fast_20260409_picks.json").write_text("{}", encoding="utf-8")
            latest = _latest_monitor_json(monitor_dir)
            self.assertEqual(latest.name, "under20_elastic_20260424_picks.json")

    def test_execute_due_orders_updates_state_and_nav(self) -> None:
        latest_rows = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-03-31"),
                    "code": "000001.SZ",
                    "trade_close": 10.0,
                    "can_buy": True,
                    "can_sell": True,
                    "name": "PingAn",
                    "industry": "Bank",
                }
            ]
        )
        state = {
            "cash": 1_000.0,
            "peak_nav": 1_000.0,
            "last_nav": 1_000.0,
            "last_mark_date": None,
            "positions": {},
            "pending_orders": [
                {
                    "signal_date": "2026-03-30",
                    "execution_date": "2026-03-31",
                    "code": "000001.SZ",
                    "side": "BUY",
                    "quantity": 10.0,
                    "trade_reason": "new_entry_stronger",
                    "risk_state": "full_risk",
                }
            ],
            "fills": [],
            "nav_history": [],
        }
        state, executed, blocked = execute_due_orders(
            state=state,
            as_of_date="2026-03-31",
            latest_rows=latest_rows,
            config=PaperExecutionConfig(commission=0.0, slippage=0.0, sell_tax=0.0),
        )
        self.assertEqual(len(executed), 1)
        self.assertTrue(blocked.empty)
        self.assertAlmostEqual(state["cash"], 900.0)
        self.assertAlmostEqual(state["positions"]["000001.SZ"]["quantity"], 10.0)

        state, nav_snapshot, holdings = mark_to_market(
            state=state,
            as_of_date="2026-03-31",
            latest_rows=latest_rows,
        )
        self.assertAlmostEqual(nav_snapshot["nav"], 1_000.0)
        self.assertEqual(len(holdings), 1)
        self.assertAlmostEqual(float(holdings.loc[0, "market_value"]), 100.0)

    def test_soft_risk_overlay_blocks_new_entries_and_scales_existing(self) -> None:
        state = {
            "peak_nav": 100.0,
            "nav_history": [
                {
                    "date": "2026-03-30",
                    "nav": 91.0,
                    "cash": 10.0,
                    "market_value": 81.0,
                    "daily_return": -0.01,
                    "drawdown": -0.09,
                    "cash_ratio": 10.0 / 91.0,
                }
            ],
        }
        risk_snapshot = assess_paper_risk(
            state=state,
            strategy_risk_state="full_risk",
            config=PaperRiskConfig(
                drawdown_soft_limit=0.08,
                drawdown_hard_limit=0.12,
                soft_risk_multiplier=0.5,
                hard_risk_multiplier=0.0,
                block_new_entries_on_soft_risk=True,
            ),
        )
        self.assertFalse(risk_snapshot.allow_new_entries)
        self.assertAlmostEqual(risk_snapshot.target_risk_multiplier, 0.5)

        targets = pd.DataFrame(
            [
                {"code": "AAA.SZ", "target_weight": 0.40, "score": 0.3, "trade_reason": "weight_gap", "risk_state": "full_risk"},
                {"code": "BBB.SZ", "target_weight": 0.20, "score": 0.2, "trade_reason": "new_entry_stronger", "risk_state": "full_risk"},
            ]
        )
        adjusted, overlay = apply_paper_risk_overlay(
            strategy_snapshot=targets,
            current_codes={"AAA.SZ"},
            risk_snapshot=risk_snapshot,
            config=PaperRiskConfig(min_cash_buffer=0.05),
        )
        self.assertEqual(adjusted["code"].tolist(), ["AAA.SZ"])
        self.assertAlmostEqual(float(adjusted["paper_target_weight"].iloc[0]), 0.20)
        self.assertEqual(overlay["dropped_new_entries"], ["BBB.SZ"])

    def test_stage_orders_skips_tiny_turnover(self) -> None:
        state = {
            "last_nav": 100_000.0,
            "positions": {"AAA.SZ": {"quantity": 1_000.0, "avg_price": 10.0, "name": "AAA", "industry": "Tech"}},
        }
        latest_rows = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-03-30"),
                    "code": "AAA.SZ",
                    "trade_close": 10.0,
                    "can_buy": True,
                    "can_sell": True,
                    "name": "AAA",
                    "industry": "Tech",
                }
            ]
        )
        targets = pd.DataFrame(
            [
                {
                    "code": "AAA.SZ",
                    "paper_target_weight": 0.102,
                    "score": 0.1,
                    "trade_reason": "weight_gap",
                    "risk_state": "full_risk",
                    "name": "AAA",
                }
            ]
        )
        pending, staged = stage_orders_for_next_execution(
            state=state,
            targets=targets,
            signal_date="2026-03-30",
            execution_date="2026-03-31",
            latest_rows=latest_rows,
            config=PaperRiskConfig(min_trade_notional=5_000.0),
        )
        self.assertEqual(pending, [])
        self.assertTrue(staged.empty)


if __name__ == "__main__":
    unittest.main()
