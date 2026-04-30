from __future__ import annotations

import unittest

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.fundamental_veto import FundamentalVetoConfig, apply_fundamental_veto
from ashare_quant.portfolio.construction import build_portfolio_targets
from ashare_quant.portfolio.portfolio_optimizer import PortfolioConfig, optimize_portfolio_weights
from ashare_quant.regime_gate import RegimeGateConfig, apply_regime_gate
from ashare_quant.strategy_classifier import classify_strategy, summarize_portfolio_exposure


class StrategyUpgradeTest(unittest.TestCase):
    def test_fundamental_veto_blocks_rows_with_bad_quality_signals(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "code": ["000001.SZ", "000002.SZ"],
                "profit_quality_factor": [0.1, -0.3],
                "earnings_growth_factor": [0.05, -0.2],
                "cashflow_quality_factor": [0.02, -0.4],
                "roe_factor": [0.1, -0.1],
            }
        )
        out, summary = apply_fundamental_veto(
            df,
            config=FundamentalVetoConfig(enabled=True),
        )
        self.assertFalse(bool(out.loc[out["code"] == "000001.SZ", "fundamental_veto"].iloc[0]))
        self.assertTrue(bool(out.loc[out["code"] == "000002.SZ", "fundamental_veto"].iloc[0]))
        self.assertGreater(summary.vetoed_rows, 0)

    def test_fundamental_veto_can_block_high_overhead_resistance_names(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "code": ["000001.SZ", "000002.SZ"],
                "overhead_resistance": [0.15, -0.20],
            }
        )
        out, summary = apply_fundamental_veto(
            df,
            config=FundamentalVetoConfig(enabled=True, min_overhead_resistance=-0.10),
        )
        self.assertFalse(bool(out.loc[out["code"] == "000001.SZ", "fundamental_veto"].iloc[0]))
        self.assertTrue(bool(out.loc[out["code"] == "000002.SZ", "fundamental_veto"].iloc[0]))
        self.assertIn("overhead_resistance", summary.reason_counts)

    def test_regime_gate_marks_low_risk_when_breadth_and_style_break(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=12)
        rows = []
        for idx, current_date in enumerate(dates):
            for code_idx, code in enumerate(["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ", "000006.SZ"]):
                if idx < 6:
                    close = 10 + idx * 0.2 + code_idx * 0.1
                else:
                    shock = -1.5 if code_idx < 3 else -0.2
                    close = 11 + (idx - 6) * shock + code_idx * 0.05
                rows.append(
                    {
                        "date": current_date,
                        "code": code,
                        "research_close": close,
                        "market_cap": 5_000_000_000 + code_idx * 500_000_000,
                    }
                )
        out, summary = apply_regime_gate(
            pd.DataFrame(rows),
            config=RegimeGateConfig(
                enabled=True,
                breadth_lookback=3,
                market_vol_lookback=3,
                style_lookback=3,
                breadth_full_risk=0.55,
                breadth_low_risk=0.45,
            ),
        )
        day_states = out.groupby("date")["risk_state"].first()
        self.assertIn("low_risk", set(day_states))
        self.assertGreater(summary.low_risk_days, 0)

    def test_entry_advantage_threshold_reduces_new_entries(self) -> None:
        candidates = pd.DataFrame(
            {
                "code": ["A", "B", "C"],
                "score": [0.32, 0.31, 0.30],
                "industry": ["Tech", "Tech", "Bank"],
            }
        )
        selected = optimize_portfolio_weights(
            candidates,
            cfg=PortfolioConfig(
                top_n=2,
                weighting_method="equal",
                max_weight=1.0,
                industry_cap=1.0,
                min_holdings=2,
                entry_score_advantage_threshold=0.05,
            ),
            previous_weights={"B": 0.5, "C": 0.5},
            previous_ranks={"B": 1, "C": 2},
        )
        self.assertIn("B", set(selected["code"]))
        self.assertIn("C", set(selected["code"]))

    def test_industry_cap_is_hard_and_can_leave_cash(self) -> None:
        candidates = pd.DataFrame(
            {
                "code": ["A1", "A2", "B1", "B2"],
                "score": [0.40, 0.39, 0.38, 0.37],
                "industry": ["Tech", "Tech", "Bank", "Bank"],
            }
        )
        selected = optimize_portfolio_weights(
            candidates,
            cfg=PortfolioConfig(
                top_n=4,
                weighting_method="equal",
                max_weight=1.0,
                industry_cap=0.15,
                min_holdings=4,
            ),
        )
        industry_weight = selected.groupby("industry")["target_weight"].sum()
        self.assertLessEqual(float(industry_weight.max()), 0.15 + 1e-9)
        self.assertLess(float(selected["target_weight"].sum()), 1.0)

    def test_no_trade_band_does_not_block_new_entries_below_weight_threshold(self) -> None:
        candidates = pd.DataFrame(
            {
                "code": ["A", "B", "C"],
                "score": [0.9, 0.8, 0.7],
                "industry": ["Tech", "Bank", "Utility"],
            }
        )
        selected = optimize_portfolio_weights(
            candidates,
            cfg=PortfolioConfig(
                top_n=3,
                weighting_method="equal",
                max_weight=0.5,
                industry_cap=1.0,
                min_holdings=3,
                weight_change_threshold=0.4,
            ),
            previous_weights={"OLD": 1.0},
            previous_ranks={"OLD": 1},
        )
        self.assertGreater(float(selected["target_weight"].sum()), 0.0)
        self.assertEqual(int((selected["target_weight"] > 0).sum()), 3)

    def test_no_trade_band_does_not_renormalize_frozen_weights_into_concentration(self) -> None:
        candidates = pd.DataFrame(
            {
                "code": ["A", "B", "C"],
                "score": [0.9, 0.8, 0.7],
                "industry": ["Tech", "Bank", "Utility"],
            }
        )
        selected = optimize_portfolio_weights(
            candidates,
            cfg=PortfolioConfig(
                top_n=3,
                weighting_method="equal",
                max_weight=0.5,
                industry_cap=1.0,
                min_holdings=3,
                weight_change_threshold=0.4,
                rank_change_threshold=5,
            ),
            previous_weights={"A": 0.1},
            previous_ranks={"A": 1},
        )
        a_weight = float(selected.loc[selected["code"] == "A", "target_weight"].iloc[0])
        self.assertLess(a_weight, 0.5)

    def test_entry_filter_overhead_masks_only_new_entries(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3),
                "code": ["OLD", "NEW_BAD", "NEW_OK"] * 2,
                "score": [0.90, 0.89, 0.88, 0.90, 0.89, 0.88],
                "industry": ["Tech", "Tech", "Bank"] * 2,
                "tradeable": [True] * 6,
                "strategy_tradeable": [True] * 6,
                "risk_state": ["full_risk"] * 6,
                "risk_multiplier": [1.0] * 6,
                "overhead_resistance": [0.10, -0.20, 0.05, 0.10, -0.20, 0.05],
            }
        )
        targets = build_portfolio_targets(
            df,
            score_col="score",
            top_n=2,
            rebalance_every=1,
            sleeve_count=1,
            execution_lag=1,
            weighting_method="equal",
            max_weight=1.0,
            industry_cap=1.0,
            min_holdings=2,
            entry_filter_overhead_enabled=True,
            entry_filter_min_overhead_resistance=-0.05,
        )
        codes = set(targets["code"].astype(str))
        self.assertIn("OLD", codes)
        self.assertIn("NEW_OK", codes)
        self.assertNotIn("NEW_BAD", codes)

    def test_premium_entry_filter_masks_only_new_entries(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3),
                "code": ["OLD", "NEW_BLOCK", "NEW_OK"] * 2,
                "score": [0.90, 0.89, 0.88, 0.90, 0.89, 0.88],
                "industry": ["Tech", "Tech", "Bank"] * 2,
                "tradeable": [True] * 6,
                "strategy_tradeable": [True] * 6,
                "risk_state": ["full_risk"] * 6,
                "risk_multiplier": [1.0] * 6,
                "premium_entry_filter": [False, True, False] * 2,
                "premium_entry_filter_reason": ["", "weak_smart_money", ""] * 2,
            }
        )
        targets = build_portfolio_targets(
            df,
            score_col="score",
            top_n=2,
            rebalance_every=1,
            sleeve_count=1,
            execution_lag=1,
            weighting_method="equal",
            max_weight=1.0,
            industry_cap=1.0,
            min_holdings=2,
        )
        codes = set(targets["code"].astype(str))
        self.assertIn("OLD", codes)
        self.assertIn("NEW_OK", codes)
        self.assertNotIn("NEW_BLOCK", codes)

    def test_strategy_classifier_flags_monitor_only_when_costs_dominate(self) -> None:
        picks = pd.DataFrame(
            {
                "code": ["A", "B", "C"],
                "industry": ["Tech", "Tech", "Tech"],
                "target_weight": [0.4, 0.35, 0.25],
                "market_cap": [5e9, 6e9, 4e9],
            }
        )
        exposure = summarize_portfolio_exposure(picks)
        assessment = classify_strategy(
            metrics={
                "gross_annual_return": 0.03,
                "annual_return": -0.07,
                "sharpe": -0.2,
                "max_drawdown": -0.44,
                "avg_turnover": 0.31,
                "after_cost_return_drag": 0.10,
            },
            exposure_summary=exposure,
        )
        self.assertEqual(assessment.classification, "research candidate engine / monitor only")
        self.assertIn("style basket", assessment.exposure_warning.lower())


if __name__ == "__main__":
    unittest.main()
