from __future__ import annotations

import unittest

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.experiment_runner import ExperimentScenario, build_ablation_summary_markdown, run_single_experiment
from ashare_quant.pipeline import DEFAULT_BACKTEST, DEFAULT_RESEARCH


class ExperimentRunnerTest(unittest.TestCase):
    def test_run_single_experiment_emits_summary_and_detail_tables(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=180)
        codes = [
            ("000001.SZ", "Bank"),
            ("000002.SZ", "Bank"),
            ("600000.SH", "Broker"),
            ("600036.SH", "Broker"),
            ("300750.SZ", "Battery"),
            ("601318.SH", "Insurance"),
        ]
        rows = []
        for date_idx, current_date in enumerate(dates):
            for code_idx, (code, industry) in enumerate(codes):
                base = 10 + code_idx * 2 + date_idx * (0.05 + code_idx * 0.01)
                close = base + ((date_idx + code_idx) % 7 - 3) * 0.03
                rows.append(
                    {
                        "date": current_date,
                        "code": code,
                        "open": close * 0.995,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "volume": 100_000 + code_idx * 5_000,
                        "amount": 50_000_000 + code_idx * 2_000_000,
                        "market_cap": 5_000_000_000 + code_idx * 500_000_000,
                        "industry": industry,
                        "is_st": False,
                        "is_suspended": False,
                        "can_buy": True,
                        "can_sell": True,
                        "adj_factor": 1.0,
                        "list_date": "2020-01-01",
                    }
                )
        df = pd.DataFrame(rows)
        research_cfg = {
            **DEFAULT_RESEARCH,
            "factor_columns": ["momentum_20_neu", "reversal_5_neu", "liquidity_20_neu"],
            "min_avg_amount_20": 0,
            "min_market_cap": 0,
            "min_price": 0,
            "min_listing_days": 0,
        }
        backtest_cfg = {**DEFAULT_BACKTEST, "use_liquidity_aware_cost": False}
        scenario = ExperimentScenario(
            name="smoke",
            description="smoke",
            research_overrides={
                "label_horizons": [5],
                "label_type": "raw",
                "ranker_type": "static",
                "execution_lag": 1,
                "execution_price": "close",
                "top_n": 3,
                "sleeve_count": 1,
            },
            backtest_overrides={"use_liquidity_aware_cost": False},
        )

        summary, details = run_single_experiment(
            df=df,
            research_cfg=research_cfg,
            backtest_cfg=backtest_cfg,
            scenario=scenario,
        )

        self.assertIn("annual_return", summary)
        self.assertIn("ic_mean", summary)
        self.assertEqual(summary["scenario"], "smoke")
        self.assertIn("equity_curve", details)
        self.assertIn("cost_sensitivity", details)
        self.assertFalse(details["equity_curve"].empty)
        self.assertFalse(details["cost_sensitivity"].empty)

    def test_build_ablation_summary_markdown_includes_audit_sections(self) -> None:
        summary_df = pd.DataFrame(
            [
                {
                    "scenario": "baseline",
                    "gross_annual_return": 0.10,
                    "annual_return": 0.08,
                    "sharpe": 0.50,
                    "max_drawdown": -0.20,
                    "avg_turnover": 0.10,
                    "after_cost_return_drag": 0.02,
                    "strategy_classification": "monitor only",
                    "top_n": 20,
                    "weighting_method": "rank",
                    "sleeve_count": 5,
                    "use_liquidity_aware_cost": True,
                }
            ]
        )
        change_log_df = pd.DataFrame(
            [{"scenario": "baseline", "previous_scenario": "", "change_summary": "baseline", "impact_summary": "baseline"}]
        )
        leakage_check_df = pd.DataFrame(
            [{"scenario": "baseline", "overall_status": "PASS", "warnings": ""}]
        )
        markdown = build_ablation_summary_markdown(
            summary_df,
            change_log_df=change_log_df,
            leakage_check_df=leakage_check_df,
        )
        self.assertIn("Strategy Change Log", markdown)
        self.assertIn("Leakage And Bias Checklist", markdown)


if __name__ == "__main__":
    unittest.main()
