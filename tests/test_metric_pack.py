from __future__ import annotations

import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.metric_pack import build_standard_metric_pack, metric_pack_dataframe


class MetricPackTest(unittest.TestCase):
    def test_maps_turnover_and_cost_drag_from_monitor_metrics(self) -> None:
        metrics = {
            "annual_return": 0.10,
            "sharpe": 0.50,
            "max_drawdown": -0.20,
            "avg_turnover": 0.12,
            "after_cost_return_drag": 0.03,
            "payoff_ratio": 1.8,
            "strategy_assessment": {"classification": "tradable prototype"},
        }
        pack = build_standard_metric_pack(metrics, selection_date="2026-04-01", source_metrics_json="x.json")
        self.assertAlmostEqual(pack["annual_return"], 0.10)
        self.assertAlmostEqual(pack["sharpe"], 0.50)
        self.assertAlmostEqual(pack["max_drawdown"], -0.20)
        self.assertAlmostEqual(pack["turnover"], 0.12)
        self.assertAlmostEqual(pack["cost_drag"], 0.03)
        self.assertAlmostEqual(pack["payoff_ratio"], 1.8)
        self.assertEqual(pack["strategy_classification"], "tradable prototype")

    def test_cost_drag_falls_back_to_gross_minus_net(self) -> None:
        metrics = {
            "annual_return": 0.08,
            "gross_annual_return": 0.11,
            "sharpe": 0.3,
            "max_drawdown": -0.1,
            "turnover": 0.2,
        }
        pack = build_standard_metric_pack(metrics)
        self.assertAlmostEqual(pack["cost_drag"], 0.03)
        df = metric_pack_dataframe(pack)
        self.assertIn("turnover", df.columns)
        self.assertIn("cost_drag", df.columns)
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
