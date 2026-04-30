from __future__ import annotations

import unittest

import pandas as pd

from ashare_quant.analysis.focused_tuning import _best_row


class FocusedTuningTests(unittest.TestCase):
    def test_best_row_prioritizes_sharpe_then_cost_and_turnover(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "scenario": "a",
                    "sharpe": 0.50,
                    "after_cost_return_drag": 0.04,
                    "avg_turnover": 0.20,
                    "max_drawdown": -0.25,
                    "annual_return": 0.10,
                },
                {
                    "scenario": "b",
                    "sharpe": 0.50,
                    "after_cost_return_drag": 0.03,
                    "avg_turnover": 0.22,
                    "max_drawdown": -0.20,
                    "annual_return": 0.09,
                },
                {
                    "scenario": "c",
                    "sharpe": 0.45,
                    "after_cost_return_drag": 0.01,
                    "avg_turnover": 0.10,
                    "max_drawdown": -0.10,
                    "annual_return": 0.08,
                },
            ]
        )

        best = _best_row(df)
        self.assertEqual(best["scenario"], "b")

    def test_best_row_prefers_tradable_candidates_when_requested(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "scenario": "monitor_best",
                    "sharpe": 0.70,
                    "after_cost_return_drag": 0.06,
                    "avg_turnover": 0.14,
                    "max_drawdown": -0.24,
                    "annual_return": 0.13,
                    "strategy_classification": "research candidate engine / monitor only",
                },
                {
                    "scenario": "tradable_good",
                    "sharpe": 0.58,
                    "after_cost_return_drag": 0.049,
                    "avg_turnover": 0.13,
                    "max_drawdown": -0.23,
                    "annual_return": 0.10,
                    "strategy_classification": "tradable prototype",
                },
            ]
        )

        best = _best_row(df, prefer_tradable=True)
        self.assertEqual(best["scenario"], "tradable_good")


if __name__ == "__main__":
    unittest.main()
