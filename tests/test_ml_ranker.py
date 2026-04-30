from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.models.ml_ranker import MLLogisticConfig, MLLogisticRanker, MLRidgeConfig, MLRidgeRanker


class MLRidgeRankerTest(unittest.TestCase):
    def test_fit_and_predict(self) -> None:
        dates = pd.date_range("2025-01-01", periods=120, freq="B")
        rng = np.random.default_rng(42)
        x1 = rng.normal(size=len(dates))
        x2 = rng.normal(size=len(dates))
        y = 0.6 * x1 - 0.2 * x2 + rng.normal(scale=0.2, size=len(dates))
        frame = pd.DataFrame(
            {
                "date": dates,
                "factor_a": x1,
                "factor_b": x2,
                "label": y,
            }
        )
        ranker = MLRidgeRanker(
            factor_cols=["factor_a", "factor_b"],
            label_col="label",
            config=MLRidgeConfig(alpha_grid=[0.1, 1.0], cv_folds=3, min_rows=60),
        ).fit(frame)
        pred = ranker.predict(frame)
        self.assertEqual(len(pred), len(frame))
        self.assertIn("factor_a", ranker.weights_)
        self.assertTrue(np.isfinite(pred).all())

    def test_logistic_fit_and_predict_probability(self) -> None:
        dates = pd.date_range("2025-01-01", periods=160, freq="B")
        rng = np.random.default_rng(7)
        x1 = rng.normal(size=len(dates))
        x2 = rng.normal(size=len(dates))
        y = 0.7 * x1 - 0.3 * x2 + rng.normal(scale=0.25, size=len(dates))
        frame = pd.DataFrame(
            {
                "date": dates,
                "factor_a": x1,
                "factor_b": x2,
                "label": y,
            }
        )
        ranker = MLLogisticRanker(
            factor_cols=["factor_a", "factor_b"],
            label_col="label",
            config=MLLogisticConfig(c_grid=[0.1, 1.0], cv_folds=3, min_rows=80, positive_threshold=0.0),
        ).fit(frame)
        pred = ranker.predict(frame)
        self.assertEqual(len(pred), len(frame))
        self.assertTrue(((pred >= 0.0) & (pred <= 1.0)).all())
        self.assertIn("factor_a", ranker.weights_)


if __name__ == "__main__":
    unittest.main()
