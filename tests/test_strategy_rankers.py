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

from ashare_quant.models.boost_ranker import BoostRankerConfig, TreeBoostRanker
from ashare_quant.models.linear_ranker import FixedWeightRanker


class StrategyRankersTest(unittest.TestCase):
    def test_fixed_weight_ranker(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        ranker = FixedWeightRanker(["a", "b"], {"a": 2.0, "b": 1.0}).fit(None)
        pred = ranker.predict(frame)
        self.assertGreater(float(pred.iloc[0]), float(pred.iloc[1]))

    def test_tree_boost_ranker_smoke(self) -> None:
        dates = pd.date_range("2025-01-01", periods=180, freq="B")
        rng = np.random.default_rng(7)
        x1 = rng.normal(size=len(dates))
        x2 = rng.normal(size=len(dates))
        y = 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=len(dates))
        frame = pd.DataFrame(
            {
                "date": dates,
                "f1": x1,
                "f2": x2,
                "label": y,
            }
        )
        ranker = TreeBoostRanker(
            factor_cols=["f1", "f2"],
            label_col="label",
            config=BoostRankerConfig(
                backend="lightgbm",
                cv_folds=2,
                min_rows=120,
                learning_rates=[0.1],
                n_estimators_grid=[20],
                max_depth_grid=[3],
            ),
        ).fit(frame)
        pred = ranker.predict(frame)
        self.assertEqual(len(pred), len(frame))
        self.assertIn("f1", ranker.weights_)


if __name__ == "__main__":
    unittest.main()
