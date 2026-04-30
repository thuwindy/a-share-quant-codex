from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.walk_forward import (
    build_walk_forward_splits,
    run_walk_forward_analysis,
)


class WalkForwardTest(unittest.TestCase):
    def test_build_walk_forward_splits_creates_multiple_folds(self) -> None:
        dates = pd.bdate_range("2020-01-01", "2024-12-31")
        splits = build_walk_forward_splits(dates=dates, train_years=2, test_years=1, step_years=1)
        self.assertGreaterEqual(len(splits), 2)
        self.assertLess(splits[0]["train_start"], splits[0]["train_end"])
        self.assertLess(splits[0]["test_start"], splits[0]["test_end"])

    def test_run_walk_forward_analysis_returns_case_summary(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sample.csv"
            research_path = Path(tmpdir) / "research.json"
            backtest_path = Path(tmpdir) / "backtest.json"
            research_path.write_text(
                (
                    '{"label_horizon": 5, "rebalance_every": 5, "top_n": 3, '
                    '"factor_columns": ["momentum_20_neu", "reversal_5_neu", "liquidity_20_neu"], '
                    '"min_avg_amount_20": 0, "min_market_cap": 0, "min_price": 0, "min_listing_days": 0}'
                ),
                encoding="utf-8",
            )
            backtest_path.write_text(
                '{"commission": 0.0003, "slippage": 0.0005, "sell_tax": 0.001, "annual_trading_days": 252}',
                encoding="utf-8",
            )

            dates = pd.bdate_range("2020-01-02", periods=900)
            codes = [
                ("000001.SZ", "Ping An", "Bank"),
                ("000002.SZ", "Vanke", "RealEstate"),
                ("600000.SH", "PF Bank", "Bank"),
                ("600036.SH", "CMB", "Bank"),
                ("300750.SZ", "CATL", "Battery"),
                ("601318.SH", "PICC", "Insurance"),
            ]
            rows = []
            for date_idx, current_date in enumerate(dates):
                for code_idx, (code, name, industry) in enumerate(codes):
                    drift = 0.03 + code_idx * 0.02
                    base = 20 + code_idx * 4 + date_idx * drift
                    shock = ((date_idx + code_idx) % 11 - 5) * 0.03
                    close = max(base + shock, 1.0)
                    rows.append(
                        {
                            "date": current_date.strftime("%Y-%m-%d"),
                            "code": code,
                            "open": close * 0.995,
                            "high": close * 1.01,
                            "low": close * 0.99,
                            "close": close,
                            "volume": 100_000 + code_idx * 5_000,
                            "amount": close * (100_000 + code_idx * 5_000),
                            "market_cap": 10_000_000_000 + code_idx * 500_000_000,
                            "industry": industry,
                            "is_st": False,
                            "is_suspended": False,
                            "can_buy": True,
                            "can_sell": True,
                            "adj_factor": 1.0,
                            "name": name,
                            "list_status": "",
                            "list_date": "",
                            "delist_date": "",
                        }
                    )
            pd.DataFrame(rows).to_csv(data_path, index=False)

            summary, folds_df, factor_df = run_walk_forward_analysis(
                data_path=data_path,
                research_config_path=research_path,
                backtest_config_path=backtest_path,
                train_years=2,
                test_years=1,
                step_years=1,
                data_adjust="qfq",
            )

            self.assertGreaterEqual(summary["folds"], 1)
            self.assertFalse(folds_df.empty)
            self.assertFalse(factor_df.empty)
            self.assertIn("mean_sharpe", summary)
            self.assertIn("stability_score", factor_df.columns)


if __name__ == "__main__":
    unittest.main()
