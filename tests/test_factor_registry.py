from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.factors.factor_registry import factors_for_family, list_factor_families, resolve_family_factor_request
from ashare_quant.factors.technical import add_technical_factors


class FactorRegistryTest(unittest.TestCase):
    def test_registry_lists_requested_families(self) -> None:
        families = list_factor_families()
        for expected in ("momentum", "volatility", "turnover_liquidity", "money_flow", "size", "value_low_freq", "quality_low_freq", "rule_based"):
            self.assertIn(expected, families)

    def test_family_request_merges_family_and_explicit_factor(self) -> None:
        requested = resolve_family_factor_request(
            factor_families=["size"],
            factor_columns=["momentum_20_neu"],
        )
        self.assertIn("log_mkt_cap", requested)
        self.assertIn("momentum_20", requested)

    def test_add_technical_factors_generates_research_columns(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=140)
        rows = []
        for i, code in enumerate(["000001.SZ", "000002.SZ"]):
            for j, date in enumerate(dates):
                close = 10 + i + j * 0.05
                rows.append(
                    {
                        "date": date,
                        "code": code,
                        "open": close * 0.99,
                        "high": close * 1.01,
                        "low": close * 0.98,
                        "close": close,
                        "volume": 100000 + i * 1000,
                        "amount": 50000000 + i * 1000000,
                        "market_cap": 5_000_000_000 + i * 100_000_000,
                        "industry": "Bank" if i == 0 else "Broker",
                        "can_buy": True,
                        "can_sell": True,
                        "is_st": False,
                        "is_suspended": False,
                        "pe_ttm": 15.0,
                        "pb": 1.8,
                        "ps_ttm": 2.1,
                        "roe": 12.0,
                        "gross_margin": 35.0,
                        "debt_to_assets": 42.0,
                        "q_ocf_to_sales": 18.0,
                    }
                )
        out = add_technical_factors(pd.DataFrame(rows))
        for col in ("momentum_5", "momentum_10", "momentum_120", "price_vs_ma20", "trend_slope_20", "volatility_60", "drawdown_20", "log_mkt_cap", "log_float_mkt_cap", "pe_ttm", "roe", "operating_cashflow_ratio"):
            self.assertIn(col, out.columns)


if __name__ == "__main__":
    unittest.main()
