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

from ashare_quant.analysis.latest_picks import (
    annotate_observation_layers,
    build_latest_picks_markdown,
    build_latest_picks_table,
)


class LatestPicksTest(unittest.TestCase):
    def test_annotate_observation_layers_keeps_leader_follow_and_overhead_tags(self) -> None:
        selection_ts = pd.Timestamp("2026-04-10")
        selected = pd.DataFrame(
            {
                "date": [selection_ts, selection_ts],
                "code": ["000001.SZ", "000002.SZ"],
                "score": [0.21, 0.18],
                "rank": [1, 2],
                "target_weight": [0.5, 0.5],
                "industry": ["Bank", "Bank"],
                "close": [10.0, 11.0],
                "amount": [1_000_000.0, 900_000.0],
                "market_cap": [10_000_000_000.0, 9_000_000_000.0],
                "industry_leader_follow_5": [0.42, 0.05],
                "overhead_density_20": [0.20, 0.70],
                "smart_money_inflow_20_neu": [0.2, -0.1],
            }
        )

        annotated = annotate_observation_layers(
            selected,
            research_cfg={
                "observe_ml_score": False,
                "observe_industry_breadth": False,
                "observe_industry_leader_follow": True,
                "industry_leader_follow_strong_threshold": 0.30,
                "industry_leader_follow_medium_threshold": 0.10,
                "observe_overhead_density": True,
                "overhead_density_strong_threshold": 0.55,
                "overhead_density_medium_threshold": 0.35,
                "observe_moneyflow_confirmation": True,
            },
            scored_df=selected.copy(),
            selection_ts=selection_ts,
            include_premium=False,
        )

        self.assertIn("industry_leader_follow_tag", annotated.columns)
        self.assertIn("overhead_density_tag", annotated.columns)
        self.assertEqual(list(annotated["industry_leader_follow_tag"]), ["龙头扩散强", "龙头扩散弱"])
        self.assertEqual(list(annotated["overhead_density_tag"]), ["兑现压力轻", "兑现压力大"])
        markdown = build_latest_picks_markdown(
            picks=annotated.loc[:, ["rank", "code", "target_weight", "score", "close", "industry", "industry_leader_follow_score", "industry_leader_follow_rank", "industry_leader_follow_tag", "overhead_density_score", "overhead_density_rank", "overhead_density_tag"]].assign(name=["A", "B"]),
            factor_weights={"momentum_20": 0.5},
            summary=type(
                "Summary",
                (),
                {
                    "layer": "monitor / observation pool",
                    "selection_date": "2026-04-10",
                    "train_start_date": "2025-01-01",
                    "train_end_date": "2026-04-09",
                    "train_rows": 10,
                    "train_dates": 5,
                    "candidate_count": 2,
                    "selected_count": 2,
                    "top_n": 2,
                    "label_horizon": 5,
                    "label_type": "industry_excess",
                    "adjust_mode": "qfq",
                    "data_path": "sample.csv",
                    "signal_time": "close",
                    "execution_time": "t+1 vwap",
                    "system_mode": "stable_observation_cycle",
                    "main_score_frozen": True,
                    "execution_layer_frozen": True,
                    "observation_layer_frozen": True,
                    "new_research_gate_passed": False,
                    "new_research_gate_reason": "does not improve candidate pool quality or action clarity",
                },
            )(),
        )
        self.assertIn("## Observation Layer", markdown)
        self.assertIn("## Stable Observation Cycle", markdown)
        self.assertIn("## 复盘三问", markdown)
        self.assertIn("龙头扩散", markdown)
        self.assertIn("兑现压力", markdown)

    def test_build_latest_picks_table_selects_latest_date_candidates(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sample.csv"
            config_path = Path(tmpdir) / "research.json"
            config_path.write_text(
                (
                    '{"label_horizon": 5, "rebalance_every": 5, "top_n": 3, '
                    '"factor_columns": ["momentum_20_neu", "reversal_5_neu", "liquidity_20_neu"], '
                    '"min_avg_amount_20": 0, "min_market_cap": 0, "min_price": 0, "min_listing_days": 0}'
                ),
                encoding="utf-8",
            )

            dates = pd.bdate_range("2024-01-02", periods=40)
            codes = [
                ("000001.SZ", "Ping An", "Bank"),
                ("000002.SZ", "Vanke", "RealEstate"),
                ("600000.SH", "PF Bank", "Bank"),
                ("600036.SH", "CMB", "Bank"),
                ("300750.SZ", "CATL", "Battery"),
                ("601318.SH", "PICC", "Insurance"),
            ]
            rows = []
            for date_idx, date in enumerate(dates):
                for code_idx, (code, name, industry) in enumerate(codes):
                    base = 10 + code_idx * 3 + date_idx * (0.08 + code_idx * 0.01)
                    close = base + (code_idx - 2) * 0.05
                    amount = 1_000_000 + code_idx * 200_000 + date_idx * 10_000
                    rows.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "code": code,
                            "open": base,
                            "high": base * 1.01,
                            "low": base * 0.99,
                            "close": close,
                            "volume": 100_000 + code_idx * 1_000,
                            "amount": amount,
                            "market_cap": 1_000_000_000 + code_idx * 100_000_000,
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

            picks, weights, summary = build_latest_picks_table(
                data_path=data_path,
                research_config_path=config_path,
                data_adjust="qfq",
            )

            self.assertEqual(summary.selected_count, 3)
            self.assertEqual(summary.selection_date, dates[-1].strftime("%Y-%m-%d"))
            self.assertEqual(len(picks), 3)
            self.assertEqual(picks["date"].nunique(), 1)
            self.assertAlmostEqual(picks["target_weight"].sum(), 1.0)
            self.assertIn("score", picks.columns)
            self.assertTrue(weights)

            markdown = build_latest_picks_markdown(picks=picks, factor_weights=weights, summary=summary)
            self.assertIn("Latest A-share Picks", markdown)
            self.assertIn("Selection Summary", markdown)
            self.assertIn("Picks", markdown)

    def test_build_latest_picks_table_flattens_multi_horizon_weights(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sample.csv"
            config_path = Path(tmpdir) / "research.json"
            config_path.write_text(
                (
                    '{"label_horizons": [3, 5, 10], "rebalance_every": 5, "top_n": 3, '
                    '"factor_combination_mode": "group", "ranker_type": "dynamic_icir", '
                    '"factor_columns": ["momentum_20_neu", "reversal_5_neu", "liquidity_20_neu"], '
                    '"min_avg_amount_20": 0, "min_market_cap": 0, "min_price": 0, "min_listing_days": 0}'
                ),
                encoding="utf-8",
            )

            dates = pd.bdate_range("2024-01-02", periods=80)
            codes = [
                ("000001.SZ", "Ping An", "Bank"),
                ("000002.SZ", "Vanke", "RealEstate"),
                ("600000.SH", "PF Bank", "Bank"),
                ("600036.SH", "CMB", "Bank"),
                ("300750.SZ", "CATL", "Battery"),
                ("601318.SH", "PICC", "Insurance"),
            ]
            rows = []
            for date_idx, date in enumerate(dates):
                for code_idx, (code, name, industry) in enumerate(codes):
                    base = 10 + code_idx * 3 + date_idx * (0.08 + code_idx * 0.01)
                    close = base + (code_idx - 2) * 0.05
                    amount = 1_000_000 + code_idx * 200_000 + date_idx * 10_000
                    rows.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "code": code,
                            "open": base,
                            "high": base * 1.01,
                            "low": base * 0.99,
                            "close": close,
                            "volume": 100_000 + code_idx * 1_000,
                            "amount": amount,
                            "market_cap": 1_000_000_000 + code_idx * 100_000_000,
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

            _, weights, _ = build_latest_picks_table(
                data_path=data_path,
                research_config_path=config_path,
                data_adjust="qfq",
            )

            self.assertTrue(weights)
            self.assertFalse({"3", "5", "10"} <= set(weights.keys()))


if __name__ == "__main__":
    unittest.main()
