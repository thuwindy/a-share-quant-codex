from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.ablation_runner import run_tradable_upgrade_ablation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the baseline-to-tradable ablation suite on an A-share daily dataset.")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--research-config", default=str(ROOT / "configs" / "recommended_default_config.json"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_liquidity.json"))
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "ablation_suite"))
    parser.add_argument("--walk-forward-compare", default=str(ROOT / "outputs" / "walk_forward_20260329_config_compare.csv"))
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    summary_df, payload = run_tradable_upgrade_ablation(
        data_path=args.data_path,
        research_config_path=args.research_config,
        backtest_config_path=args.backtest_config,
        output_dir=args.output_dir,
        data_adjust=args.adjust,
        walk_forward_compare_path=args.walk_forward_compare,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"[OK] ablation outputs saved to {args.output_dir}")
