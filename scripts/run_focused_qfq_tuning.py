from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.focused_tuning import run_focused_qfq_tuning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a focused real-qfq tuning pass for Top N, sleeves, no-trade band, and fundamental veto.")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "daily_monitor_slice_fundamental.csv"))
    parser.add_argument("--research-config", default=str(ROOT / "configs" / "recommended_default_config.json"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_liquidity.json"))
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "focused_qfq_tuning"))
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    summary_df, recommended_config = run_focused_qfq_tuning(
        data_path=args.data_path,
        research_config_path=args.research_config,
        backtest_config_path=args.backtest_config,
        output_dir=args.output_dir,
        data_adjust=args.adjust,
    )
    print(
        json.dumps(
            {
                "recommended_config": recommended_config,
                "scenario_summary": summary_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    print(f"[OK] focused qfq tuning outputs saved to {args.output_dir}")
