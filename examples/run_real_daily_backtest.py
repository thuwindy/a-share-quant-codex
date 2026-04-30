from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.research_report import extract_display_weights
from ashare_quant.pipeline import run_research_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the research pipeline on a real daily A-share dataset.")
    parser.add_argument(
        "--data-path",
        default=str(ROOT / "data" / "a_share_daily.csv"),
        help="Path to the local daily dataset in the repo schema.",
    )
    parser.add_argument(
        "--adjust",
        choices=("none", "qfq", "hfq"),
        default="qfq",
        help="Optional price adjustment mode; qfq is usually the safest default for long-horizon daily research.",
    )
    parser.add_argument(
        "--research-config",
        default=str(ROOT / "configs" / "recommended_default_config.json"),
        help="Research config JSON path.",
    )
    parser.add_argument(
        "--backtest-config",
        default=str(ROOT / "configs" / "backtest_liquidity.json"),
        help="Backtest config JSON path.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist. Prepare your base history file first, then run scripts/update_tushare_daily_dataset.py."
        )

    result, metrics, score_details, targets = run_research_pipeline(
        data_path=data_path,
        research_config_path=args.research_config,
        backtest_config_path=args.backtest_config,
        data_adjust=args.adjust,
    )
    weights = extract_display_weights(score_details)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    suffix = f"real_{args.adjust}"
    result.to_csv(out_dir / f"{suffix}_equity_curve.csv", index=False)
    targets.to_csv(out_dir / f"{suffix}_target_weights.csv", index=False)
    (out_dir / f"{suffix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"{suffix}_factor_weights.json").write_text(
        json.dumps(weights, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"{suffix}_score_details.json").write_text(
        json.dumps(score_details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] saved real-daily outputs to {out_dir}")
