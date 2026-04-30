from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.research_report import (
    extract_display_weights,
    write_research_summary,
    write_research_summary_json,
)
from ashare_quant.data.research_slice import build_research_slice
from ashare_quant.pipeline import run_research_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a manageable research slice from the full A-share daily CSV, run the backtest, and emit an LLM-ready summary."
    )
    parser.add_argument("--input-path", default=str(ROOT / "data" / "a_share_daily.csv"))
    parser.add_argument("--slice-path", default=str(ROOT / "data" / "research_slice.csv"))
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--max-codes", type=int, default=500, help="Use 0 to keep the full eligible universe.")
    parser.add_argument(
        "--suffixes",
        default="SH,SZ",
        help="Comma-separated market suffixes to keep in the research slice.",
    )
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--research-config", default=str(ROOT / "configs" / "research.json"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest.json"))
    parser.add_argument("--output-prefix", default="workflow_real")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    suffixes = tuple(part.strip().upper() for part in args.suffixes.split(",") if part.strip())

    profile = build_research_slice(
        input_path=args.input_path,
        output_path=args.slice_path,
        start_date=args.start_date,
        end_date=args.end_date,
        include_suffixes=suffixes,
        max_codes=args.max_codes,
    )
    print(json.dumps(profile.__dict__, ensure_ascii=False, indent=2))

    result, metrics, score_details, targets = run_research_pipeline(
        data_path=args.slice_path,
        research_config_path=args.research_config,
        backtest_config_path=args.backtest_config,
        data_adjust=args.adjust,
    )
    weights = extract_display_weights(score_details)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    universe_tag = "all" if args.max_codes <= 0 else str(args.max_codes)
    prefix = f"{args.output_prefix}_{args.adjust}_{universe_tag}"
    result.to_csv(out_dir / f"{prefix}_equity_curve.csv", index=False)
    targets.to_csv(out_dir / f"{prefix}_target_weights.csv", index=False)
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{prefix}_factor_weights.json").write_text(
        json.dumps(weights, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"{prefix}_score_details.json").write_text(
        json.dumps(score_details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_research_summary(
        out_dir / f"{prefix}_summary.md",
        profile=profile,
        metrics=metrics,
        factor_weights=weights,
        slice_path=args.slice_path,
        adjust_mode=args.adjust,
    )
    write_research_summary_json(
        out_dir / f"{prefix}_summary.json",
        profile=profile,
        metrics=metrics,
        factor_weights=weights,
        slice_path=args.slice_path,
        adjust_mode=args.adjust,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] workflow outputs saved to {out_dir}")
