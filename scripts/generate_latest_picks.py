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

from ashare_quant.analysis.latest_picks import (
    build_latest_picks_table,
    write_latest_picks_json,
    write_latest_picks_markdown,
)
from ashare_quant.data.research_slice import build_research_slice


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a latest-date A-share candidate list from a manageable research slice."
    )
    parser.add_argument(
        "--data-path",
        default="",
        help="Existing slice CSV to score directly. If omitted, the script first builds a slice from --input-path.",
    )
    parser.add_argument("--input-path", default=str(ROOT / "data" / "a_share_daily.csv"))
    parser.add_argument("--slice-path", default=str(ROOT / "data" / "research_slice_latest.csv"))
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--max-codes", type=int, default=200, help="Use 0 to keep the full eligible universe.")
    parser.add_argument(
        "--suffixes",
        default="SH,SZ",
        help="Comma-separated market suffixes to keep in the research slice.",
    )
    parser.add_argument("--prediction-date", default="")
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--research-config", default=str(ROOT / "configs" / "research.json"))
    parser.add_argument("--output-prefix", default="latest_picks")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    suffixes = tuple(part.strip().upper() for part in args.suffixes.split(",") if part.strip())

    data_path = Path(args.data_path) if args.data_path else Path(args.slice_path)
    if not args.data_path:
        profile = build_research_slice(
            input_path=args.input_path,
            output_path=data_path,
            start_date=args.start_date,
            end_date=args.end_date,
            include_suffixes=suffixes,
            max_codes=args.max_codes,
        )
        print(json.dumps(profile.__dict__, ensure_ascii=False, indent=2))

    picks, weights, summary = build_latest_picks_table(
        data_path=data_path,
        research_config_path=args.research_config,
        data_adjust=args.adjust,
        prediction_date=args.prediction_date or None,
    )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    prefix = f"{args.output_prefix}_{args.adjust}_{summary.selected_count}"
    picks.to_csv(out_dir / f"{prefix}.csv", index=False)
    write_latest_picks_markdown(out_dir / f"{prefix}.md", picks=picks, factor_weights=weights, summary=summary)
    write_latest_picks_json(out_dir / f"{prefix}.json", picks=picks, factor_weights=weights, summary=summary)

    print(json.dumps({"summary": summary.__dict__, "factor_weights": weights}, ensure_ascii=False, indent=2))
    print(f"[OK] latest picks saved to {out_dir}")
