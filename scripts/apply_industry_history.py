from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.industry_history import rewrite_daily_with_industry_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply point-in-time industry events to the canonical daily A-share CSV."
    )
    parser.add_argument("--existing-path", default=str(ROOT / "data" / "a_share_daily.csv"))
    parser.add_argument("--events-path", default=str(ROOT / "data" / "industry_events_cninfo.csv"))
    parser.add_argument("--output-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--chunk-size", type=int, default=300000)
    parser.add_argument(
        "--overwrite-all",
        action="store_true",
        help="Overwrite all industry values instead of only filling Unknown/missing rows.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    target_path, rows = rewrite_daily_with_industry_events(
        existing_path=Path(args.existing_path),
        events_path=Path(args.events_path),
        output_path=Path(args.output_path),
        chunk_size=args.chunk_size,
        overwrite_unknown_only=not args.overwrite_all,
    )
    print(f"[OK] applied industry events to {rows} rows -> {target_path}")
