from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.tushare_sync import enrich_local_history_fundamentals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill point-in-time fundamental veto fields into a local A-share daily dataset using Tushare."
    )
    parser.add_argument(
        "--existing-path",
        default=str(ROOT / "data" / "daily_monitor_slice.csv"),
        help="Path to the canonical local daily dataset or research slice.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output path. Defaults to overwriting --existing-path.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Optional pause between Tushare API calls.",
    )
    parser.add_argument(
        "--http-url",
        default=None,
        help="Optional custom Tushare HTTP endpoint. Falls back to TUSHARE_HTTP_URL if unset.",
    )
    parser.add_argument(
        "--proxy-url",
        default=None,
        help="Optional HTTP proxy for requests. Falls back to TUSHARE_PROXY_URL if unset.",
    )
    parser.add_argument(
        "--bypass-system-proxy",
        action="store_true",
        help="Bypass macOS/system proxy discovery and force direct connections unless --proxy-url is set.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    existing_path = Path(args.existing_path)
    if not existing_path.exists():
        raise FileNotFoundError(f"{existing_path} does not exist.")

    target_path, rows, event_rows = enrich_local_history_fundamentals(
        existing_path=existing_path,
        output_path=args.output_path,
        http_url=args.http_url,
        proxy_url=args.proxy_url,
        bypass_system_proxy=args.bypass_system_proxy,
        pause_seconds=args.pause_seconds,
    )
    print(f"[OK] enriched fundamentals for {rows} rows using {event_rows} Tushare announcement events -> {target_path}")
