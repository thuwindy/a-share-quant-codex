from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.tushare_sync import update_local_history_with_tushare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append Tushare daily-bar updates into a local A-share history file without re-pulling full history."
    )
    parser.add_argument(
        "--existing-path",
        default=str(ROOT / "data" / "a_share_daily.csv"),
        help="Path to your local base history CSV.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output path. Defaults to overwriting --existing-path.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional explicit update start date in YYYY-MM-DD. Defaults to the day after the latest local row.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional update end date in YYYY-MM-DD. Defaults to today.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Optional pause between Tushare API calls to reduce rate pressure.",
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
        raise FileNotFoundError(
            f"{existing_path} does not exist. Put your 10-year base history here first, then run this updater."
        )

    target_path, total_rows, source = update_local_history_with_tushare(
        existing_path=existing_path,
        output_path=args.output_path,
        start_date=args.start_date,
        end_date=args.end_date,
        http_url=args.http_url,
        proxy_url=args.proxy_url,
        bypass_system_proxy=args.bypass_system_proxy,
        pause_seconds=args.pause_seconds,
    )
    summary = source.last_summary_ if source is not None else None
    if summary is None:
        print(f"[OK] no Tushare pull was needed; dataset remains at {target_path}")
    else:
        print(
            "[OK] updated dataset",
            f"path={target_path}",
            f"rows={total_rows}",
            f"pull_start={summary.start_date}",
            f"pull_end={summary.end_date}",
            f"trade_dates={summary.trade_dates}",
            f"new_rows={summary.rows}",
            f"daily_basic={summary.used_daily_basic}",
            f"adj_factor={summary.used_adj_factor}",
            f"stk_limit={summary.used_stk_limit}",
            f"stock_st={summary.used_stock_st}",
        )
