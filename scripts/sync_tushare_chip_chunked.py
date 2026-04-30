from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.tushare_adapter import TushareDataSource
from ashare_quant.data.tushare_sync import (
    load_or_empty_raw_sync_table,
    merge_raw_sync_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunked sync for Tushare cyq_perf to avoid long black-box runs.")
    parser.add_argument("--codes-path", required=True, help="CSV file containing a 'code' column.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--http-url", default=None)
    parser.add_argument("--proxy-url", default=None)
    parser.add_argument("--bypass-system-proxy", action="store_true")
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    codes = (
        pd.read_csv(args.codes_path, usecols=["code"])["code"]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_or_empty_raw_sync_table(output_path)
    source = TushareDataSource(
        pause_seconds=args.pause_seconds,
        http_url=args.http_url,
        proxy_url=args.proxy_url,
        bypass_system_proxy=args.bypass_system_proxy,
    )

    total = len(codes)
    processed = 0
    for idx in range(0, total, args.batch_size):
        batch = codes[idx : idx + args.batch_size]
        frame = source.load_chip_daily(
            start=args.start_date,
            end=args.end_date,
            codes=batch,
        )
        existing = merge_raw_sync_table(
            existing,
            frame,
            key_columns=["date", "code", "source_endpoint"],
            sort_columns=["date", "code", "source_endpoint"],
        )
        existing.to_csv(output_path, index=False)
        processed += len(batch)
        print(f"[OK] chip batch {processed}/{total} -> {len(existing)} rows @ {output_path}", flush=True)

    print(f"[DONE] wrote {len(existing)} rows -> {output_path}")


if __name__ == "__main__":
    main()
