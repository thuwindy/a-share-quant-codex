from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.tushare_adapter import TushareDataSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether the configured Tushare token and HTTP endpoint can return basic data."
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
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many rows to request from index_basic.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    pro = TushareDataSource._build_pro_client(
        http_url=args.http_url,
        proxy_url=args.proxy_url,
        bypass_system_proxy=args.bypass_system_proxy,
    )
    frame = pro.index_basic(limit=args.limit)
    print(frame.to_string(index=False))
