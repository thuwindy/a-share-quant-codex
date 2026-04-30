from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.service.trade_api import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the lightweight paper-trade API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--sell-tax", type=float, default=0.001)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency reminder path
        raise SystemExit(f"uvicorn is not installed: {exc}")
    app = create_app(
        initial_cash=args.initial_cash,
        commission=args.commission,
        slippage=args.slippage,
        sell_tax=args.sell_tax,
    )
    uvicorn.run(app, host=args.host, port=args.port)
