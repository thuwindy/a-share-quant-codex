from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.service.dashboard_api import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the lightweight FastAPI dashboard for latest monitor outputs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--root-dir", default=str(ROOT))
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency reminder path
        raise SystemExit(f"uvicorn is not installed: {exc}")
    app = create_app(args.root_dir)
    uvicorn.run(app, host=args.host, port=args.port)
