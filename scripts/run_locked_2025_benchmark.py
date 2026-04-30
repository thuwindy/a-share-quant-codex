from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run locked 2025 benchmark.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--research-config", default=str(ROOT / "configs" / "research_production_default.json"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_production_managed_15bps.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "locked_benchmark_2025"))
    parser.add_argument("--output-prefix", default="locked_2025_production")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(
        '{"stage":"locked_2025_benchmark:start","year":2025,"output_dir":"%s"}'
        % args.output_dir,
        flush=True,
    )
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_calendar_year_review.py"),
        "--year",
        str(args.year),
        "--data-path",
        str(args.data_path),
        "--research-config",
        str(args.research_config),
        "--backtest-config",
        str(args.backtest_config),
        "--output-dir",
        str(args.output_dir),
        "--output-prefix",
        str(args.output_prefix),
    ]
    rc = subprocess.call(cmd)
    print(
        '{"stage":"locked_2025_benchmark:done","return_code":%d}' % rc,
        flush=True,
    )
    raise SystemExit(rc)
