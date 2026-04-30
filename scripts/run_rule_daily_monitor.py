from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the separate rule-based daily-monitor baseline.")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "daily_monitor_slice.csv"))
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_liquidity_stress_15bps.json"))
    parser.add_argument("--prediction-date", default="")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "daily_monitor_rule"))
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_daily_monitor.py"),
        "--skip-update",
        "--data-path",
        args.data_path,
        "--adjust",
        args.adjust,
        "--research-config",
        str(ROOT / "configs" / "research_rule_production.json"),
        "--backtest-config",
        args.backtest_config,
        "--output-dir",
        args.output_dir,
        "--output-prefix",
        "daily_monitor_rule",
    ]
    if args.prediction_date:
        cmd.extend(["--prediction-date", args.prediction_date])
    raise SystemExit(subprocess.call(cmd))
