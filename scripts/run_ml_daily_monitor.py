from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the separate ML strategy daily-monitor baseline.")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_production_managed_15bps.json"))
    parser.add_argument("--prediction-date", default="")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "daily_monitor_ml"))
    parser.add_argument("--model", choices=("ridge", "xgboost", "lightgbm", "logistic"), default="ridge")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    config_map = {
        "ridge": ROOT / "configs" / "research_ml_baseline.json",
        "xgboost": ROOT / "configs" / "research_ml_xgboost.json",
        "lightgbm": ROOT / "configs" / "research_ml_lightgbm.json",
        "logistic": ROOT / "configs" / "research_production_ml_classification.json",
    }
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_daily_monitor.py"),
        "--skip-update",
        "--data-path",
        args.data_path,
        "--adjust",
        args.adjust,
        "--research-config",
        str(config_map[str(args.model)]),
        "--backtest-config",
        args.backtest_config,
        "--output-dir",
        args.output_dir,
        "--output-prefix",
        f"daily_monitor_ml_{args.model}",
    ]
    if args.prediction_date:
        cmd.extend(["--prediction-date", args.prediction_date])
    raise SystemExit(subprocess.call(cmd))
