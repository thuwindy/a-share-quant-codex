from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the daily stable-observation-cycle candidate-pool dashboard."
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument(
        "--research-config",
        default=str(ROOT / "configs" / "research_production_default.json"),
    )
    parser.add_argument(
        "--output-prefix",
        default="candidate_pool_quality_dashboard",
        help="Output directory prefix under outputs/.",
    )
    parser.add_argument(
        "--date-key",
        default="",
        help="Optional YYYYMMDD override. Default uses local date.",
    )
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--rolling-window", type=int, default=60)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    date_key = args.date_key.strip() or datetime.now().strftime("%Y%m%d")
    output_dir = ROOT / "outputs" / f"{args.output_prefix}_{date_key}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "analysis" / "run_candidate_pool_quality_dashboard.py"),
        "--data-path",
        str(args.data_path),
        "--research-config",
        str(args.research_config),
        "--output-dir",
        str(output_dir),
        "--adjust",
        str(args.adjust),
        "--rolling-window",
        str(int(args.rolling_window)),
    ]
    subprocess.run(cmd, check=True)
    print(f"stable_observation_output_dir={output_dir}")


if __name__ == "__main__":
    main()
