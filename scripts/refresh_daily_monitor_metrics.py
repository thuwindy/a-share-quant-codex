from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.metric_pack import (
    build_standard_metric_pack,
    metric_pack_dataframe,
    metric_pack_markdown,
)
from ashare_quant.pipeline import run_research_pipeline


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern} under {directory}.")
    return files[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh metrics artifacts for the latest daily monitor picks.")
    parser.add_argument("--monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_auto"))
    parser.add_argument("--monitor-json", default="")
    parser.add_argument("--slice-path", default="")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--research-config", default=str(ROOT / "configs" / "research_production_default.json"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_production_managed_15bps.json"))
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    monitor_json_path = Path(args.monitor_json) if args.monitor_json else _latest_file(Path(args.monitor_dir), "*_picks.json")
    payload = _load_json(monitor_json_path)
    summary = payload.get("summary", {})
    strategy_assessment = payload.get("strategy_assessment", {}) or {}
    selection_date = str(summary.get("selection_date", "") or "")

    prefix = monitor_json_path.name.replace("_picks.json", "")
    output_dir = monitor_json_path.parent
    metrics_path = output_dir / f"{prefix}_metrics.json"
    metric_pack_json_path = output_dir / f"{prefix}_metric_pack.json"
    metric_pack_csv_path = output_dir / f"{prefix}_metric_pack.csv"
    metric_pack_md_path = output_dir / f"{prefix}_metric_pack.md"

    candidate_paths = []
    if args.slice_path:
        candidate_paths.append(Path(args.slice_path))
    summary_data_path = str(summary.get("data_path", "") or "").strip()
    if summary_data_path:
        candidate_paths.append(Path(summary_data_path))
    candidate_paths.append(ROOT / "data" / "daily_monitor_main_slice.csv")
    candidate_paths.append(Path(args.data_path))
    data_path = next((path for path in candidate_paths if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError(f"No usable data path found for metrics refresh. candidates={candidate_paths}")

    if metrics_path.exists() and not args.force:
        print(
            json.dumps(
                {
                    "selection_date": selection_date,
                    "monitor_json": str(monitor_json_path),
                    "data_path": str(data_path),
                    "metrics_path": str(metrics_path),
                    "skipped": True,
                    "reason": "metrics already exist",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    result, metrics, _score_details, _targets = run_research_pipeline(
        data_path=data_path,
        research_config_path=args.research_config,
        backtest_config_path=args.backtest_config,
        data_adjust=args.adjust,
    )
    _ = result  # kept for interface parity

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_pack = build_standard_metric_pack(
        metrics=metrics,
        selection_date=selection_date,
        source_metrics_json=str(metrics_path),
        strategy_classification=str(strategy_assessment.get("classification") or metrics.get("strategy_assessment", {}).get("classification", "")),
    )
    metric_pack_json_path.write_text(json.dumps(metric_pack, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_pack_dataframe(metric_pack).to_csv(metric_pack_csv_path, index=False)
    metric_pack_md_path.write_text(metric_pack_markdown(metric_pack), encoding="utf-8")

    print(
        json.dumps(
            {
                "selection_date": selection_date,
                "monitor_json": str(monitor_json_path),
                "data_path": str(data_path),
                "metrics_path": str(metrics_path),
                "metric_pack_json": str(metric_pack_json_path),
                "metric_pack_csv": str(metric_pack_csv_path),
                "metric_pack_md": str(metric_pack_md_path),
                "annual_return": metrics.get("annual_return"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
