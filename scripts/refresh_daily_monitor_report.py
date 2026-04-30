from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.latest_picks import LatestPickSummary
from scripts.run_daily_monitor import build_daily_monitor_markdown


def _latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern} under {directory}.")
    return files[-1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_monitor_json(monitor_dir: Path) -> Path:
    return _latest_file(monitor_dir, "*_picks.json")


def _risk_json_for_selection_date(risk_dir: Path, selection_date: str) -> Path:
    if selection_date:
        dated = risk_dir / f"risk_gate_{selection_date.replace('-', '')}.json"
        if dated.exists():
            return dated
    return _latest_file(risk_dir, "risk_gate_*.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh latest daily monitor markdown with risk governor summary.")
    parser.add_argument("--monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_auto"))
    parser.add_argument("--risk-dir", default=str(ROOT / "outputs" / "risk_governor"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    monitor_dir = Path(args.monitor_dir)
    risk_dir = Path(args.risk_dir)

    picks_json_path = _latest_monitor_json(monitor_dir)
    payload = _load_json(picks_json_path)
    summary_payload = payload.get("summary", {})
    summary = LatestPickSummary(**summary_payload)
    observation_table = pd.DataFrame(payload.get("observation_pool", []))
    strategy_table = pd.DataFrame(payload.get("strategy_snapshot", []))
    factor_weights = payload.get("factor_weights", {})
    strategy_assessment = payload.get("strategy_assessment", {})
    selection_date = str(summary_payload.get("selection_date", "") or "")

    metrics_path = picks_json_path.with_name(picks_json_path.name.replace("_picks.json", "_metrics.json"))
    report_path = picks_json_path.with_name(picks_json_path.name.replace("_picks.json", "_report.md"))
    metrics = _load_json(metrics_path) if metrics_path.exists() else None
    risk_json_path = _risk_json_for_selection_date(risk_dir, selection_date)
    risk_summary = _load_json(risk_json_path)

    report_path.write_text(
        build_daily_monitor_markdown(
            summary=summary,
            picks_json_path=picks_json_path,
            observation_table=observation_table,
            strategy_table=strategy_table,
            factor_weights=factor_weights,
            metrics=metrics,
            update_info=None,
            strategy_assessment=strategy_assessment,
            risk_summary=risk_summary,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "selection_date": selection_date,
                "picks_json": str(picks_json_path),
                "risk_json": str(risk_json_path),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
