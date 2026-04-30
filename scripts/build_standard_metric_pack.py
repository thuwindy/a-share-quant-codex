from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.metric_pack import (
    build_standard_metric_pack,
    metric_pack_dataframe,
    metric_pack_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build standardized metric pack (T30) from monitor metrics JSON.")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_auto"))
    parser.add_argument("--selection-date", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-prefix", default="")
    return parser


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _latest_metrics_json(monitor_dir: str | Path) -> Path:
    files = sorted(Path(monitor_dir).glob("*_metrics.json"))
    if not files:
        raise FileNotFoundError(f"No metrics JSON found under {monitor_dir}.")
    return files[-1]


def _infer_selection_date(raw: str, metrics_json_path: Path) -> str:
    if raw:
        return raw
    m = re.search(r"(\d{8})_metrics\.json$", metrics_json_path.name)
    if m:
        return f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
    return date.today().isoformat()


if __name__ == "__main__":
    args = build_parser().parse_args()
    metrics_json_path = Path(args.metrics_json) if args.metrics_json else _latest_metrics_json(args.monitor_dir)
    metrics = _load_json(metrics_json_path)
    selection_date = _infer_selection_date(args.selection_date, metrics_json_path)

    output_dir = Path(args.output_dir) if args.output_dir else metrics_json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or metrics_json_path.name.replace("_metrics.json", "_metric_pack")

    pack = build_standard_metric_pack(
        metrics=metrics,
        selection_date=selection_date,
        source_metrics_json=str(metrics_json_path),
        strategy_classification=metrics.get("strategy_assessment", {}).get("classification", ""),
    )

    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    md_path = output_dir / f"{prefix}.md"

    json_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_pack_dataframe(pack).to_csv(csv_path, index=False)
    md_path.write_text(metric_pack_markdown(pack), encoding="utf-8")

    print(
        json.dumps(
            {
                "selection_date": selection_date,
                "source_metrics_json": str(metrics_json_path),
                "metric_pack_json": str(json_path),
                "metric_pack_csv": str(csv_path),
                "metric_pack_md": str(md_path),
                "annual_return": pack["annual_return"],
                "sharpe": pack["sharpe"],
                "max_drawdown": pack["max_drawdown"],
                "turnover": pack["turnover"],
                "cost_drag": pack["cost_drag"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[OK] standard metric pack saved to {output_dir}")

