from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.research_report import extract_display_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize multiple workflow metrics/summary files into a comparison table."
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument(
        "--case",
        action="append",
        nargs=4,
        metavar=("LABEL", "METRICS_JSON", "SUMMARY_JSON", "FACTOR_JSON"),
        help="Case tuple: label metrics_json summary_json factor_json",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if not args.case:
        raise ValueError("At least one --case is required.")

    rows = []
    for label, metrics_path, summary_path, factor_path in args.case:
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        factor_weights = extract_display_weights(json.loads(Path(factor_path).read_text(encoding="utf-8")))
        top_factor = ""
        top_weight = 0.0
        if factor_weights:
            top_factor, top_weight = max(factor_weights.items(), key=lambda item: abs(item[1]))
        rows.append(
            {
                "label": label,
                "rows": summary["profile"]["rows"],
                "unique_codes": summary["profile"]["unique_codes"],
                "st_ratio": summary["profile"]["st_ratio"],
                "unknown_industry_ratio": summary["profile"]["unknown_industry_ratio"],
                "annual_return": metrics["annual_return"],
                "annual_volatility": metrics["annual_volatility"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "hit_rate": metrics["hit_rate"],
                "avg_turnover": metrics["avg_turnover"],
                "top_factor": top_factor,
                "top_factor_weight": top_weight,
            }
        )

    df = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    md_lines = [
        "# Research Case Comparison",
        "",
        "| label | codes | unknown_industry | annual_return | sharpe | max_drawdown | avg_turnover | top_factor |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in df.itertuples(index=False):
        md_lines.append(
            "| {label} | {codes} | {unknown:.2%} | {ret:.4f} | {sharpe:.4f} | {mdd:.4f} | {turnover:.4f} | {factor} |".format(
                label=row.label,
                codes=int(row.unique_codes),
                unknown=float(row.unknown_industry_ratio),
                ret=float(row.annual_return),
                sharpe=float(row.sharpe),
                mdd=float(row.max_drawdown),
                turnover=float(row.avg_turnover),
                factor=row.top_factor,
            )
        )
    Path(args.output_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[OK] comparison table saved to {args.output_csv} and {args.output_md}")
