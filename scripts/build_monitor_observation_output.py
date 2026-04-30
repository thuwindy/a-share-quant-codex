from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.latest_picks import build_latest_picks_markdown, build_latest_picks_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build lightweight observation-only monitor outputs from an existing slice."
    )
    parser.add_argument("--data-path", default="")
    parser.add_argument("--input-path", default="")
    parser.add_argument("--slice-path", default="")
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-codes", type=int, default=500)
    parser.add_argument("--suffixes", default="SH,SZ")
    parser.add_argument("--research-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--prediction-date", default="")
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    return parser


def _profile_from_csv(path: Path) -> dict[str, object]:
    rows = 0
    start_date = ""
    end_date = ""
    unique_codes: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        date_idx = header.index("date") if "date" in header else -1
        code_idx = header.index("code") if "code" in header else -1
        for line in fh:
            text = line.strip()
            if not text:
                continue
            parts = text.split(",")
            rows += 1
            if date_idx >= 0 and date_idx < len(parts):
                value = parts[date_idx]
                if not start_date:
                    start_date = value
                end_date = value
            if code_idx >= 0 and code_idx < len(parts):
                unique_codes.add(parts[code_idx])
    return {
        "rows": rows,
        "unique_codes": len(unique_codes),
        "start_date": start_date,
        "end_date": end_date,
    }


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data_path) if args.data_path else None
    if data_path is None:
        if not args.input_path or not args.slice_path:
            raise ValueError("Either --data-path or both --input-path and --slice-path are required.")
        from ashare_quant.data.research_slice import build_research_slice

        suffixes = tuple(part.strip().upper() for part in str(args.suffixes).split(",") if part.strip())
        data_path = Path(args.slice_path)
        build_research_slice(
            input_path=args.input_path,
            output_path=data_path,
            start_date=args.start_date,
            end_date=args.end_date or args.prediction_date,
            include_suffixes=suffixes,
            max_codes=int(args.max_codes),
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    picks, factor_weights, summary = build_latest_picks_table(
        data_path=data_path,
        research_config_path=args.research_config,
        data_adjust=args.adjust,
        prediction_date=args.prediction_date or None,
    )
    date_key = summary.selection_date.replace("-", "")
    prefix = f"{args.output_prefix}_{date_key}"

    (output_dir / f"{prefix}_picks.csv").write_text(picks.to_csv(index=False), encoding="utf-8")
    (output_dir / f"{prefix}_report.md").write_text(
        build_latest_picks_markdown(picks=picks, factor_weights=factor_weights, summary=summary),
        encoding="utf-8",
    )
    payload = {
        "profile": _profile_from_csv(data_path),
        "summary": summary.__dict__,
        "factor_weights": factor_weights,
        "strategy_assessment": {},
        "observation_pool": json.loads(picks.to_json(orient="records", date_format="iso", force_ascii=False)),
        "strategy_snapshot": [],
    }
    (output_dir / f"{prefix}_picks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"selection_date": summary.selection_date, "output_prefix": prefix}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
