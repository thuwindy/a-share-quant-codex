from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


NULL_TOKENS = {"", "nan", "none", "null", "<na>", "nat"}


@dataclass
class CheckRow:
    check: str
    status: str
    value: float | int | str
    threshold: str
    note: str

    def as_dict(self) -> dict:
        return {
            "check": self.check,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
            "note": self.note,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run T13: point-in-time event alignment checks (event_date <= date).")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "data_quality"))
    parser.add_argument("--output-prefix", default="t13_point_in_time")
    parser.add_argument("--chunk-size", type=int, default=300_000)
    parser.add_argument("--max-samples", type=int, default=300)
    return parser


def _is_nonempty_value(series: pd.Series) -> pd.Series:
    as_text = series.astype("string").str.strip().str.lower()
    return series.notna() & ~as_text.isin(NULL_TOKENS)


def _markdown(rows: list[CheckRow]) -> str:
    lines = [
        "# T13 Point-in-Time Alignment",
        "",
        "| check | status | value | threshold | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {check} | {status} | {value} | {threshold} | {note} |".format(
                check=row.check,
                status=row.status,
                value=row.value,
                threshold=row.threshold,
                note=row.note.replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    args = build_parser().parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header_cols = pd.read_csv(data_path, nrows=0).columns.tolist()
    if "date" not in header_cols or "code" not in header_cols:
        raise ValueError("Input dataset must include `date` and `code` columns.")

    event_columns = [col for col in ("fundamental_ann_date", "analyst_report_date") if col in header_cols]
    usecols = ["date", "code", *event_columns]

    rows_total = 0
    missing_key_rows = 0
    violations: list[dict[str, str]] = []
    counters: dict[str, dict[str, int]] = {
        col: {
            "nonempty": 0,
            "parse_invalid": 0,
            "future_violation": 0,
        }
        for col in event_columns
    }

    for chunk in pd.read_csv(data_path, usecols=usecols, chunksize=args.chunk_size, low_memory=False):
        rows_total += len(chunk)
        signal_date = pd.to_datetime(chunk["date"], errors="coerce")
        code = chunk["code"].astype(str)
        missing_key_rows += int(signal_date.isna().sum() + code.isna().sum())

        for col in event_columns:
            raw = chunk[col]
            nonempty = _is_nonempty_value(raw)
            parsed = pd.to_datetime(raw, errors="coerce")
            parse_invalid_mask = nonempty & parsed.isna()
            future_mask = nonempty & parsed.notna() & signal_date.notna() & (parsed > signal_date)

            counters[col]["nonempty"] += int(nonempty.sum())
            counters[col]["parse_invalid"] += int(parse_invalid_mask.sum())
            counters[col]["future_violation"] += int(future_mask.sum())

            if len(violations) < args.max_samples:
                sample_space = chunk.loc[parse_invalid_mask | future_mask, ["date", "code", col]].head(
                    args.max_samples - len(violations)
                )
                for row in sample_space.itertuples(index=False):
                    raw_event = getattr(row, col)
                    event_ts = pd.to_datetime(raw_event, errors="coerce")
                    signal_ts = pd.to_datetime(getattr(row, "date"), errors="coerce")
                    violation_type = (
                        "future_event_date"
                        if pd.notna(event_ts) and pd.notna(signal_ts) and event_ts > signal_ts
                        else "invalid_event_date_parse"
                    )
                    violations.append(
                        {
                            "date": signal_ts.strftime("%Y-%m-%d") if pd.notna(signal_ts) else "",
                            "code": str(getattr(row, "code")),
                            "event_column": col,
                            "event_date_raw": str(raw_event),
                            "event_date_parsed": event_ts.strftime("%Y-%m-%d") if pd.notna(event_ts) else "",
                            "violation_type": violation_type,
                        }
                    )

    checks: list[CheckRow] = [
        CheckRow("rows_total", "INFO", int(rows_total), "-", "Total scanned rows."),
        CheckRow(
            "missing_key_rows",
            "PASS" if missing_key_rows == 0 else "WARN",
            int(missing_key_rows),
            "== 0",
            "Rows with missing/invalid date or code.",
        ),
        CheckRow(
            "event_columns_detected",
            "INFO",
            ",".join(event_columns) if event_columns else "none",
            "-",
            "Event date columns found in this dataset.",
        ),
    ]

    if not event_columns:
        checks.append(
            CheckRow(
                "point_in_time_check_skipped",
                "WARN",
                "no event columns",
                "expect >=1 event column",
                "Dataset lacks fundamental_ann_date/analyst_report_date.",
            )
        )
    else:
        for col in event_columns:
            c = counters[col]
            checks.append(
                CheckRow(
                    f"event_nonempty::{col}",
                    "INFO" if c["nonempty"] > 0 else "WARN",
                    int(c["nonempty"]),
                    "> 0 (recommended)",
                    "Rows with non-empty event date values. Zero means PIT check has no coverage on this dataset.",
                )
            )
            checks.append(
                CheckRow(
                    f"event_parse_invalid::{col}",
                    "PASS" if c["parse_invalid"] == 0 else "WARN",
                    int(c["parse_invalid"]),
                    "== 0 (recommended)",
                    "Non-empty event dates that failed datetime parse.",
                )
            )
            checks.append(
                CheckRow(
                    f"event_future_violation::{col}",
                    "PASS" if c["future_violation"] == 0 else "FAIL",
                    int(c["future_violation"]),
                    "== 0",
                    "Rows where event_date > signal date.",
                )
            )

    status_rank = {"FAIL": 3, "WARN": 2, "PASS": 1, "INFO": 0}
    max_rank = max(status_rank.get(row.status, 0) for row in checks)
    overall = next(label for label, rank in status_rank.items() if rank == max_rank)

    checks_csv_path = out_dir / f"{args.output_prefix}_checks.csv"
    checks_md_path = out_dir / f"{args.output_prefix}_checks.md"
    samples_csv_path = out_dir / f"{args.output_prefix}_violation_samples.csv"
    summary_json_path = out_dir / f"{args.output_prefix}_summary.json"

    pd.DataFrame([row.as_dict() for row in checks]).to_csv(checks_csv_path, index=False)
    checks_md_path.write_text(_markdown(checks), encoding="utf-8")
    pd.DataFrame(
        violations,
        columns=["date", "code", "event_column", "event_date_raw", "event_date_parsed", "violation_type"],
    ).to_csv(samples_csv_path, index=False)

    summary = {
        "overall_status": overall,
        "data_path": str(data_path),
        "rows_total": int(rows_total),
        "event_columns": event_columns,
        "missing_key_rows": int(missing_key_rows),
        "counters": counters,
        "violation_samples_count": int(len(violations)),
        "checks_csv_path": str(checks_csv_path),
        "checks_md_path": str(checks_md_path),
        "violation_samples_csv_path": str(samples_csv_path),
        "run_date": str(date.today()),
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] T13 point-in-time alignment outputs saved to {out_dir}")
