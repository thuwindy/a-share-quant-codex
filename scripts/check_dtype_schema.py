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


REQUIRED_COLUMNS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "industry",
    "is_st",
    "is_suspended",
    "can_buy",
    "can_sell",
    "adj_factor",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "market_cap",
    "adj_factor",
]

DATE_COLUMNS = [
    "date",
    "list_date",
    "delist_date",
    "fundamental_ann_date",
    "fundamental_end_date",
]

BOOL_COLUMNS = [
    "is_st",
    "is_suspended",
    "can_buy",
    "can_sell",
    "distress_risk_flag",
    "profit_warning_flag",
]

BOOL_TRUE_TOKENS = {"1", "true", "t", "yes", "y", "on"}
BOOL_FALSE_TOKENS = {"0", "false", "f", "no", "n", "off"}
BOOL_NULL_TOKENS = {"", "nan", "none", "null", "<na>", "nat"}


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
    parser = argparse.ArgumentParser(description="Run T11: dtype/schema checks on local daily panel.")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "data_quality"))
    parser.add_argument("--output-prefix", default="t11_dtype_schema")
    parser.add_argument("--chunk-size", type=int, default=300_000)
    parser.add_argument("--max-unknown-bool-samples", type=int, default=100)
    return parser


def _to_markdown(rows: list[CheckRow]) -> str:
    lines = [
        "# T11 Dtype/Schema Check",
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


def _is_nonempty_value(series: pd.Series) -> pd.Series:
    as_text = series.astype("string").str.strip().str.lower()
    return series.notna() & ~as_text.isin(BOOL_NULL_TOKENS)


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    header_cols = pd.read_csv(data_path, nrows=0).columns.tolist()
    missing_required = [col for col in REQUIRED_COLUMNS if col not in header_cols]

    active_numeric_cols = [col for col in NUMERIC_COLUMNS if col in header_cols]
    active_date_cols = [col for col in DATE_COLUMNS if col in header_cols]
    active_bool_cols = [col for col in BOOL_COLUMNS if col in header_cols]

    bool_dtype_hints = {col: "string" for col in active_bool_cols}
    usecols = list(dict.fromkeys(["date", "code"] + active_numeric_cols + active_date_cols + active_bool_cols))
    usecols = [col for col in usecols if col in header_cols]

    rows_total = 0
    per_numeric_invalid: dict[str, int] = {col: 0 for col in active_numeric_cols}
    per_numeric_nonempty: dict[str, int] = {col: 0 for col in active_numeric_cols}
    per_date_invalid: dict[str, int] = {col: 0 for col in active_date_cols}
    per_date_nonempty: dict[str, int] = {col: 0 for col in active_date_cols}
    per_bool_unknown: dict[str, int] = {col: 0 for col in active_bool_cols}
    bool_unknown_samples: list[dict[str, str]] = []

    for chunk in pd.read_csv(
        data_path,
        usecols=usecols,
        dtype=bool_dtype_hints if bool_dtype_hints else None,
        low_memory=False,
        chunksize=args.chunk_size,
    ):
        rows_total += len(chunk)

        for col in active_numeric_cols:
            raw = chunk[col]
            nonempty = _is_nonempty_value(raw)
            parsed = pd.to_numeric(raw, errors="coerce")
            invalid = nonempty & parsed.isna()
            per_numeric_nonempty[col] += int(nonempty.sum())
            per_numeric_invalid[col] += int(invalid.sum())

        for col in active_date_cols:
            raw = chunk[col]
            nonempty = _is_nonempty_value(raw)
            parsed = pd.to_datetime(raw, errors="coerce")
            invalid = nonempty & parsed.isna()
            per_date_nonempty[col] += int(nonempty.sum())
            per_date_invalid[col] += int(invalid.sum())

        for col in active_bool_cols:
            raw = chunk[col].astype("string").str.strip().str.lower()
            normalized = raw.fillna("")
            known = normalized.isin(BOOL_TRUE_TOKENS | BOOL_FALSE_TOKENS | BOOL_NULL_TOKENS)
            unknown = ~known
            count = int(unknown.sum())
            per_bool_unknown[col] += count
            if count > 0 and len(bool_unknown_samples) < args.max_unknown_bool_samples:
                sample_rows = chunk.loc[unknown, ["date", "code", col]].head(
                    args.max_unknown_bool_samples - len(bool_unknown_samples)
                )
                for row in sample_rows.itertuples(index=False):
                    bool_unknown_samples.append(
                        {
                            "date": pd.Timestamp(getattr(row, "date")).strftime("%Y-%m-%d")
                            if pd.notna(getattr(row, "date"))
                            else "",
                            "code": str(getattr(row, "code")),
                            "column": col,
                            "raw_value": str(getattr(row, col)),
                        }
                    )

    checks: list[CheckRow] = [
        CheckRow("rows_total", "INFO", int(rows_total), "-", "Total scanned rows."),
        CheckRow(
            "required_columns_missing",
            "PASS" if len(missing_required) == 0 else "FAIL",
            int(len(missing_required)),
            "== 0",
            f"Missing: {','.join(missing_required) if missing_required else 'none'}",
        ),
    ]

    total_numeric_invalid = 0
    total_numeric_nonempty = 0
    for col in active_numeric_cols:
        invalid = per_numeric_invalid[col]
        nonempty = per_numeric_nonempty[col]
        total_numeric_invalid += invalid
        total_numeric_nonempty += nonempty
        checks.append(
            CheckRow(
                f"numeric_parse_invalid::{col}",
                "PASS" if invalid == 0 else "WARN",
                int(invalid),
                "== 0 (recommended)",
                f"nonempty={nonempty}",
            )
        )

    total_date_invalid = 0
    total_date_nonempty = 0
    for col in active_date_cols:
        invalid = per_date_invalid[col]
        nonempty = per_date_nonempty[col]
        total_date_invalid += invalid
        total_date_nonempty += nonempty
        checks.append(
            CheckRow(
                f"date_parse_invalid::{col}",
                "PASS" if invalid == 0 else "WARN",
                int(invalid),
                "== 0 (recommended)",
                f"nonempty={nonempty}",
            )
        )

    total_bool_unknown = 0
    for col in active_bool_cols:
        unknown = per_bool_unknown[col]
        total_bool_unknown += unknown
        checks.append(
            CheckRow(
                f"bool_unknown_tokens::{col}",
                "PASS" if unknown == 0 else "FAIL",
                int(unknown),
                "== 0",
                "Allowed tokens: 1/0/true/false/yes/no/on/off/null",
            )
        )

    checks.append(
        CheckRow(
            "numeric_parse_invalid_total",
            "PASS" if total_numeric_invalid == 0 else "WARN",
            int(total_numeric_invalid),
            "== 0 (recommended)",
            f"nonempty={total_numeric_nonempty}",
        )
    )
    checks.append(
        CheckRow(
            "date_parse_invalid_total",
            "PASS" if total_date_invalid == 0 else "WARN",
            int(total_date_invalid),
            "== 0 (recommended)",
            f"nonempty={total_date_nonempty}",
        )
    )
    checks.append(
        CheckRow(
            "bool_unknown_tokens_total",
            "PASS" if total_bool_unknown == 0 else "FAIL",
            int(total_bool_unknown),
            "== 0",
            f"samples={len(bool_unknown_samples)}",
        )
    )

    status_rank = {"FAIL": 3, "WARN": 2, "PASS": 1, "INFO": 0}
    max_rank = max(status_rank.get(row.status, 0) for row in checks)
    overall = next(label for label, rank in status_rank.items() if rank == max_rank)

    checks_df = pd.DataFrame([row.as_dict() for row in checks])
    checks_csv_path = out_dir / f"{prefix}_checks.csv"
    checks_md_path = out_dir / f"{prefix}_checks.md"
    unknown_bool_csv_path = out_dir / f"{prefix}_unknown_bool_samples.csv"
    summary_json_path = out_dir / f"{prefix}_summary.json"

    checks_df.to_csv(checks_csv_path, index=False)
    checks_md_path.write_text(_to_markdown(checks), encoding="utf-8")
    pd.DataFrame(bool_unknown_samples, columns=["date", "code", "column", "raw_value"]).to_csv(
        unknown_bool_csv_path, index=False
    )

    summary = {
        "overall_status": overall,
        "data_path": str(data_path),
        "rows_total": int(rows_total),
        "required_columns_missing": int(len(missing_required)),
        "numeric_parse_invalid_total": int(total_numeric_invalid),
        "date_parse_invalid_total": int(total_date_invalid),
        "bool_unknown_tokens_total": int(total_bool_unknown),
        "checks_csv_path": str(checks_csv_path),
        "checks_md_path": str(checks_md_path),
        "unknown_bool_samples_csv_path": str(unknown_bool_csv_path),
        "run_date": str(date.today()),
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] T11 dtype/schema outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
