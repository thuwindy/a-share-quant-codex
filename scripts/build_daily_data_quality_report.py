from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


TRACK_COLUMNS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "market_cap",
    "industry",
    "is_st",
    "is_suspended",
    "can_buy",
    "can_sell",
]
NULL_TOKENS = {"", "nan", "none", "null", "<na>", "nat"}


@dataclass
class MetricRow:
    metric: str
    status: str
    value: float | int | str
    threshold: str
    note: str

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
            "note": self.note,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a daily data-quality report for missing ratio, duplicate ratio, and unknown industry ratio."
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "data_quality"))
    parser.add_argument("--output-prefix", default="daily_quality")
    parser.add_argument("--chunk-size", type=int, default=300_000)
    parser.add_argument("--sample-duplicate-keys", type=int, default=20)
    parser.add_argument("--unknown-industry-tokens", default="unknown,unk,其他,other")
    parser.add_argument("--warn-unknown-industry-ratio", type=float, default=0.01)
    parser.add_argument("--fail-unknown-industry-ratio", type=float, default=0.03)
    parser.add_argument("--legacy-industry-prefix", default="legacy_unclassified_")
    parser.add_argument("--warn-legacy-industry-ratio", type=float, default=0.08)
    parser.add_argument("--fail-legacy-industry-ratio", type=float, default=0.12)
    parser.add_argument("--warn-missing-ratio", type=float, default=0.02)
    parser.add_argument("--fail-missing-ratio", type=float, default=0.10)
    return parser


def _to_text_lower(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _is_missing(series: pd.Series) -> pd.Series:
    text = _to_text_lower(series)
    return series.isna() | text.isin(NULL_TOKENS)


def _status_by_ratio(ratio: float, warn: float, fail: float) -> str:
    if ratio > fail:
        return "FAIL"
    if ratio > warn:
        return "WARN"
    return "PASS"


def _sqlite_prepare(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS key_counts (
            date TEXT NOT NULL,
            code TEXT NOT NULL,
            cnt INTEGER NOT NULL,
            PRIMARY KEY (date, code)
        );
        """
    )
    return conn


def _sqlite_upsert_key_counts(conn: sqlite3.Connection, key_count_df: pd.DataFrame) -> None:
    if key_count_df.empty:
        return
    payload = list(
        zip(
            key_count_df["date"].astype(str).tolist(),
            key_count_df["code"].astype(str).tolist(),
            key_count_df["cnt"].astype(int).tolist(),
        )
    )
    conn.executemany(
        """
        INSERT INTO key_counts(date, code, cnt)
        VALUES (?, ?, ?)
        ON CONFLICT(date, code) DO UPDATE SET cnt = key_counts.cnt + excluded.cnt;
        """,
        payload,
    )
    conn.commit()


def _markdown(summary_rows: list[MetricRow]) -> str:
    lines = [
        "# Daily Data Quality Report",
        "",
        "| metric | status | value | threshold | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {metric} | {status} | {value} | {threshold} | {note} |".format(
                metric=row.metric,
                status=row.status,
                value=row.value,
                threshold=row.threshold,
                note=row.note.replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unknown_tokens = {item.strip().lower() for item in args.unknown_industry_tokens.split(",") if item.strip()}
    header = pd.read_csv(data_path, nrows=0).columns.tolist()
    usecols = [col for col in TRACK_COLUMNS if col in header]
    if "date" not in usecols or "code" not in usecols:
        raise ValueError("Input dataset must include date and code columns.")

    rows_total = 0
    missing_key_rows = 0
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    unique_codes_seen: set[str] = set()
    missing_counts: dict[str, int] = {col: 0 for col in usecols}
    per_date_total: dict[str, int] = {}
    per_date_unknown_industry: dict[str, int] = {}
    per_date_legacy_industry: dict[str, int] = {}
    unknown_industry_rows = 0
    legacy_industry_rows = 0

    with tempfile.TemporaryDirectory(prefix="aq_t12_") as tempdir:
        db_path = Path(tempdir) / "key_counts.sqlite"
        conn = _sqlite_prepare(db_path)
        try:
            for chunk in pd.read_csv(data_path, usecols=usecols, chunksize=args.chunk_size, low_memory=False):
                rows_total += len(chunk)
                chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                chunk["code"] = chunk["code"].astype("string").str.strip()

                for col in usecols:
                    missing_counts[col] += int(_is_missing(chunk[col]).sum())

                missing_key_mask = chunk["date"].isna() | chunk["code"].isna() | (chunk["code"] == "")
                missing_key_rows += int(missing_key_mask.sum())
                valid = chunk.loc[~missing_key_mask].copy()
                if valid.empty:
                    continue

                chunk_min = valid["date"].min()
                chunk_max = valid["date"].max()
                min_date = chunk_min if min_date is None or chunk_min < min_date else min_date
                max_date = chunk_max if max_date is None or chunk_max > max_date else max_date
                unique_codes_seen.update(valid["code"].astype(str).unique().tolist())

                if "industry" in valid.columns:
                    industry_text = _to_text_lower(valid["industry"])
                    unknown_mask = industry_text.isin(unknown_tokens) | _is_missing(valid["industry"])
                    legacy_mask = industry_text.str.startswith(args.legacy_industry_prefix.lower())
                    unknown_industry_rows += int(unknown_mask.sum())
                    legacy_industry_rows += int(legacy_mask.sum())
                else:
                    unknown_mask = pd.Series(False, index=valid.index)
                    legacy_mask = pd.Series(False, index=valid.index)

                date_text = valid["date"].dt.strftime("%Y-%m-%d")
                date_counts = date_text.value_counts(sort=False)
                date_unknown_counts = date_text.loc[unknown_mask].value_counts(sort=False)
                date_legacy_counts = date_text.loc[legacy_mask].value_counts(sort=False)
                for d, c in date_counts.items():
                    per_date_total[str(d)] = per_date_total.get(str(d), 0) + int(c)
                for d, c in date_unknown_counts.items():
                    per_date_unknown_industry[str(d)] = per_date_unknown_industry.get(str(d), 0) + int(c)
                for d, c in date_legacy_counts.items():
                    per_date_legacy_industry[str(d)] = per_date_legacy_industry.get(str(d), 0) + int(c)

                keys = pd.DataFrame({"date": date_text, "code": valid["code"].astype(str)})
                grouped = keys.value_counts(sort=False).reset_index(name="cnt")
                _sqlite_upsert_key_counts(conn, grouped)

            cur = conn.execute("SELECT COUNT(*) FROM key_counts")
            unique_key_count = int(cur.fetchone()[0])
            cur = conn.execute("SELECT COUNT(*) FROM key_counts WHERE cnt > 1")
            duplicate_key_groups = int(cur.fetchone()[0])
            duplicate_rows = int(rows_total - unique_key_count)
            cur = conn.execute(
                "SELECT date, code, cnt FROM key_counts WHERE cnt > 1 ORDER BY cnt DESC, date DESC, code ASC LIMIT ?",
                (int(args.sample_duplicate_keys),),
            )
            duplicate_samples = [{"date": r[0], "code": r[1], "count": int(r[2])} for r in cur.fetchall()]
        finally:
            conn.close()

    duplicate_ratio = float(duplicate_rows / rows_total) if rows_total else 0.0
    unknown_industry_ratio = float(unknown_industry_rows / rows_total) if rows_total else 0.0
    legacy_industry_ratio = float(legacy_industry_rows / rows_total) if rows_total else 0.0
    latest_date = max_date.strftime("%Y-%m-%d") if max_date is not None else ""
    latest_total = int(per_date_total.get(latest_date, 0))
    latest_unknown = int(per_date_unknown_industry.get(latest_date, 0))
    latest_legacy = int(per_date_legacy_industry.get(latest_date, 0))
    latest_unknown_ratio = float(latest_unknown / latest_total) if latest_total else 0.0
    latest_legacy_ratio = float(latest_legacy / latest_total) if latest_total else 0.0

    missing_rows = []
    for col in usecols:
        miss = int(missing_counts[col])
        ratio = float(miss / rows_total) if rows_total else 0.0
        missing_rows.append(
            {
                "column": col,
                "missing_count": miss,
                "missing_ratio": ratio,
                "status": _status_by_ratio(ratio, args.warn_missing_ratio, args.fail_missing_ratio),
            }
        )
    missing_df = pd.DataFrame(missing_rows).sort_values(["missing_ratio", "column"], ascending=[False, True]).reset_index(drop=True)

    summary_rows = [
        MetricRow("rows_total", "INFO", int(rows_total), "-", "Total scanned rows"),
        MetricRow("unique_codes", "INFO", int(len(unique_codes_seen)), "-", "Distinct code count"),
        MetricRow(
            "date_range",
            "INFO",
            f"{min_date.strftime('%Y-%m-%d') if min_date is not None else 'n/a'} -> {latest_date or 'n/a'}",
            "-",
            "Observed date range",
        ),
        MetricRow("missing_key_rows", "PASS" if missing_key_rows == 0 else "FAIL", int(missing_key_rows), "== 0", "Rows with missing date/code"),
        MetricRow(
            "duplicate_rows_ratio",
            "PASS" if duplicate_ratio == 0 else "FAIL",
            f"{duplicate_ratio:.6%}",
            "== 0",
            f"duplicate_rows={duplicate_rows}",
        ),
        MetricRow(
            "duplicate_key_groups",
            "PASS" if duplicate_key_groups == 0 else "FAIL",
            int(duplicate_key_groups),
            "== 0",
            "Groups with (date,code) count > 1",
        ),
        MetricRow(
            "unknown_industry_ratio_all",
            _status_by_ratio(unknown_industry_ratio, args.warn_unknown_industry_ratio, args.fail_unknown_industry_ratio),
            f"{unknown_industry_ratio:.6%}",
            f"warn>{args.warn_unknown_industry_ratio:.2%}, fail>{args.fail_unknown_industry_ratio:.2%}",
            f"unknown_rows={unknown_industry_rows}",
        ),
        MetricRow(
            "unknown_industry_ratio_latest_date",
            _status_by_ratio(latest_unknown_ratio, args.warn_unknown_industry_ratio, args.fail_unknown_industry_ratio),
            f"{latest_unknown_ratio:.6%}",
            f"warn>{args.warn_unknown_industry_ratio:.2%}, fail>{args.fail_unknown_industry_ratio:.2%}",
            f"date={latest_date}, unknown={latest_unknown}, total={latest_total}",
        ),
        MetricRow(
            "legacy_unclassified_ratio_all",
            _status_by_ratio(legacy_industry_ratio, args.warn_legacy_industry_ratio, args.fail_legacy_industry_ratio),
            f"{legacy_industry_ratio:.6%}",
            f"warn>{args.warn_legacy_industry_ratio:.2%}, fail>{args.fail_legacy_industry_ratio:.2%}",
            f"prefix={args.legacy_industry_prefix}, rows={legacy_industry_rows}",
        ),
        MetricRow(
            "legacy_unclassified_ratio_latest_date",
            _status_by_ratio(latest_legacy_ratio, args.warn_legacy_industry_ratio, args.fail_legacy_industry_ratio),
            f"{latest_legacy_ratio:.6%}",
            f"warn>{args.warn_legacy_industry_ratio:.2%}, fail>{args.fail_legacy_industry_ratio:.2%}",
            f"date={latest_date}, legacy={latest_legacy}, total={latest_total}",
        ),
    ]

    status_rank = {"FAIL": 3, "WARN": 2, "PASS": 1, "INFO": 0}
    overall_rank = max(status_rank.get(r.status, 0) for r in summary_rows)
    overall_status = next(label for label, rank in status_rank.items() if rank == overall_rank)

    run_tag = date.today().strftime("%Y%m%d")
    file_prefix = f"{args.output_prefix}_{run_tag}"
    summary_csv_path = out_dir / f"{file_prefix}_summary.csv"
    summary_md_path = out_dir / f"{file_prefix}_summary.md"
    summary_json_path = out_dir / f"{file_prefix}_summary.json"
    missing_csv_path = out_dir / f"{file_prefix}_missing_by_column.csv"
    duplicate_csv_path = out_dir / f"{file_prefix}_duplicate_key_samples.csv"

    pd.DataFrame([row.as_dict() for row in summary_rows]).to_csv(summary_csv_path, index=False)
    summary_md_path.write_text(_markdown(summary_rows), encoding="utf-8")
    missing_df.to_csv(missing_csv_path, index=False)
    pd.DataFrame(duplicate_samples, columns=["date", "code", "count"]).to_csv(duplicate_csv_path, index=False)

    payload = {
        "overall_status": overall_status,
        "data_path": str(data_path),
        "rows_total": int(rows_total),
        "unique_codes": int(len(unique_codes_seen)),
        "date_min": min_date.strftime("%Y-%m-%d") if min_date is not None else "",
        "date_max": latest_date,
        "duplicate_rows": int(duplicate_rows),
        "duplicate_key_groups": int(duplicate_key_groups),
        "duplicate_ratio": duplicate_ratio,
        "unknown_industry_rows": int(unknown_industry_rows),
        "unknown_industry_ratio": unknown_industry_ratio,
        "unknown_industry_ratio_latest_date": latest_unknown_ratio,
        "legacy_industry_rows": int(legacy_industry_rows),
        "legacy_industry_ratio": legacy_industry_ratio,
        "legacy_industry_ratio_latest_date": latest_legacy_ratio,
        "legacy_industry_prefix": args.legacy_industry_prefix,
        "summary_csv_path": str(summary_csv_path),
        "summary_md_path": str(summary_md_path),
        "missing_by_column_csv_path": str(missing_csv_path),
        "duplicate_key_samples_csv_path": str(duplicate_csv_path),
    }
    summary_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[OK] daily data-quality report saved to {out_dir}")


if __name__ == "__main__":
    main()
