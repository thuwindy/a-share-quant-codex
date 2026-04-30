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


CORE_NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
BOOL_MIXED_COLUMNS = ["is_st", "is_suspended", "can_buy", "can_sell", "distress_risk_flag", "profit_warning_flag"]
KEY_COLUMNS = ["date", "code"]


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
    parser = argparse.ArgumentParser(description="Run T10: daily sync consistency and dedupe checks on (date, code).")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "data_quality"))
    parser.add_argument("--output-prefix", default="daily_sync_consistency")
    parser.add_argument("--chunk-size", type=int, default=300_000)
    parser.add_argument("--sample-duplicate-keys", type=int, default=20)
    parser.add_argument("--sample-anomaly-rows", type=int, default=50)
    return parser


def _header_columns(path: str | Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


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
    records = list(
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
        records,
    )
    conn.commit()


def _rows_to_markdown_table(rows: list[CheckRow]) -> str:
    lines = [
        "# T10 Daily Sync Consistency",
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


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    header_cols = _header_columns(data_path)
    required_cols = [col for col in KEY_COLUMNS + CORE_NUMERIC_COLUMNS if col in header_cols]
    missing_required = [col for col in KEY_COLUMNS + CORE_NUMERIC_COLUMNS if col not in required_cols]
    if missing_required:
        raise ValueError(f"Missing required columns in {data_path}: {missing_required}")

    bool_hints = {col: "string" for col in BOOL_MIXED_COLUMNS if col in header_cols}
    usecols = required_cols + [col for col in BOOL_MIXED_COLUMNS if col in header_cols]
    usecols = list(dict.fromkeys(usecols))

    total_rows = 0
    missing_key_rows = 0
    missing_core_rows = 0
    invalid_ohlc_rows = 0
    negative_turnover_rows = 0
    future_date_rows = 0
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    unique_codes_seen: set[str] = set()
    missing_core_samples: list[dict] = []
    invalid_ohlc_samples: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="aq_t10_") as tmpdir:
        sqlite_path = Path(tmpdir) / "t10_key_counts.sqlite"
        conn = _sqlite_prepare(sqlite_path)
        try:
            for chunk in pd.read_csv(
                data_path,
                usecols=usecols,
                dtype=bool_hints if bool_hints else None,
                chunksize=args.chunk_size,
                low_memory=False,
            ):
                total_rows += len(chunk)
                chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                chunk["code"] = chunk["code"].astype("string").str.strip()

                missing_key_mask = chunk["date"].isna() | chunk["code"].isna() | (chunk["code"] == "")
                missing_key_rows += int(missing_key_mask.sum())

                valid_chunk = chunk.loc[~missing_key_mask].copy()
                if valid_chunk.empty:
                    continue

                today_ts = pd.Timestamp(date.today())
                future_date_rows += int((valid_chunk["date"] > today_ts).sum())

                chunk_min = valid_chunk["date"].min()
                chunk_max = valid_chunk["date"].max()
                min_date = chunk_min if min_date is None or chunk_min < min_date else min_date
                max_date = chunk_max if max_date is None or chunk_max > max_date else max_date
                unique_codes_seen.update(valid_chunk["code"].astype(str).unique().tolist())

                core_numeric = valid_chunk[CORE_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
                missing_core_mask = core_numeric.isna().any(axis=1)
                missing_core_rows += int(missing_core_mask.sum())
                if len(missing_core_samples) < args.sample_anomaly_rows and int(missing_core_mask.sum()) > 0:
                    sample = valid_chunk.loc[missing_core_mask, ["date", "code"] + CORE_NUMERIC_COLUMNS].head(
                        args.sample_anomaly_rows - len(missing_core_samples)
                    )
                    for row in sample.itertuples(index=False):
                        missing_core_samples.append(
                            {
                                "date": pd.Timestamp(getattr(row, "date")).strftime("%Y-%m-%d"),
                                "code": str(getattr(row, "code")),
                                "open": getattr(row, "open"),
                                "high": getattr(row, "high"),
                                "low": getattr(row, "low"),
                                "close": getattr(row, "close"),
                                "volume": getattr(row, "volume"),
                                "amount": getattr(row, "amount"),
                            }
                        )

                valid_price_mask = core_numeric[["open", "high", "low", "close"]].notna().all(axis=1)
                if int(valid_price_mask.sum()) > 0:
                    o = core_numeric.loc[valid_price_mask, "open"]
                    h = core_numeric.loc[valid_price_mask, "high"]
                    l = core_numeric.loc[valid_price_mask, "low"]
                    c = core_numeric.loc[valid_price_mask, "close"]
                    invalid_ohlc_mask = (l > h) | (o < l) | (o > h) | (c < l) | (c > h)
                    invalid_ohlc_rows += int(invalid_ohlc_mask.sum())
                    if len(invalid_ohlc_samples) < args.sample_anomaly_rows and int(invalid_ohlc_mask.sum()) > 0:
                        invalid_slice = valid_chunk.loc[valid_price_mask].loc[
                            invalid_ohlc_mask, ["date", "code"] + CORE_NUMERIC_COLUMNS
                        ]
                        sample = invalid_slice.head(args.sample_anomaly_rows - len(invalid_ohlc_samples))
                        for row in sample.itertuples(index=False):
                            invalid_ohlc_samples.append(
                                {
                                    "date": pd.Timestamp(getattr(row, "date")).strftime("%Y-%m-%d"),
                                    "code": str(getattr(row, "code")),
                                    "open": getattr(row, "open"),
                                    "high": getattr(row, "high"),
                                    "low": getattr(row, "low"),
                                    "close": getattr(row, "close"),
                                    "volume": getattr(row, "volume"),
                                    "amount": getattr(row, "amount"),
                                }
                            )

                negative_turnover_mask = (
                    core_numeric["volume"].fillna(0.0) < 0
                ) | (
                    core_numeric["amount"].fillna(0.0) < 0
                )
                negative_turnover_rows += int(negative_turnover_mask.sum())

                keys = valid_chunk.assign(date=valid_chunk["date"].dt.strftime("%Y-%m-%d"))[["date", "code"]]
                grouped = keys.value_counts(sort=False).reset_index(name="cnt")
                _sqlite_upsert_key_counts(conn, grouped)

            q = conn.execute("SELECT COUNT(*) FROM key_counts")
            unique_keys = int(q.fetchone()[0])
            q = conn.execute("SELECT COUNT(*) FROM key_counts WHERE cnt > 1")
            duplicate_key_groups = int(q.fetchone()[0])
            duplicate_rows = int(total_rows - unique_keys)
            q = conn.execute(
                "SELECT date, code, cnt FROM key_counts WHERE cnt > 1 ORDER BY cnt DESC, date DESC, code ASC LIMIT ?",
                (int(args.sample_duplicate_keys),),
            )
            duplicate_key_samples = [
                {"date": row[0], "code": row[1], "count": int(row[2])}
                for row in q.fetchall()
            ]
        finally:
            conn.close()

    checks: list[CheckRow] = [
        CheckRow("rows_total", "INFO", int(total_rows), "-", "Total rows scanned."),
        CheckRow("unique_codes", "INFO", int(len(unique_codes_seen)), "-", "Distinct stock codes."),
        CheckRow(
            "date_range",
            "INFO",
            f"{min_date.strftime('%Y-%m-%d') if min_date is not None else 'n/a'} -> {max_date.strftime('%Y-%m-%d') if max_date is not None else 'n/a'}",
            "-",
            "Observed min/max trade date.",
        ),
        CheckRow(
            "duplicate_rows_on_date_code",
            "PASS" if duplicate_rows == 0 else "FAIL",
            int(duplicate_rows),
            "== 0",
            "Exact duplicate rows counted by key table.",
        ),
        CheckRow(
            "duplicate_key_groups_on_date_code",
            "PASS" if duplicate_key_groups == 0 else "FAIL",
            int(duplicate_key_groups),
            "== 0",
            "Number of (date,code) groups with count > 1.",
        ),
        CheckRow(
            "missing_key_rows",
            "PASS" if missing_key_rows == 0 else "FAIL",
            int(missing_key_rows),
            "== 0",
            "Rows where date/code is missing.",
        ),
        CheckRow(
            "missing_core_numeric_rows",
            "PASS" if missing_core_rows == 0 else "WARN",
            int(missing_core_rows),
            "== 0 (recommended)",
            "Rows missing one of open/high/low/close/volume/amount.",
        ),
        CheckRow(
            "invalid_ohlc_rows",
            "PASS" if invalid_ohlc_rows == 0 else "FAIL",
            int(invalid_ohlc_rows),
            "== 0",
            "Rows violating OHLC consistency.",
        ),
        CheckRow(
            "negative_volume_or_amount_rows",
            "PASS" if negative_turnover_rows == 0 else "FAIL",
            int(negative_turnover_rows),
            "== 0",
            "Rows with negative volume or amount.",
        ),
        CheckRow(
            "future_date_rows",
            "PASS" if future_date_rows == 0 else "WARN",
            int(future_date_rows),
            "== 0",
            "Rows later than current local date.",
        ),
    ]

    status_rank = {"FAIL": 3, "WARN": 2, "PASS": 1, "INFO": 0}
    overall = "PASS"
    max_rank = max(status_rank.get(row.status, 0) for row in checks)
    for label, rank in status_rank.items():
        if rank == max_rank:
            overall = label
            break

    checks_df = pd.DataFrame([row.as_dict() for row in checks])
    checks_csv_path = out_dir / f"{prefix}_checks.csv"
    checks_md_path = out_dir / f"{prefix}_checks.md"
    summary_json_path = out_dir / f"{prefix}_summary.json"
    duplicate_csv_path = out_dir / f"{prefix}_duplicate_key_samples.csv"
    missing_core_sample_csv_path = out_dir / f"{prefix}_missing_core_samples.csv"
    invalid_ohlc_sample_csv_path = out_dir / f"{prefix}_invalid_ohlc_samples.csv"

    checks_df.to_csv(checks_csv_path, index=False)
    checks_md_path.write_text(_rows_to_markdown_table(checks), encoding="utf-8")
    pd.DataFrame(duplicate_key_samples, columns=["date", "code", "count"]).to_csv(duplicate_csv_path, index=False)
    pd.DataFrame(missing_core_samples, columns=["date", "code"] + CORE_NUMERIC_COLUMNS).to_csv(
        missing_core_sample_csv_path, index=False
    )
    pd.DataFrame(invalid_ohlc_samples, columns=["date", "code"] + CORE_NUMERIC_COLUMNS).to_csv(
        invalid_ohlc_sample_csv_path, index=False
    )

    summary_payload = {
        "overall_status": overall,
        "data_path": str(data_path),
        "rows_total": int(total_rows),
        "unique_codes": int(len(unique_codes_seen)),
        "date_min": min_date.strftime("%Y-%m-%d") if min_date is not None else "",
        "date_max": max_date.strftime("%Y-%m-%d") if max_date is not None else "",
        "duplicate_rows_on_date_code": int(duplicate_rows),
        "duplicate_key_groups_on_date_code": int(duplicate_key_groups),
        "checks_csv_path": str(checks_csv_path),
        "checks_md_path": str(checks_md_path),
        "duplicate_key_samples_csv_path": str(duplicate_csv_path),
        "missing_core_samples_csv_path": str(missing_core_sample_csv_path),
        "invalid_ohlc_samples_csv_path": str(invalid_ohlc_sample_csv_path),
    }
    summary_json_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"[OK] T10 consistency outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
