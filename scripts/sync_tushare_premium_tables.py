from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.tushare_sync import (
    load_local_history_csv,
    sync_tushare_chip_daily,
    sync_tushare_flow_daily,
    sync_tushare_fundamental_events,
    sync_tushare_hk_hold_daily,
    sync_tushare_limit_sentiment_daily,
    sync_tushare_report_rc_events,
)


@dataclass(frozen=True)
class PremiumSyncTask:
    name: str
    filename: str
    date_columns: tuple[str, ...]
    optional: bool = False


def _max_timestamp_from_csv(path: str | Path, date_columns: tuple[str, ...]) -> pd.Timestamp | None:
    source = Path(path)
    if not source.exists():
        return None
    header = pd.read_csv(source, nrows=0).columns.tolist()
    active_cols = [col for col in date_columns if col in header]
    if not active_cols:
        return None
    max_ts: pd.Timestamp | None = None
    for chunk in pd.read_csv(source, usecols=active_cols, chunksize=300_000, low_memory=False):
        for col in active_cols:
            col_ts = pd.to_datetime(chunk[col], errors="coerce")
            col_max = col_ts.max()
            if pd.isna(col_max):
                continue
            max_ts = col_max if max_ts is None or col_max > max_ts else max_ts
    return max_ts


def _count_rows(path: str | Path) -> int:
    source = Path(path)
    if not source.exists():
        return 0
    rows = 0
    for chunk in pd.read_csv(source, chunksize=300_000, low_memory=False):
        rows += len(chunk)
    return int(rows)


def _is_quota_or_permission_error(exc: Exception) -> bool:
    text = str(exc).lower()
    tokens = (
        "每天最多访问该接口",
        "每天最多调用",
        "daily limit",
        "today's limit",
        "权限",
        "permission",
        "抱歉",
    )
    return any(token in text for token in tokens)


def _resolve_table_start_date(
    *,
    args_start_date: str,
    global_start_date: str,
    global_end_date: str,
    existing_path: Path,
    date_columns: tuple[str, ...],
    overlap_days: int,
    full_refresh: bool,
) -> tuple[str, pd.Timestamp | None]:
    if args_start_date:
        return pd.Timestamp(args_start_date).strftime("%Y-%m-%d"), None
    if full_refresh:
        return pd.Timestamp(global_start_date).strftime("%Y-%m-%d"), None

    existing_max = _max_timestamp_from_csv(existing_path, date_columns=date_columns)
    if existing_max is None:
        return pd.Timestamp(global_start_date).strftime("%Y-%m-%d"), None

    overlap = max(int(overlap_days), 0)
    start_ts = existing_max.normalize() - pd.Timedelta(days=overlap)
    floor_ts = pd.Timestamp(global_start_date).normalize()
    ceil_ts = pd.Timestamp(global_end_date).normalize()
    if start_ts < floor_ts:
        start_ts = floor_ts
    if start_ts > ceil_ts:
        start_ts = ceil_ts
    return start_ts.strftime("%Y-%m-%d"), existing_max


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync premium Tushare raw tables into local CSV caches.")
    parser.add_argument("--daily-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=5,
        help="When incremental sync is used, backfill this many calendar days from the local max date.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore local max-date inference and pull from --start-date (or daily min date) to --end-date.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "data"))
    parser.add_argument("--http-url", default=None)
    parser.add_argument("--proxy-url", default=None)
    parser.add_argument("--bypass-system-proxy", action="store_true")
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument(
        "--task-max-retries",
        type=int,
        default=2,
        help="Max retries per premium sync task before fallback/fail.",
    )
    parser.add_argument(
        "--task-retry-sleep-seconds",
        type=float,
        default=5.0,
        help="Base sleep between task retries (exponential backoff).",
    )
    parser.add_argument(
        "--degrade-report-rc-on-error",
        dest="degrade_report_rc_on_error",
        action="store_true",
        help="When report_rc task still fails after retries, keep existing cache and continue the workflow.",
    )
    parser.add_argument(
        "--no-degrade-report-rc-on-error",
        dest="degrade_report_rc_on_error",
        action="store_false",
        help="Disable report_rc downgrade; let report_rc failure stop the workflow.",
    )
    parser.set_defaults(degrade_report_rc_on_error=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    daily = load_local_history_csv(args.daily_path)
    start_date = args.start_date or daily["date"].min().strftime("%Y-%m-%d")
    end_date = args.end_date or daily["date"].max().strftime("%Y-%m-%d")
    codes = sorted(daily["code"].astype(str).dropna().unique().tolist())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        PremiumSyncTask(name="fundamental_events", filename="tushare_fundamental_events.csv", date_columns=("ann_date", "end_date")),
        PremiumSyncTask(name="report_rc_events", filename="tushare_report_rc_events.csv", date_columns=("report_date",), optional=True),
        PremiumSyncTask(name="flow_daily", filename="tushare_flow_daily.csv", date_columns=("date",)),
        PremiumSyncTask(name="chip_daily", filename="tushare_chip_daily.csv", date_columns=("date",)),
        PremiumSyncTask(name="hk_hold_daily", filename="tushare_hk_hold_daily.csv", date_columns=("date",)),
        PremiumSyncTask(name="limit_sentiment_daily", filename="tushare_limit_sentiment_daily.csv", date_columns=("date",)),
    ]

    for task in tasks:
        existing_path = out_dir / task.filename
        table_start, existing_max = _resolve_table_start_date(
            args_start_date=args.start_date,
            global_start_date=start_date,
            global_end_date=end_date,
            existing_path=existing_path,
            date_columns=task.date_columns,
            overlap_days=args.overlap_days,
            full_refresh=args.full_refresh,
        )
        if pd.Timestamp(table_start) > pd.Timestamp(end_date):
            print(
                f"[SKIP] {task.name}: inferred start `{table_start}` is after end `{end_date}`; keep existing cache.",
                flush=True,
            )
            continue

        existing_max_text = existing_max.strftime("%Y-%m-%d") if existing_max is not None else "none"
        print(
            f"[START] {task.name}: range `{table_start}` -> `{end_date}` "
            f"(existing_max={existing_max_text}, overlap_days={args.overlap_days}, full_refresh={args.full_refresh})",
            flush=True,
        )
        def _run_task_once() -> tuple[Path, int]:
            if task.name == "fundamental_events":
                return sync_tushare_fundamental_events(
                    existing_path=existing_path,
                    start_date=table_start,
                    end_date=end_date,
                    codes=codes,
                    http_url=args.http_url,
                    proxy_url=args.proxy_url,
                    bypass_system_proxy=args.bypass_system_proxy,
                    pause_seconds=args.pause_seconds,
                )
            if task.name == "report_rc_events":
                return sync_tushare_report_rc_events(
                    existing_path=existing_path,
                    start_date=table_start,
                    end_date=end_date,
                    codes=codes,
                    http_url=args.http_url,
                    proxy_url=args.proxy_url,
                    bypass_system_proxy=args.bypass_system_proxy,
                    pause_seconds=args.pause_seconds,
                )
            if task.name == "flow_daily":
                return sync_tushare_flow_daily(
                    existing_path=existing_path,
                    start_date=table_start,
                    end_date=end_date,
                    http_url=args.http_url,
                    proxy_url=args.proxy_url,
                    bypass_system_proxy=args.bypass_system_proxy,
                    pause_seconds=args.pause_seconds,
                )
            if task.name == "chip_daily":
                return sync_tushare_chip_daily(
                    existing_path=existing_path,
                    start_date=table_start,
                    end_date=end_date,
                    codes=codes,
                    http_url=args.http_url,
                    proxy_url=args.proxy_url,
                    bypass_system_proxy=args.bypass_system_proxy,
                    pause_seconds=args.pause_seconds,
                )
            if task.name == "hk_hold_daily":
                return sync_tushare_hk_hold_daily(
                    existing_path=existing_path,
                    start_date=table_start,
                    end_date=end_date,
                    http_url=args.http_url,
                    proxy_url=args.proxy_url,
                    bypass_system_proxy=args.bypass_system_proxy,
                    pause_seconds=args.pause_seconds,
                )
            if task.name == "limit_sentiment_daily":
                return sync_tushare_limit_sentiment_daily(
                    existing_path=existing_path,
                    start_date=table_start,
                    end_date=end_date,
                    http_url=args.http_url,
                    proxy_url=args.proxy_url,
                    bypass_system_proxy=args.bypass_system_proxy,
                    pause_seconds=args.pause_seconds,
                )
            raise ValueError(f"Unexpected task name: {task.name}")

        started = time.perf_counter()
        retry_total = max(int(args.task_max_retries), 0)
        path: Path | None = None
        rows: int | None = None
        last_exc: Exception | None = None
        for attempt in range(retry_total + 1):
            try:
                path, rows = _run_task_once()
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                is_last = attempt >= retry_total
                if not is_last:
                    sleep_seconds = max(float(args.task_retry_sleep_seconds), 0.0) * (2**attempt)
                    print(
                        f"[WARN] {task.name}: attempt {attempt + 1}/{retry_total + 1} failed: {exc} ; "
                        f"sleep {sleep_seconds:.1f}s then retry",
                        flush=True,
                    )
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    continue

                if task.name == "report_rc_events" and args.degrade_report_rc_on_error and (
                    task.optional or _is_quota_or_permission_error(exc)
                ):
                    rows = _count_rows(existing_path)
                    path = existing_path
                    print(
                        f"[DEGRADE] {task.name}: fallback to existing cache due to error: {exc} "
                        f"(rows={rows}, path={path})",
                        flush=True,
                    )
                    last_exc = None
                    break

        if last_exc is not None or path is None or rows is None:
            raise last_exc if last_exc is not None else RuntimeError(f"{task.name} failed without output.")

        latest_ts = _max_timestamp_from_csv(path, task.date_columns)
        latest_text = latest_ts.strftime("%Y-%m-%d") if latest_ts is not None else "none"
        elapsed = time.perf_counter() - started
        print(f"[DONE] {task.name}: rows={rows}, latest={latest_text}, elapsed={elapsed:.1f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
