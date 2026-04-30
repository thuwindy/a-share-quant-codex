from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.backtest.engine import BacktestConfig, DailyBacktester
from ashare_quant.data.csv_adapter import CSVDataSource, ensure_price_views
from ashare_quant.labels.label_builder import ExecutionSpec, add_label_columns, label_column_name
from ashare_quant.pipeline import DEFAULT_RESEARCH, load_json


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
    parser = argparse.ArgumentParser(
        description="Run T21: validate signal/label/execution alignment under close/open/vwap variants."
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "daily_monitor_slice_fundamental.csv"))
    parser.add_argument("--research-config", default=str(ROOT / "configs" / "research_production_default.json"))
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--horizons", default="", help="Comma-separated horizons override, e.g. 5,10,20.")
    parser.add_argument("--execution-lag", type=int, default=1)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "data_quality"))
    parser.add_argument("--output-prefix", default="t21_execution_alignment")
    parser.add_argument("--sample-size", type=int, default=300_000, help="Max rows for validation speed.")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser


def _parse_horizons(raw: str, cfg: dict) -> list[int]:
    if raw.strip():
        return sorted({int(v) for v in raw.split(",") if v.strip() and int(v) > 0})
    use_h = cfg.get("use_horizons") or cfg.get("label_horizons") or [cfg.get("label_horizon", 5)]
    return sorted({int(v) for v in use_h if int(v) > 0})


def _normalize_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _markdown(rows: list[CheckRow]) -> str:
    lines = [
        "# T21 Execution Alignment",
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


def _expected_label_arrays(df: pd.DataFrame, horizon: int, execution_price: str, execution_lag: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    g = df.groupby("code", group_keys=False)
    entry_close = g["research_close"].shift(-execution_lag)
    if execution_price == "close":
        expected = g["research_close"].shift(-(execution_lag + horizon)).div(entry_close.where(entry_close != 0)) - 1.0
        expected_available = g["date"].shift(-(execution_lag + horizon))
    elif execution_price == "open":
        entry_open = g["research_open"].shift(-execution_lag)
        expected = g["research_close"].shift(-horizon).div(entry_open.where(entry_open != 0)) - 1.0
        expected_available = g["date"].shift(-horizon)
    elif execution_price == "vwap":
        entry_vwap = g["research_vwap"].shift(-execution_lag)
        expected = g["research_close"].shift(-horizon).div(entry_vwap.where(entry_vwap != 0)) - 1.0
        expected_available = g["date"].shift(-horizon)
    else:
        raise ValueError(f"Unsupported execution_price={execution_price}")
    expected_execution = g["date"].shift(-execution_lag)
    return expected, _normalize_datetime(expected_execution), _normalize_datetime(expected_available)


def _datetime_mismatch_count(left: pd.Series, right: pd.Series) -> int:
    left_dt = _normalize_datetime(left)
    right_dt = _normalize_datetime(right)
    both_na = left_dt.isna() & right_dt.isna()
    return int((~both_na & (left_dt != right_dt)).sum())


def _run_engine_smoke_variant(execution_price: str) -> tuple[float, float]:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "code": ["000001.SZ", "000001.SZ"],
            "research_open": [10.0, 12.0],
            "research_high": [10.0, 20.0],
            "research_low": [10.0, 12.0],
            "research_close": [10.0, 18.0],
            "research_vwap": [10.0, 15.5],
            "amount": [100_000_000.0, 100_000_000.0],
            "can_buy": [True, True],
            "can_sell": [True, True],
        }
    )
    targets = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-02")],
            "execution_date": [pd.Timestamp("2024-01-03")],
            "sleeve": [0],
            "code": ["000001.SZ"],
            "target_weight": [1.0],
        }
    )
    result, _ = DailyBacktester(
        config=BacktestConfig(
            execution_price=execution_price,
            execution_lag=1,
            commission=0.0,
            slippage=0.0,
            sell_tax=0.0,
            use_liquidity_aware_cost=False,
            sleeve_count=1,
        )
    ).run(panel=panel, targets=targets)
    observed = float(result.iloc[-1]["gross_return"])
    if execution_price == "close":
        expected = 0.0
    elif execution_price == "open":
        expected = 18.0 / 12.0 - 1.0
    else:
        expected = 18.0 / 15.5 - 1.0
    return observed, expected


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    cfg_path = Path(args.research_config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Research config not found: {cfg_path}")

    research_cfg = {**DEFAULT_RESEARCH, **load_json(cfg_path)}
    horizons = _parse_horizons(args.horizons, research_cfg)
    lag = int(args.execution_lag)

    df = CSVDataSource(data_path, adjust=args.adjust).load()
    df = ensure_price_views(df).sort_values(["date", "code"]).reset_index(drop=True)
    if args.sample_size > 0 and len(df) > args.sample_size:
        keep_dates = sorted(df["date"].drop_duplicates())[-300:]
        sampled = df[df["date"].isin(keep_dates)].copy()
        if len(sampled) > args.sample_size:
            sampled = sampled.iloc[-args.sample_size :].copy()
        df = sampled.reset_index(drop=True)

    base_cols = ["date", "code", "research_open", "research_close", "research_vwap", "industry", "market_cap"]
    frame = df[base_cols].copy()

    checks: list[CheckRow] = []
    checks.append(CheckRow("rows_scanned", "INFO", int(len(frame)), "-", "Rows used in T21 validation."))
    checks.append(CheckRow("horizons", "INFO", ",".join(str(h) for h in horizons), "-", "Horizons used for label checks."))
    checks.append(CheckRow("execution_lag", "INFO", int(lag), "-", "Execution lag used in checks."))

    summary_rows: list[dict] = []

    for execution_price in ("close", "open", "vwap"):
        labeled = add_label_columns(
            frame,
            horizons=horizons,
            spec=ExecutionSpec(
                signal_time="close",
                execution_price=execution_price,
                execution_lag=lag,
                holding_window=max(horizons),
            ),
        )
        for horizon in horizons:
            raw_col = label_column_name(horizon, "raw")
            actual = pd.to_numeric(labeled[raw_col], errors="coerce")
            expected, expected_exec, expected_available = _expected_label_arrays(
                labeled, horizon=horizon, execution_price=execution_price, execution_lag=lag
            )
            expected = pd.to_numeric(expected, errors="coerce")
            valid = actual.notna() & expected.notna()
            diffs = (actual[valid] - expected[valid]).abs() if int(valid.sum()) > 0 else pd.Series(dtype=float)
            max_abs_diff = float(diffs.max()) if not diffs.empty else 0.0
            mismatch_count = int((diffs > args.tolerance).sum()) if not diffs.empty else 0

            actual_exec = labeled[f"execution_date_{horizon}d"]
            actual_available = labeled[f"label_available_date_{horizon}d"]
            execution_date_mismatch = _datetime_mismatch_count(actual_exec, expected_exec)
            available_date_mismatch = _datetime_mismatch_count(actual_available, expected_available)

            status = (
                "PASS"
                if mismatch_count == 0 and execution_date_mismatch == 0 and available_date_mismatch == 0
                else "FAIL"
            )
            check_name = f"label_formula::{execution_price}::{horizon}d"
            note = (
                f"raw label vs explicit formula; execution_date and label_available_date match."
            )
            checks.append(
                CheckRow(
                    check=check_name,
                    status=status,
                    value=f"max_abs_diff={max_abs_diff:.3e}, mismatches={mismatch_count}, exec_mismatch={execution_date_mismatch}, available_mismatch={available_date_mismatch}",
                    threshold=f"tol<={args.tolerance:.1e} and mismatches==0",
                    note=note,
                )
            )
            summary_rows.append(
                {
                    "execution_price": execution_price,
                    "horizon": horizon,
                    "rows_compared": int(valid.sum()),
                    "max_abs_diff": max_abs_diff,
                    "formula_mismatch_count": mismatch_count,
                    "execution_date_mismatch_count": execution_date_mismatch,
                    "label_available_date_mismatch_count": available_date_mismatch,
                    "status": status,
                }
            )

        observed, expected = _run_engine_smoke_variant(execution_price)
        smoke_diff = abs(observed - expected)
        smoke_status = "PASS" if smoke_diff <= args.tolerance else "FAIL"
        checks.append(
            CheckRow(
                check=f"engine_smoke::{execution_price}",
                status=smoke_status,
                value=f"observed={observed:.12f}, expected={expected:.12f}, abs_diff={smoke_diff:.3e}",
                threshold=f"abs_diff<={args.tolerance:.1e}",
                note="Engine execution-price behavior on deterministic synthetic panel.",
            )
        )

    status_rank = {"FAIL": 3, "WARN": 2, "PASS": 1, "INFO": 0}
    max_rank = max(status_rank.get(row.status, 0) for row in checks)
    overall_status = next(label for label, rank in status_rank.items() if rank == max_rank)

    checks_csv_path = out_dir / f"{args.output_prefix}_checks.csv"
    checks_md_path = out_dir / f"{args.output_prefix}_checks.md"
    summary_csv_path = out_dir / f"{args.output_prefix}_summary.csv"
    summary_json_path = out_dir / f"{args.output_prefix}_summary.json"

    pd.DataFrame([row.as_dict() for row in checks]).to_csv(checks_csv_path, index=False)
    checks_md_path.write_text(_markdown(checks), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    summary = {
        "overall_status": overall_status,
        "data_path": str(data_path),
        "research_config": str(cfg_path),
        "adjust": args.adjust,
        "rows_scanned": int(len(frame)),
        "horizons": horizons,
        "execution_lag": lag,
        "checks_csv_path": str(checks_csv_path),
        "checks_md_path": str(checks_md_path),
        "summary_csv_path": str(summary_csv_path),
        "run_date": str(date.today()),
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] T21 execution alignment outputs saved to {out_dir}")


if __name__ == "__main__":
    main()

