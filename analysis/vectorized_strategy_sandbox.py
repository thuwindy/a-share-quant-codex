from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.strategy_audit import audit_report_markdown, build_audit_report


REQUIRED_COLUMNS = ["date", "stock_code", "open", "high", "low", "close", "volume", "amount"]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {"code": "stock_code"}
    out = out.rename(columns=rename_map)
    missing = [col for col in REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["stock_code"] = out["stock_code"].astype(str)
    for col in ("open", "high", "low", "close", "volume", "amount"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["stock_code", "date"]).reset_index(drop=True)


def _rolling_sma(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return series.rolling(window, min_periods=min_periods or max(window // 2, 1)).mean()


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = -delta.where(delta < 0.0, 0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _add_common_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("stock_code", group_keys=False)
    out["ret_1"] = g["close"].pct_change().fillna(0.0)
    out["ma20"] = g["close"].transform(lambda s: _rolling_sma(s, 20, 10))
    out["ma60"] = g["close"].transform(lambda s: _rolling_sma(s, 60, 20))
    out["ma200"] = g["close"].transform(lambda s: _rolling_sma(s, 200, 60))
    out["rolling_high_20_prev"] = g["high"].transform(lambda s: s.rolling(20, min_periods=10).max()).shift(1)
    out["rolling_low_10_prev"] = g["low"].transform(lambda s: s.rolling(10, min_periods=5).min()).shift(1)
    out["rsi14"] = g["close"].transform(_compute_rsi)
    return out


def _stateful_target(entry: pd.Series, exit_: pd.Series, *, date_index: pd.Index) -> pd.Series:
    in_pos = False
    result: list[float] = []
    for is_entry, is_exit in zip(entry.fillna(False).astype(bool), exit_.fillna(False).astype(bool)):
        if not in_pos and is_entry:
            in_pos = True
        elif in_pos and is_exit:
            in_pos = False
        result.append(1.0 if in_pos else 0.0)
    return pd.Series(result, index=date_index, dtype=float)


def generate_template_signals(df: pd.DataFrame, template: str) -> pd.DataFrame:
    out = _add_common_features(df)
    frames: list[pd.DataFrame] = []
    for code, sl in out.groupby("stock_code", sort=False):
        part = sl.copy()
        if template == "breakout":
            entry = part["close"].gt(part["rolling_high_20_prev"]) & part["close"].gt(part["ma200"])
            exit_ = part["close"].lt(part["rolling_low_10_prev"])
        elif template == "ma_trend":
            entry = part["ma20"].gt(part["ma60"]) & part["close"].gt(part["ma20"])
            exit_ = part["ma20"].lt(part["ma60"]) | part["close"].lt(part["ma20"])
        elif template == "reversal":
            entry = part["rsi14"].lt(30.0) & part["close"].gt(part.groupby("stock_code")["close"].shift(1))
            exit_ = part["rsi14"].gt(55.0)
        else:
            raise ValueError(f"unsupported template: {template}")
        part["signal_target"] = _stateful_target(entry, exit_, date_index=part.index)
        part["entry_rule"] = entry.astype(int)
        part["exit_rule"] = exit_.astype(int)
        frames.append(part)
    return pd.concat(frames, ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)


def run_sandbox(
    df: pd.DataFrame,
    *,
    template: str,
    execution_lag: int = 1,
    commission: float = 0.0003,
    slippage: float = 0.0005,
    sell_tax: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if execution_lag < 1:
        raise ValueError("sandbox defaults to T+1 style execution; execution_lag must be >= 1")
    detail = generate_template_signals(df, template=template)
    detail["position"] = detail.groupby("stock_code")["signal_target"].shift(execution_lag).fillna(0.0)
    detail["position_change"] = (
        detail.groupby("stock_code")["position"].diff().fillna(detail["position"]).abs()
    )
    active_count = detail.groupby("date")["position"].transform("sum").replace(0.0, np.nan)
    detail["weight"] = (detail["position"] / active_count).fillna(0.0)
    detail["gross_ret"] = detail["weight"] * detail["ret_1"].fillna(0.0)
    entry_cost = detail["position_change"] * detail["weight"].abs() * (commission + slippage)
    sell_cost = (
        detail.groupby("stock_code")["weight"].diff().fillna(detail["weight"]).lt(0).astype(float).abs()
        * detail["weight"].abs()
        * sell_tax
    )
    detail["turnover"] = detail["position_change"] * detail["weight"].abs()
    detail["cost"] = entry_cost.fillna(0.0) + sell_cost.fillna(0.0)
    detail["net_ret"] = detail["gross_ret"] - detail["cost"]

    portfolio = (
        detail.groupby("date", as_index=False)
        .agg(
            gross_ret=("gross_ret", "sum"),
            net_ret=("net_ret", "sum"),
            turnover=("turnover", "sum"),
            active_positions=("position", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    portfolio["gross_nav"] = (1.0 + portfolio["gross_ret"]).cumprod()
    portfolio["net_nav"] = (1.0 + portfolio["net_ret"]).cumprod()

    audit_input = detail[["date", "stock_code", "signal_target", "position", "ret_1", "gross_ret", "net_ret", "turnover", "close", "high", "low"]].copy()
    report = build_audit_report(
        audit_input,
        signal_col="signal_target",
        position_col="position",
        return_col="ret_1",
        gross_ret_col="gross_ret",
        net_ret_col="net_ret",
        turnover_col="turnover",
    )
    report.update(
        {
            "template": template,
            "signal_definition": f"{template} target_position at close",
            "position_definition": f"signal_target.shift({execution_lag})",
            "execution_lag": execution_lag,
            "execution_note": "close-generated signal, close-to-close return proxy with explicit T+1 position alignment",
            "costs": {
                "commission": commission,
                "slippage": slippage,
                "sell_tax": sell_tax,
            },
            "missing_value_policy": "drop invalid price rows; position and return fields fillna(0) after alignment",
            "suspension_policy": "missing bars remain gaps and are surfaced by missing trading days audit",
            "extreme_edge_cases": [
                "连续涨停可能无法买入",
                "连续跌停可能无法卖出",
                "开盘跳空与收盘信号并不等价",
            ],
        }
    )
    return detail, portfolio, report


def _write_outputs(
    *,
    output_dir: Path,
    detail: pd.DataFrame,
    portfolio: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "sandbox_detail.csv", index=False)
    portfolio.to_csv(output_dir / "sandbox_portfolio.csv", index=False)
    (output_dir / "strategy_audit_report.md").write_text(audit_report_markdown(report), encoding="utf-8")
    (output_dir / "strategy_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a transparent vectorized strategy sandbox.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--template", choices=("breakout", "ma_trend", "reversal"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--execution-lag", type=int, default=1)
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--sell-tax", type=float, default=0.001)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.data_path, low_memory=False)
    df = _ensure_columns(df)
    if args.start_date:
        df = df.loc[df["date"] >= pd.Timestamp(args.start_date)].copy()
    if args.end_date:
        df = df.loc[df["date"] <= pd.Timestamp(args.end_date)].copy()
    detail, portfolio, report = run_sandbox(
        df,
        template=str(args.template),
        execution_lag=int(args.execution_lag),
        commission=float(args.commission),
        slippage=float(args.slippage),
        sell_tax=float(args.sell_tax),
    )
    _write_outputs(output_dir=Path(args.output_dir), detail=detail, portfolio=portfolio, report=report)
    print(
        json.dumps(
            {
                "template": report["template"],
                "overall_status": report["overall_status"],
                "rows": int(len(detail)),
                "dates": int(portfolio["date"].nunique()),
                "final_net_nav": float(portfolio["net_nav"].iloc[-1]) if not portfolio.empty else 1.0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
