from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _sorted_frame(df: pd.DataFrame, *, date_col: str = "date", code_col: str = "stock_code") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[code_col] = out[code_col].astype(str)
    return out.sort_values([code_col, date_col]).reset_index(drop=True)


def check_future_leakage(
    df: pd.DataFrame,
    signal_col: str,
    position_col: str,
    return_col: str,
    *,
    date_col: str = "date",
    code_col: str = "stock_code",
) -> dict[str, Any]:
    out = _sorted_frame(df, date_col=date_col, code_col=code_col)
    same_day = pd.to_numeric(out[signal_col], errors="coerce")
    lagged = out.groupby(code_col)[signal_col].shift(1)
    position = pd.to_numeric(out[position_col], errors="coerce")
    same_day_match = float((position.round(8) == pd.to_numeric(same_day, errors="coerce").round(8)).mean())
    lagged_match = float((position.round(8) == pd.to_numeric(lagged, errors="coerce").round(8)).mean())
    suspicious_rows = int(((position.round(8) == pd.to_numeric(same_day, errors="coerce").round(8)) & (position != 0)).sum())
    return {
        "check": "future_leakage",
        "status": "PASS" if lagged_match >= same_day_match else "WARN",
        "same_day_match_ratio": same_day_match,
        "lagged_match_ratio": lagged_match,
        "suspicious_same_day_rows": suspicious_rows,
        "return_col": return_col,
        "message": "position should look more like signal.shift(1) than same-day signal",
    }


def check_signal_alignment(
    df: pd.DataFrame,
    signal_col: str,
    position_col: str,
    *,
    date_col: str = "date",
    code_col: str = "stock_code",
) -> dict[str, Any]:
    out = _sorted_frame(df, date_col=date_col, code_col=code_col)
    expected = out.groupby(code_col)[signal_col].shift(1).fillna(0.0)
    position = pd.to_numeric(out[position_col], errors="coerce").fillna(0.0)
    mismatch = (position.round(8) != pd.to_numeric(expected, errors="coerce").round(8))
    return {
        "check": "signal_alignment",
        "status": "PASS" if int(mismatch.sum()) == 0 else "WARN",
        "mismatch_rows": int(mismatch.sum()),
        "mismatch_ratio": float(mismatch.mean()),
        "message": "position should align with previous-day target signal",
    }


def check_cost_model_applied(
    df: pd.DataFrame,
    gross_ret_col: str,
    net_ret_col: str,
    *,
    turnover_col: str | None = "turnover",
) -> dict[str, Any]:
    gross = pd.to_numeric(df[gross_ret_col], errors="coerce").fillna(0.0)
    net = pd.to_numeric(df[net_ret_col], errors="coerce").fillna(0.0)
    cost_drag = gross - net
    turnover = pd.to_numeric(df[turnover_col], errors="coerce").fillna(0.0) if turnover_col and turnover_col in df.columns else None
    has_turnover = turnover is not None and bool(turnover.abs().sum() > 0)
    nonzero_cost = bool(cost_drag.abs().sum() > 1e-12)
    status = "PASS" if (nonzero_cost or not has_turnover) else "WARN"
    return {
        "check": "cost_model",
        "status": status,
        "avg_cost_drag": float(cost_drag.mean()),
        "total_cost_drag": float(cost_drag.sum()),
        "turnover_present": bool(has_turnover),
        "message": "net return should differ from gross return when turnover exists",
    }


def check_missing_trading_days(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    code_col: str = "stock_code",
) -> dict[str, Any]:
    out = _sorted_frame(df, date_col=date_col, code_col=code_col)
    gaps = out.groupby(code_col)[date_col].diff().dt.days
    gap_rows = gaps[gaps > 7]
    return {
        "check": "missing_trading_days",
        "status": "PASS" if gap_rows.empty else "WARN",
        "gap_rows": int(len(gap_rows)),
        "max_gap_days": int(gap_rows.max()) if not gap_rows.empty else 0,
        "message": "large date gaps may indicate suspensions or missing daily bars",
    }


def check_limit_up_down_edge_cases(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    up_limit_col: str = "up_limit",
    down_limit_col: str = "down_limit",
) -> dict[str, Any]:
    out = df.copy()
    close = pd.to_numeric(out[close_col], errors="coerce")
    high = pd.to_numeric(out[high_col], errors="coerce")
    low = pd.to_numeric(out[low_col], errors="coerce")
    if up_limit_col in out.columns and down_limit_col in out.columns:
        up_limit = pd.to_numeric(out[up_limit_col], errors="coerce")
        down_limit = pd.to_numeric(out[down_limit_col], errors="coerce")
        up_block = close.ge(up_limit * 0.999) & high.ge(up_limit * 0.999)
        down_block = close.le(down_limit * 1.001) & low.le(down_limit * 1.001)
        heuristic = False
    else:
        ret_1 = close.groupby(out.get("stock_code", out.get("code"))).pct_change().fillna(0.0)
        up_block = (close == high) & ret_1.gt(0.095)
        down_block = (close == low) & ret_1.lt(-0.095)
        heuristic = True
    return {
        "check": "limit_edge_cases",
        "status": "PASS",
        "up_limit_like_rows": int(up_block.fillna(False).sum()),
        "down_limit_like_rows": int(down_block.fillna(False).sum()),
        "used_heuristic": heuristic,
        "message": "continuous limit-up/down rows can block real execution even if the sandbox books returns",
    }


def build_audit_report(
    df: pd.DataFrame,
    *,
    signal_col: str,
    position_col: str,
    return_col: str,
    gross_ret_col: str,
    net_ret_col: str,
    turnover_col: str = "turnover",
    date_col: str = "date",
    code_col: str = "stock_code",
) -> dict[str, Any]:
    checks = [
        check_future_leakage(
            df,
            signal_col=signal_col,
            position_col=position_col,
            return_col=return_col,
            date_col=date_col,
            code_col=code_col,
        ),
        check_signal_alignment(
            df,
            signal_col=signal_col,
            position_col=position_col,
            date_col=date_col,
            code_col=code_col,
        ),
        check_cost_model_applied(
            df,
            gross_ret_col=gross_ret_col,
            net_ret_col=net_ret_col,
            turnover_col=turnover_col,
        ),
        check_missing_trading_days(df, date_col=date_col, code_col=code_col),
        check_limit_up_down_edge_cases(df),
    ]
    status = "PASS" if all(item["status"] == "PASS" for item in checks[:3]) else "WARN"
    return {
        "overall_status": status,
        "checks": checks,
    }


def audit_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Strategy Audit Report",
        "",
        f"- overall_status: `{report.get('overall_status', 'WARN')}`",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for item in report.get("checks", []):
        detail = item.get("message", "")
        if "mismatch_rows" in item:
            detail = f"{detail}; mismatch_rows={item['mismatch_rows']}"
        if "suspicious_same_day_rows" in item:
            detail = f"{detail}; suspicious_same_day_rows={item['suspicious_same_day_rows']}"
        lines.append(f"| {item.get('check', '')} | {item.get('status', '')} | {detail} |")
    return "\n".join(lines)
