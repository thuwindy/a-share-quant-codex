from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_common import ROOT, call_openai_compatible_chat, compact_json, resolve_llm_settings  # noqa: E402
from llm_report_renderer import (  # noqa: E402
    build_structured_facts,
    render_deterministic_report,
    resolve_artifacts,
    validate_numbers_grounded_in_json,
    validate_report_context,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a one-off PushPlus LLM-enhanced evening brief.")
    parser.add_argument("--date-key", default="")
    parser.add_argument("--master-data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--main-monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_auto"))
    parser.add_argument("--elastic-monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_under20_elastic"))
    parser.add_argument("--shortline-dir", default=str(ROOT / "outputs" / "shortline_opportunities"))
    parser.add_argument("--risk-dir", default=str(ROOT / "outputs" / "risk_governor"))
    parser.add_argument("--candidate-quality-root", default=str(ROOT / "outputs"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "llm_pushplus_evening_brief"))
    parser.add_argument("--token", default="")
    parser.add_argument("--topic", default="")
    parser.add_argument("--channel", default="wechat")
    parser.add_argument("--top-main", type=int, default=5)
    parser.add_argument("--top-elastic", type=int, default=5)
    parser.add_argument("--top-shortline", type=int, default=3)
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--llm-max-tokens", type=int, default=900)
    parser.add_argument("--stock-agent-max-tokens", type=int, default=520)
    parser.add_argument("--skip-master-freshness-check", action="store_true")
    parser.add_argument("--allow-historical-date", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_missing(value: Any) -> bool:
    if value in (None, ""):
        return True
    try:
        return bool(math.isnan(float(value)))
    except Exception:
        return False


def _fmt_score(value: Any) -> str:
    return f"{_safe_float(value):.4f}"


def _fmt_price(value: Any) -> str:
    if _is_missing(value):
        return "暂无"
    return f"{_safe_float(value):.2f}元"


def _pct(value: Any) -> str:
    return f"{_safe_float(value) * 100.0:+.2f}%"


def _trend_label(value: Any) -> str:
    val = _safe_float(value)
    if val >= 0.015:
        return "强反弹"
    if val >= 0.003:
        return "反弹"
    if val > -0.003:
        return "震荡"
    if val > -0.015:
        return "回调"
    return "下跌"


_NUMERIC_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?%?")


def _date_key_from_date(value: str) -> str:
    return str(value or "").replace("-", "")[:8]


def _latest_master_date_key(path: Path) -> str:
    try:
        from llm_common import latest_master_date

        return _date_key_from_date(latest_master_date(path))
    except Exception:
        return ""


def _latest_tushare_daily_date_key() -> tuple[str, str]:
    try:
        import pandas as pd
        import tushare as ts
    except Exception as exc:
        return "", f"dependency_unavailable:{exc}"
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        return "", "missing_tushare_token"
    try:
        today = pd.Timestamp.today().strftime("%Y%m%d")
        start = (pd.Timestamp.today() - pd.Timedelta(days=14)).strftime("%Y%m%d")
        pro = ts.pro_api(token)
        df = pro.daily(ts_code="000001.SZ", start_date=start, end_date=today)
        if df is None or df.empty:
            return "", "empty_tushare_daily_probe"
        return str(df["trade_date"].astype(str).max()), ""
    except Exception as exc:
        return "", str(exc)[:200]


def _run_stale_master_repair(args: argparse.Namespace) -> dict[str, Any]:
    env = os.environ.copy()
    env.pop("TUSHARE_HTTP_URL", None)
    env["DATA_UPDATE_RETRIES"] = env.get("DATA_UPDATE_RETRIES", "1")
    env["DATA_UPDATE_SLEEP_SECONDS"] = env.get("DATA_UPDATE_SLEEP_SECONDS", "10")
    cmd = ["bash", str(ROOT / "scripts" / "run_cloud_post_close_stable.sh")]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(900, int(args.llm_timeout_seconds) * 8),
            check=False,
        )
        return {
            "cmd": " ".join(cmd),
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-3000:],
        }
    except Exception as exc:
        return {"cmd": " ".join(cmd), "returncode": -1, "error": str(exc)[:500]}


def _run_preflight_repair(args: argparse.Namespace) -> dict[str, Any]:
    env = os.environ.copy()
    env.pop("TUSHARE_HTTP_URL", None)
    cmd = [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "preflight_evening_brief.py")]
    if str(os.environ.get("TUSHARE_BYPASS_SYSTEM_PROXY", "")).lower() in {"1", "true", "yes", "on"}:
        cmd.append("--bypass-system-proxy")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(600, int(args.llm_timeout_seconds) * 5),
            check=False,
        )
        return {
            "cmd": " ".join(cmd),
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-3000:],
        }
    except Exception as exc:
        return {"cmd": " ".join(cmd), "returncode": -1, "error": str(exc)[:500]}


def _ensure_master_fresh(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_master_freshness_check:
        return {"status": "skipped"}
    master_path = Path(args.master_data_path)
    before = _latest_master_date_key(master_path)
    latest, latest_error = _latest_tushare_daily_date_key()
    status: dict[str, Any] = {
        "status": "checked",
        "master_date_key_before": before,
        "tushare_latest_date_key": latest,
        "tushare_error": latest_error,
        "repair_attempted": False,
    }
    if latest_error:
        status["status"] = "failed"
        status["error"] = f"unable_to_verify_tushare_latest:{latest_error}"
        raise RuntimeError(json.dumps(status, ensure_ascii=False))
    if before and latest and before >= latest:
        if args.date_key and not args.allow_historical_date and args.date_key < latest:
            status["status"] = "failed"
            status["error"] = f"historical_date_blocked:{args.date_key}<{latest}"
            raise RuntimeError(json.dumps(status, ensure_ascii=False))
        status["status"] = "fresh"
        return status

    status["repair_attempted"] = True
    status["repair"] = _run_stale_master_repair(args)
    after = _latest_master_date_key(master_path)
    status["master_date_key_after"] = after
    if latest and after < latest:
        status["status"] = "failed"
        status["error"] = f"master_still_stale_after_repair:{after}<{latest}"
        raise RuntimeError(json.dumps(status, ensure_ascii=False))
    status["status"] = "repaired"
    return status


def _strategy_rows(facts: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "主策略": facts.get("main_strategy", {}).get("top_rows", [])[:5],
        "弹性池": facts.get("elastic_pool", {}).get("top_rows", [])[:5],
        "短线机会": facts.get("shortline", {}).get("cards", [])[:3],
    }


def _allowed_stock_codes(facts: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    for rows in _strategy_rows(facts).values():
        for row in rows:
            code = str(row.get("code") or "").strip()
            if code:
                codes.add(code)
    return codes


def _allowed_stock_names(facts: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for rows in _strategy_rows(facts).values():
        for row in rows:
            name = str(row.get("name") or "").strip()
            if name:
                names.add(name)
    return names


def _latest_master_name_map(data_path: str, date_key: str) -> dict[str, str]:
    try:
        import pandas as pd
    except Exception:
        return {}
    path = Path(data_path)
    if not path.exists() or not date_key:
        return {}
    target_date = f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:]}"
    mapping: dict[str, str] = {}
    try:
        for chunk in pd.read_csv(path, usecols=["date", "code", "name"], chunksize=500_000):
            chunk["date"] = chunk["date"].astype(str).str[:10]
            today = chunk.loc[chunk["date"] == target_date, ["code", "name"]].dropna()
            for row in today.to_dict("records"):
                name = str(row.get("name") or "").strip()
                code = str(row.get("code") or "").strip()
                if name and code:
                    mapping[name] = code
    except Exception:
        return {}
    return mapping


def _agent_text(agent: dict[str, Any]) -> str:
    parts = [str(agent.get("content") or "")]
    for key in ("ai_fundamental", "ai_quant_reason", "ai_operation_logic"):
        parts.append(str(agent.get(key) or ""))
    return "\n".join(parts)


def _unknown_stock_mentions(text: str, *, allowed_codes: set[str], allowed_names: set[str], master_names: dict[str, str]) -> list[str]:
    unknown: list[str] = []
    for code in sorted(set(re.findall(r"\b(?:00|30|60|68)\d{4}\.(?:SZ|SH)\b", text))):
        if code not in allowed_codes:
            unknown.append(code)
    for name, code in master_names.items():
        if name in allowed_names:
            continue
        if len(name) >= 2 and name in text:
            unknown.append(f"{name}/{code}")
            if len(unknown) >= 8:
                break
    return unknown


def _extract_agent_numbers(text: str) -> list[str]:
    visible = re.sub(r"<[^>]+>", "", text)
    return sorted(set(_NUMERIC_TOKEN_RE.findall(visible)))


def _numeric_variants_from_text(text: str) -> set[str]:
    variants: set[str] = set()
    for token in _NUMERIC_TOKEN_RE.findall(text):
        raw = token.strip().rstrip("%")
        for item in {raw, raw.lstrip("+")}:
            variants.add(item)
            try:
                value = float(item)
            except Exception:
                continue
            variants.update(
                {
                    str(int(value)) if float(value).is_integer() else str(value),
                    f"{value:.1f}",
                    f"{value:.2f}",
                    f"{value:.4f}",
                    f"{value:+.1f}",
                    f"{value:+.2f}",
                    f"{value:+.4f}",
                }
            )
    return variants


def _ungrounded_numbers_tolerant(text: str, facts: dict[str, Any]) -> list[str]:
    source = json.dumps(facts, ensure_ascii=False, default=str)
    source_variants = _numeric_variants_from_text(source)
    bad: list[str] = []
    for token in _extract_agent_numbers(text):
        normalized = token.rstrip("%")
        if normalized in source_variants or normalized.lstrip("+") in source_variants:
            continue
        try:
            value = float(normalized.lstrip("+"))
        except Exception:
            bad.append(token)
            continue
        candidates = {f"{value:.1f}", f"{value:.2f}", f"{value:.4f}", f"{value:+.1f}", f"{value:+.2f}", f"{value:+.4f}"}
        if not candidates.intersection(source_variants):
            bad.append(token)
    return bad


def _missing_required_stock_inputs(strategy_name: str, row: dict[str, Any], stock: dict[str, Any] | None) -> list[str]:
    missing: list[str] = []
    common = ["rank", "code", "name", "industry", "close"]
    if strategy_name == "主策略":
        common.extend(["score", "ml_score"])
    elif strategy_name == "弹性池":
        common.extend(["score"])
        if _is_missing(row.get("market_cap")) and _is_missing(row.get("amount")):
            missing.append("market_cap_or_amount")
    else:
        common.extend(["shortline_score", "risk_level", "current_streak", "fd_amount", "turnover_ratio", "buy_range", "stop_loss"])
    for key in common:
        if _is_missing(row.get(key)):
            missing.append(key)
    return missing


def _missing_fundamental_inputs(stock: dict[str, Any] | None) -> list[str]:
    if not stock:
        return ["tushare_fundamental"]
    missing: list[str] = []
    valuation = stock.get("valuation", {}) or {}
    financial = stock.get("financial", {}) or {}
    if _is_missing(valuation.get("pe_ttm")) and _is_missing(valuation.get("pb")):
        missing.append("valuation")
    if _is_missing(financial.get("roe_dt")):
        missing.append("roe")
    if _is_missing(financial.get("revenue_yoy")):
        missing.append("revenue_yoy")
    if _is_missing(financial.get("netprofit_yoy")):
        missing.append("netprofit_yoy")
    if _is_missing(financial.get("debt_to_assets")):
        missing.append("debt_to_assets")
    if _is_missing(financial.get("grossprofit_margin")):
        missing.append("grossprofit_margin")
    return missing


def _fallback_stock_agent(
    *,
    status: str,
    row: dict[str, Any],
    strategy_name: str,
    stock: dict[str, Any] | None,
    risk: dict[str, Any],
    error: str = "",
    raw: str = "",
) -> dict[str, Any]:
    return {
        "status": status,
        "ai_fundamental": _stock_fundamental_line(stock),
        "ai_quant_reason": _quant_reason(row, strategy_name),
        "ai_operation_logic": _operation_logic(row, strategy_name, risk),
        "error": error,
        "raw": raw[:500],
    }


def _fallback_strategy_agent(strategy_name: str, rows: list[dict[str, Any]], facts: dict[str, Any], reason: str) -> dict[str, Any]:
    if not rows:
        content = "今日该策略没有符合当前规则的候选标的，日报只保留日期一致性占位，不引用旧日期股票，也不生成交易建议。"
    else:
        risk = facts.get("risk", {})
        content = (
            f"{strategy_name}保留结构化候选池展示；本段大模型解释因硬校验未通过而降级。"
            f"风控状态为{risk.get('status') or '暂缺'}，操作建议以量化卡片和风控模块为准。"
        )
    return {"status": "fallback_guardrail", "agent": f"strategy_agent:{strategy_name}", "content": content, "error": reason}


def _fallback_market_agent(facts: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "status": "fallback_guardrail",
        "agent": "market_agent",
        "content": render_deterministic_report(facts, reason=reason),
        "error": reason,
    }


def _validate_agent_content(
    *,
    agent: dict[str, Any],
    facts: dict[str, Any],
    allowed_codes: set[str],
    allowed_names: set[str],
    master_names: dict[str, str],
    allow_numbers: bool,
) -> list[str]:
    text = _agent_text(agent)
    errors: list[str] = []
    unknown = _unknown_stock_mentions(text, allowed_codes=allowed_codes, allowed_names=allowed_names, master_names=master_names)
    if unknown:
        errors.append("unknown_stock_mentions:" + ",".join(unknown[:5]))
    if not allow_numbers and _extract_agent_numbers(text):
        errors.append("numeric_tokens_not_allowed:" + ",".join(_extract_agent_numbers(text)[:8]))
    elif allow_numbers:
        ungrounded = _ungrounded_numbers_tolerant(text, facts)
        if ungrounded:
            errors.append("ungrounded_numbers:" + ",".join(ungrounded[:8]))
    return errors


def _load_index_review(date_key: str, *, lookback: int = 5) -> dict[str, Any]:
    try:
        import pandas as pd
        import tushare as ts
    except Exception as exc:
        return {"status": "unavailable", "reason": f"dependency_unavailable:{exc}"}
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        return {"status": "unavailable", "reason": "missing_tushare_token"}
    end_date = date_key
    start_date = "20200101"
    indices = {
        "上证指数": "000001.SH",
        "创业板指": "399006.SZ",
    }
    pro = ts.pro_api(token)
    out: dict[str, Any] = {"status": "ok", "indices": {}}
    for name, code in indices.items():
        try:
            df = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
        except Exception as exc:
            out["indices"][name] = {"status": "unavailable", "reason": str(exc)[:160], "ts_code": code}
            continue
        if df is None or df.empty:
            out["indices"][name] = {"status": "unavailable", "reason": "empty_index_daily", "ts_code": code}
            continue
        df = df.sort_values("trade_date").tail(max(lookback, 20)).copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")
        latest = df.iloc[-1].to_dict()
        last_5 = df.tail(5)
        last_10 = df.tail(10)
        amount_5d_mean = float(last_5["amount"].mean()) if not last_5.empty else 0.0
        amount_latest = _safe_float(latest.get("amount"))
        ma5 = float(last_5["close"].mean()) if not last_5.empty else 0.0
        ma10 = float(last_10["close"].mean()) if not last_10.empty else 0.0
        ma20 = float(df.tail(20)["close"].mean()) if len(df) >= 20 else 0.0
        rows_df = df.tail(lookback).copy()
        rows = []
        for row in rows_df.to_dict("records"):
            pct = _safe_float(row.get("pct_chg")) / 100.0
            amount_yi = _safe_float(row.get("amount")) / 100000.0
            rows.append(
                {
                    "date": f"{str(row.get('trade_date'))[4:6]}-{str(row.get('trade_date'))[6:8]}",
                    "close": round(_safe_float(row.get("close")), 2),
                    "pct_chg": round(pct, 4),
                    "pct_chg_pct": round(pct * 100.0, 2),
                    "amount_yi": round(amount_yi, 2),
                    "trend": _trend_label(pct),
                }
            )
        week_return = rows[-1]["close"] / rows[0]["close"] - 1.0 if len(rows) >= 2 and rows[0]["close"] else 0.0
        ret_10d = _safe_float(df.iloc[-1]["close"]) / _safe_float(df.iloc[-10]["close"]) - 1.0 if len(df) >= 10 and _safe_float(df.iloc[-10]["close"]) else 0.0
        amount_vs_5d = amount_latest / amount_5d_mean - 1.0 if amount_5d_mean else 0.0
        out["indices"][name] = {
            "status": "ok",
            "ts_code": code,
            "rows": rows,
            "week_return": round(week_return, 4),
            "week_return_pct": round(week_return * 100.0, 2),
            "latest_close": rows[-1]["close"] if rows else None,
            "latest_pct_chg": rows[-1]["pct_chg"] if rows else None,
            "latest_trend": rows[-1]["trend"] if rows else "",
            "ret_10d": round(ret_10d, 4),
            "ret_10d_pct": round(ret_10d * 100.0, 2),
            "ma5": round(ma5, 2),
            "ma10": round(ma10, 2),
            "ma20": round(ma20, 2),
            "close_vs_ma20": round(_safe_float(latest.get("close")) / ma20 - 1.0, 4) if ma20 else 0.0,
            "close_vs_ma20_pct": round((_safe_float(latest.get("close")) / ma20 - 1.0) * 100.0, 2) if ma20 else 0.0,
            "amount_vs_5d": round(amount_vs_5d, 4),
            "amount_vs_5d_pct": round(amount_vs_5d * 100.0, 2),
        }
    return out


def _load_market_snapshot(data_path: str, date_key: str) -> dict[str, Any]:
    try:
        import pandas as pd
    except Exception as exc:
        return {"status": "unavailable", "reason": f"pandas_unavailable:{exc}"}
    path = Path(data_path)
    if not path.exists() or not date_key:
        return {"status": "unavailable", "reason": "missing_data_path_or_date"}

    target_date = f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:]}"
    dates: set[str] = set()
    try:
        for chunk in pd.read_csv(path, usecols=["date"], chunksize=600_000):
            dates.update(str(value)[:10] for value in chunk["date"].dropna().unique())
    except Exception as exc:
        return {"status": "unavailable", "reason": f"date_scan_failed:{exc}"}
    all_trade_dates = sorted(value for value in dates if value <= target_date)
    prev_dates = [value for value in all_trade_dates if value < target_date]
    if not prev_dates:
        return {"status": "unavailable", "reason": "missing_previous_trading_date", "target_date": target_date}
    prev_date = prev_dates[-1]
    lookback_dates = set(all_trade_dates[-20:])

    usecols = ["date", "code", "open", "high", "low", "close", "amount", "industry"]
    rows = []
    try:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=300_000):
            chunk["date"] = chunk["date"].astype(str).str[:10]
            rows.append(chunk.loc[chunk["date"].isin(lookback_dates)].copy())
    except Exception as exc:
        return {"status": "unavailable", "reason": f"market_scan_failed:{exc}", "target_date": target_date}
    if not rows:
        return {"status": "unavailable", "reason": "no_rows", "target_date": target_date}
    frame = pd.concat(rows, ignore_index=True)
    today = frame.loc[frame["date"] == target_date].copy()
    prev = frame.loc[frame["date"] == prev_date, ["code", "close"]].rename(columns={"close": "prev_close"})
    if today.empty or prev.empty:
        return {"status": "unavailable", "reason": "missing_target_or_previous_rows", "target_date": target_date, "previous_date": prev_date}
    merged = today.merge(prev, on="code", how="left")
    for col in ["close", "open", "high", "low", "amount", "prev_close"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.loc[(merged["prev_close"] > 0) & merged["close"].notna()].copy()
    if merged.empty:
        return {"status": "unavailable", "reason": "no_valid_return_rows", "target_date": target_date, "previous_date": prev_date}
    merged["ret_1d"] = merged["close"] / merged["prev_close"] - 1.0
    merged["intraday_ret"] = merged["close"] / merged["open"].where(merged["open"] > 0) - 1.0
    close_wide = frame.pivot_table(index="code", columns="date", values="close", aggfunc="last")
    ma20 = close_wide.mean(axis=1)
    latest_close = close_wide[target_date] if target_date in close_wide.columns else None
    above_ma20_ratio = float((latest_close > ma20).mean()) if latest_close is not None and not ma20.empty else 0.0
    up_count = int((merged["ret_1d"] > 0).sum())
    down_count = int((merged["ret_1d"] < 0).sum())
    flat_count = int((merged["ret_1d"] == 0).sum())
    limit_up_count = int((merged["ret_1d"] >= 0.098).sum())
    limit_down_count = int((merged["ret_1d"] <= -0.098).sum())
    strong_up_count = int((merged["ret_1d"] >= 0.05).sum())
    strong_down_count = int((merged["ret_1d"] <= -0.05).sum())
    industry = (
        merged.groupby("industry", dropna=False)
        .agg(avg_ret=("ret_1d", "mean"), up_ratio=("ret_1d", lambda s: float((s > 0).mean())), count=("code", "count"))
        .reset_index()
    )
    industry = industry.loc[industry["count"] >= 5].sort_values("avg_ret", ascending=False)
    top_industries = [
        {
            "industry": str(row["industry"]),
            "avg_ret": round(float(row["avg_ret"]), 4),
            "up_ratio": round(float(row["up_ratio"]), 4),
            "count": int(row["count"]),
        }
        for row in industry.head(5).to_dict("records")
    ]
    weak_industries = [
        {
            "industry": str(row["industry"]),
            "avg_ret": round(float(row["avg_ret"]), 4),
            "up_ratio": round(float(row["up_ratio"]), 4),
            "count": int(row["count"]),
        }
        for row in industry.tail(5).sort_values("avg_ret").to_dict("records")
    ]
    return {
        "status": "ok",
        "target_date": target_date,
        "previous_date": prev_date,
        "stock_count": int(len(merged)),
        "up_count": up_count,
        "down_count": down_count,
        "up_count_rounded_100": int(round(up_count / 100.0) * 100),
        "down_count_rounded_100": int(round(down_count / 100.0) * 100),
        "flat_count": flat_count,
        "limit_up_count": limit_up_count,
        "limit_down_count": limit_down_count,
        "strong_up_count": strong_up_count,
        "strong_down_count": strong_down_count,
        "up_ratio": round(up_count / max(1, len(merged)), 4),
        "up_ratio_pct": round(up_count / max(1, len(merged)) * 100.0, 2),
        "up_ratio_pct_int": int(round(up_count / max(1, len(merged)) * 100.0, 0)),
        "median_return": round(float(merged["ret_1d"].median()), 4),
        "median_return_pct": round(float(merged["ret_1d"].median()) * 100.0, 2),
        "average_return": round(float(merged["ret_1d"].mean()), 4),
        "average_return_pct": round(float(merged["ret_1d"].mean()) * 100.0, 2),
        "above_ma20_ratio": round(above_ma20_ratio, 4),
        "above_ma20_ratio_pct": round(above_ma20_ratio * 100.0, 2),
        "amount_yi": round(float(merged["amount"].sum()) / 100000000.0, 2),
        "amount_wan_yi": round(float(merged["amount"].sum()) / 1000000000000.0, 2),
        "top_industries": top_industries,
        "weak_industries": weak_industries,
    }


def _load_sector_rotation(data_path: str, date_key: str, *, lookback: int = 5) -> dict[str, Any]:
    try:
        import pandas as pd
    except Exception as exc:
        return {"status": "unavailable", "reason": f"pandas_unavailable:{exc}"}
    path = Path(data_path)
    if not path.exists() or not date_key:
        return {"status": "unavailable", "reason": "missing_data_path_or_date"}
    target_date = f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:]}"
    try:
        dates: set[str] = set()
        for chunk in pd.read_csv(path, usecols=["date"], chunksize=600_000):
            dates.update(str(value)[:10] for value in chunk["date"].dropna().unique())
        trade_dates = sorted(value for value in dates if value <= target_date)[-lookback:]
        if len(trade_dates) < 2:
            return {"status": "unavailable", "reason": "not_enough_trade_dates"}
        pieces = []
        for chunk in pd.read_csv(path, usecols=["date", "code", "close", "amount", "industry"], chunksize=300_000):
            chunk["date"] = chunk["date"].astype(str).str[:10]
            pieces.append(chunk.loc[chunk["date"].isin(trade_dates)].copy())
        frame = pd.concat(pieces, ignore_index=True)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"sector_rotation_failed:{exc}"}
    if frame.empty:
        return {"status": "unavailable", "reason": "empty_frame"}
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
    first = frame.loc[frame["date"] == trade_dates[0], ["code", "close"]].rename(columns={"close": "first_close"})
    last = frame.loc[frame["date"] == trade_dates[-1], ["code", "close", "industry", "amount"]].rename(columns={"close": "last_close"})
    merged = last.merge(first, on="code", how="inner")
    merged = merged.loc[(merged["first_close"] > 0) & merged["last_close"].notna()].copy()
    if merged.empty:
        return {"status": "unavailable", "reason": "empty_merged_frame"}
    merged["ret"] = merged["last_close"] / merged["first_close"] - 1.0
    grouped = (
        merged.groupby("industry", dropna=False)
        .agg(avg_ret=("ret", "mean"), median_ret=("ret", "median"), up_ratio=("ret", lambda s: float((s > 0).mean())), amount_yi=("amount", lambda s: float(s.sum()) / 100000000.0), count=("code", "count"))
        .reset_index()
    )
    grouped = grouped.loc[grouped["count"] >= 5].copy()
    if grouped.empty:
        return {"status": "unavailable", "reason": "no_industry_with_enough_members"}
    total_amount_yi = float(grouped["amount_yi"].sum())
    grouped = grouped.sort_values("avg_ret", ascending=False)
    def _rows(df):
        return [
            {
                "industry": str(row["industry"]),
                "avg_ret": round(float(row["avg_ret"]), 4),
                "avg_ret_pct": round(float(row["avg_ret"]) * 100.0, 2),
                "median_ret": round(float(row["median_ret"]), 4),
                "median_ret_pct": round(float(row["median_ret"]) * 100.0, 2),
                "up_ratio": round(float(row["up_ratio"]), 4),
                "up_ratio_pct": round(float(row["up_ratio"]) * 100.0, 2),
                "amount_yi": round(float(row["amount_yi"]), 2),
                "amount_share_pct": round(float(row["amount_yi"]) / total_amount_yi * 100.0, 2) if total_amount_yi else 0.0,
                "count": int(row["count"]),
            }
            for row in df.to_dict("records")
        ]
    top5_amount_share_pct = round(float(grouped.head(5)["amount_yi"].sum()) / total_amount_yi * 100.0, 2) if total_amount_yi else 0.0
    return {
        "status": "ok",
        "start_date": trade_dates[0],
        "end_date": trade_dates[-1],
        "industry_count": int(len(grouped)),
        "total_amount_yi": round(total_amount_yi, 2),
        "top5_amount_share_pct": top5_amount_share_pct,
        "top_industries": _rows(grouped.head(5)),
        "weak_industries": _rows(grouped.tail(5).sort_values("avg_ret")),
    }


def _load_tushare_limit_stats(date_key: str) -> dict[str, Any]:
    tushare_limit_stats: dict[str, Any] = {"status": "unavailable", "reason": "not_queried"}
    try:
        import tushare as ts

        token = os.environ.get("TUSHARE_TOKEN", "")
        if token:
            pro = ts.pro_api(token)
            limit_df = pro.limit_list_d(trade_date=date_key)
            if limit_df is not None and not limit_df.empty:
                status_col = "limit" if "limit" in limit_df.columns else "limit_type" if "limit_type" in limit_df.columns else ""
                up_df = limit_df
                down_df = limit_df.iloc[0:0]
                if status_col:
                    status_text = limit_df[status_col].astype(str)
                    up_df = limit_df.loc[status_text.str.contains("U|涨", regex=True, na=False)]
                    down_df = limit_df.loc[status_text.str.contains("D|跌", regex=True, na=False)]
                industry_col = "industry" if "industry" in limit_df.columns else ""
                hot_industries = []
                if industry_col:
                    hot_industries = [
                        {"industry": str(k), "count": int(v)}
                        for k, v in up_df[industry_col].fillna("Unknown").value_counts().head(5).items()
                    ]
                tushare_limit_stats = {
                    "status": "ok",
                    "rows": int(len(limit_df)),
                    "limit_up_count": int(len(up_df)),
                    "limit_down_count": int(len(down_df)),
                    "hot_industries": hot_industries,
                }
            else:
                tushare_limit_stats = {"status": "unavailable", "reason": "empty_limit_list_d"}
    except Exception as exc:
        tushare_limit_stats = {"status": "unavailable", "reason": str(exc)[:160]}
    return tushare_limit_stats


def _load_news_context(date_key: str) -> dict[str, Any]:
    path = ROOT / "outputs" / "morning_brief" / f"morning_brief_{date_key}.json"
    tushare_limit_stats = _load_tushare_limit_stats(date_key)
    if not path.exists():
        return {
            "status": "ok",
            "morning_brief_status": "missing",
            "title": "",
            "focus_sectors": [],
            "focus_stocks": [],
            "macro_items": [],
            "mx_data_text": "",
            "tushare_limit_text": "",
            "tushare_report_text": "",
            "tushare_limit_stats": tushare_limit_stats,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "ok",
            "morning_brief_status": f"read_failed:{exc}",
            "title": "",
            "focus_sectors": [],
            "focus_stocks": [],
            "macro_items": [],
            "mx_data_text": "",
            "tushare_limit_text": "",
            "tushare_report_text": "",
            "tushare_limit_stats": tushare_limit_stats,
        }
    macro_items = []
    for item in payload.get("macro_items", [])[:4]:
        macro_items.append(
            {
                "title": str(item.get("title") or "")[:80],
                "date": str(item.get("date") or "")[:19],
                "type": str(item.get("type") or ""),
                "content": str(item.get("content") or "")[:180],
            }
        )
    return {
        "status": "ok",
        "morning_brief_status": "ok",
        "title": payload.get("title", ""),
        "focus_sectors": payload.get("focus_sectors", []),
        "focus_stocks": payload.get("focus_stocks", []),
        "macro_items": macro_items,
        "mx_data_text": str(payload.get("mx_data_text") or "")[:600],
        "tushare_limit_text": str(payload.get("tushare_limit_text") or "")[:240],
        "tushare_report_text": str(payload.get("tushare_report_text") or "")[:240],
        "tushare_limit_stats": tushare_limit_stats,
    }


def _first_available(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _valuation_tag(pe_ttm: Any, pb: Any) -> str:
    pe = _safe_float(pe_ttm, -1)
    pb_value = _safe_float(pb, -1)
    if pe > 80 or pb_value > 8:
        return "估值偏高"
    if 0 < pe < 15 and 0 < pb_value < 2:
        return "估值偏低"
    if pe > 0 or pb_value > 0:
        return "估值中性"
    return "估值缺失"


def _fundamental_tag(financial: dict[str, Any], valuation: dict[str, Any]) -> str:
    roe = _safe_float(financial.get("roe_dt"), 0)
    revenue_yoy = _safe_float(financial.get("revenue_yoy"), 0)
    profit_yoy = _safe_float(financial.get("netprofit_yoy"), 0)
    debt = _safe_float(financial.get("debt_to_assets"), 0)
    if roe > 10 and (revenue_yoy > 0 or profit_yoy > 0) and (debt <= 70 or debt == 0):
        return "基本面偏强"
    if debt > 75 or (revenue_yoy < -20 and profit_yoy < -20):
        return "基本面承压"
    if valuation.get("valuation_tag") == "估值偏高" and roe < 8:
        return "高估值待验证"
    return "基本面中性"


def _load_fundamental_context(date_key: str, facts: dict[str, Any]) -> dict[str, Any]:
    try:
        import pandas as pd
        import tushare as ts
    except Exception as exc:
        return {"status": "unavailable", "reason": f"dependency_unavailable:{exc}", "stocks": {}}
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        return {"status": "unavailable", "reason": "missing_tushare_token", "stocks": {}}
    codes: dict[str, dict[str, Any]] = {}
    for strategy_name, rows in _strategy_rows(facts).items():
        for row in rows:
            code = str(row.get("code") or "").strip()
            if not code:
                continue
            item = codes.setdefault(code, {"code": code, "name": row.get("name") or code, "strategies": []})
            item["strategies"].append(strategy_name)
            if row.get("industry") and not item.get("industry"):
                item["industry"] = row.get("industry")
    if not codes:
        return {"status": "unavailable", "reason": "empty_recommendation_codes", "stocks": {}}

    pro = ts.pro_api(token)
    start_date = f"{max(2000, int(date_key[:4]) - 2)}0101" if len(date_key) >= 4 else "20240101"
    stocks: dict[str, Any] = {}
    for code, meta in codes.items():
        valuation: dict[str, Any] = {}
        financial: dict[str, Any] = {}
        errors: list[str] = []
        try:
            basic = pro.daily_basic(ts_code=code, start_date=start_date, end_date=date_key)
            if basic is not None and not basic.empty:
                basic = basic.sort_values("trade_date").tail(1)
                row = basic.iloc[0].to_dict()
                valuation = {
                    "trade_date": str(row.get("trade_date") or ""),
                    "pe_ttm": _first_available(row, ["pe_ttm", "pe"]),
                    "pb": row.get("pb"),
                    "ps_ttm": _first_available(row, ["ps_ttm", "ps"]),
                    "total_mv_yi": round(_safe_float(row.get("total_mv")) / 10000.0, 2) if row.get("total_mv") is not None else None,
                    "circ_mv_yi": round(_safe_float(row.get("circ_mv")) / 10000.0, 2) if row.get("circ_mv") is not None else None,
                    "turnover_rate": row.get("turnover_rate"),
                    "turnover_rate_f": row.get("turnover_rate_f"),
                    "volume_ratio": row.get("volume_ratio"),
                }
                valuation["valuation_tag"] = _valuation_tag(valuation.get("pe_ttm"), valuation.get("pb"))
            else:
                errors.append("daily_basic_empty")
        except Exception as exc:
            errors.append(f"daily_basic:{str(exc)[:120]}")
        time.sleep(0.12)
        try:
            fina = pro.fina_indicator(ts_code=code, start_date=start_date, end_date=date_key)
            if fina is not None and not fina.empty:
                sort_col = "ann_date" if "ann_date" in fina.columns else "end_date"
                fina = fina.sort_values(sort_col).tail(1)
                row = fina.iloc[0].to_dict()
                financial = {
                    "end_date": str(row.get("end_date") or ""),
                    "ann_date": str(row.get("ann_date") or ""),
                    "roe_dt": _first_available(row, ["roe_dt", "roe"]),
                    "grossprofit_margin": row.get("grossprofit_margin"),
                    "debt_to_assets": row.get("debt_to_assets"),
                    "revenue_yoy": _first_available(row, ["or_yoy", "revenue_yoy"]),
                    "netprofit_yoy": _first_available(row, ["q_netprofit_yoy", "netprofit_yoy", "profit_dedt_yoy"]),
                    "ocf_to_or": _first_available(row, ["ocf_to_or", "ocf_to_opincome", "or_ocf_to_operate_profit"]),
                }
            else:
                errors.append("fina_indicator_empty")
        except Exception as exc:
            errors.append(f"fina_indicator:{str(exc)[:120]}")
        time.sleep(0.12)
        tag = _fundamental_tag(financial, valuation)
        stocks[code] = {
            **meta,
            "valuation": valuation,
            "financial": financial,
            "fundamental_tag": tag,
            "errors": errors,
        }
    return {"status": "ok", "stocks": stocks}


def _stock_fundamental_line(stock: dict[str, Any] | None) -> str:
    if not stock:
        return "基本面数据暂缺"
    valuation = stock.get("valuation", {}) or {}
    financial = stock.get("financial", {}) or {}
    parts = [str(stock.get("fundamental_tag") or "基本面待定")]
    if not _is_missing(valuation.get("pe_ttm")):
        parts.append(f"PE(TTM)={_safe_float(valuation.get('pe_ttm')):.2f}")
    if not _is_missing(valuation.get("pb")):
        parts.append(f"PB={_safe_float(valuation.get('pb')):.2f}")
    if not _is_missing(financial.get("roe_dt")):
        parts.append(f"ROE={_safe_float(financial.get('roe_dt')):.2f}")
    if not _is_missing(financial.get("revenue_yoy")):
        parts.append(f"营收同比={_safe_float(financial.get('revenue_yoy')):.2f}%")
    if not _is_missing(financial.get("netprofit_yoy")):
        parts.append(f"净利同比={_safe_float(financial.get('netprofit_yoy')):.2f}%")
    return "；".join(parts)


def _quant_reason(row: dict[str, Any], strategy_name: str) -> str:
    if strategy_name == "短线机会":
        return (
            f"短线分={_fmt_score(row.get('shortline_score'))}；"
            f"风险={row.get('risk_level') or '暂缺'}；"
            f"{row.get('operation_logic') or '涨停/连板情绪候选'}"
        )
    return (
        f"分数={_fmt_score(row.get('score'))}；"
        f"{row.get('ml_observation_tag') or 'ML观察暂缺'}；"
        f"{row.get('industry_leader_follow_tag') or '龙头扩散暂缺'}；"
        f"{row.get('overhead_density_tag') or '兑现压力暂缺'}"
    )


def _operation_logic(row: dict[str, Any], strategy_name: str, risk: dict[str, Any]) -> str:
    risk_status = str(risk.get("status") or "")
    if strategy_name == "短线机会":
        return f"只适合盘中确认强度后轻仓试错；参考买入区间：{row.get('buy_range') or '暂缺'}，止损：{row.get('stop_loss') or '暂缺'}。"
    if "WARN" in risk_status:
        return "风控为 WARN，优先观察不追高；只在放量承接、回撤可控时小仓位跟踪。"
    if row.get("overhead_density_tag") == "兑现压力轻":
        return "兑现压力轻，若次日量价延续可作为候选跟踪；跌破短线承接则放弃。"
    return "作为候选池观察票处理，等待趋势、资金和板块共振进一步确认。"


def _tiny_prompt_facts(facts: dict[str, Any]) -> dict[str, Any]:
    main_rows = facts.get("main_strategy", {}).get("top_rows", [])[:5]
    elastic_rows = facts.get("elastic_pool", {}).get("top_rows", [])[:3]
    short_rows = facts.get("shortline", {}).get("cards", [])[:3]
    return {
        "date": facts.get("selection_date"),
        "index_review": facts.get("index_review", {}),
        "market": facts.get("market_snapshot", {}),
        "sector_rotation": facts.get("sector_rotation", {}),
        "news": facts.get("news_context", {}),
        "fundamentals": facts.get("fundamental_context", {}),
        "main": [
            {
                "rank": row.get("rank"),
                "name": row.get("name"),
                "score": row.get("score"),
                "ml": row.get("ml_observation_tag"),
                "leader": row.get("industry_leader_follow_tag"),
                "overhead": row.get("overhead_density_tag"),
                "fundamental": (facts.get("fundamental_context", {}).get("stocks", {}) or {}).get(str(row.get("code") or ""), {}),
            }
            for row in main_rows
        ],
        "elastic": [
            {
                "rank": row.get("rank"),
                "name": row.get("name"),
                "score": row.get("score"),
                "fundamental": (facts.get("fundamental_context", {}).get("stocks", {}) or {}).get(str(row.get("code") or ""), {}),
            }
            for row in elastic_rows
        ],
        "shortline": [
            {
                "rank": row.get("rank"),
                "name": row.get("name"),
                "score": row.get("shortline_score"),
                "risk": row.get("risk_level"),
                "fundamental": (facts.get("fundamental_context", {}).get("stocks", {}) or {}).get(str(row.get("code") or ""), {}),
            }
            for row in short_rows
        ],
        "risk": {
            "status": facts.get("risk", {}).get("status"),
            "action_hint": facts.get("risk", {}).get("action_hint"),
            "max_drawdown": facts.get("risk", {}).get("max_drawdown"),
            "max_drawdown_pct": facts.get("risk", {}).get("max_drawdown_pct"),
        },
        "quality": {
            "headline": facts.get("candidate_pool_quality", {}).get("headline"),
            "keep_frozen": facts.get("candidate_pool_quality", {}).get("keep_frozen"),
        },
    }


def _add_derived_numeric_facts(facts: dict[str, Any]) -> None:
    risk = facts.get("risk")
    if isinstance(risk, dict) and risk.get("max_drawdown") is not None:
        risk["max_drawdown_pct"] = round(_safe_float(risk.get("max_drawdown")) * 100.0, 2)
    for container_key in ("sector_rotation", "market_snapshot"):
        container = facts.get(container_key)
        if not isinstance(container, dict):
            continue
        for list_key in ("top_industries", "weak_industries"):
            rows = container.get(list_key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for key in ("avg_ret", "median_ret", "up_ratio"):
                    if key in row and f"{key}_pct" not in row:
                        row[f"{key}_pct"] = round(_safe_float(row.get(key)) * 100.0, 2)


def _render_local_markdown(facts: dict[str, Any], *, llm_text: str, render_mode: str, fallback_reason: str) -> str:
    main_rows = facts.get("main_strategy", {}).get("top_rows", [])[:5]
    elastic_rows = facts.get("elastic_pool", {}).get("top_rows", [])[:3]
    short_rows = facts.get("shortline", {}).get("cards", [])[:3]
    risk = facts.get("risk", {})
    index_review = facts.get("index_review", {})
    market = facts.get("market_snapshot", {})
    sector_rotation = facts.get("sector_rotation", {})
    news_context = facts.get("news_context", {})
    fundamentals = facts.get("fundamental_context", {}).get("stocks", {}) or {}
    lines = [
        f"# LLM 三合一量化日报试发 {facts.get('selection_date') or facts.get('date_key')}",
        "",
        f"- 渲染状态：`{render_mode}`",
        f"- 日期硬校验：`{facts.get('validation', {}).get('date_consistent')}`",
    ]
    if fallback_reason:
        lines.append(f"- 降级原因：`{fallback_reason}`")
    lines.extend(["", "## 大模型分析", llm_text.strip() or "LLM 暂不可用，已使用模板摘要。", "", "## 大盘指数复盘"])
    if index_review.get("status") == "ok":
        for name, item in (index_review.get("indices") or {}).items():
            if item.get("status") != "ok":
                lines.append(f"- {name}：暂不可用 `{item.get('reason')}`")
                continue
            lines.append(f"### {name}")
            lines.append("| 日期 | 收盘 | 涨跌% | 成交额 | 趋势 |")
            lines.append("|---|---:|---:|---:|---|")
            for row in item.get("rows", []):
                lines.append(f"| {row.get('date')} | {row.get('close')} | {row.get('pct_chg_pct'):+.2f}% | {row.get('amount_yi')}亿 | {row.get('trend')} |")
            lines.append(f"- 本周累计：`{item.get('week_return_pct'):+.2f}%`；最新趋势：`{item.get('latest_trend')}`")
    else:
        lines.append(f"- 指数复盘暂不可用：`{index_review.get('reason') or 'unknown'}`")
    lines.extend(["", "## 市场宽度"])
    if market.get("status") == "ok":
        lines.extend(
            [
                f"- 上涨 `{market.get('up_count')}` 家，下跌 `{market.get('down_count')}` 家，上涨占比 `{market.get('up_ratio')}`",
                f"- 中位涨跌幅 `{market.get('median_return')}`，平均涨跌幅 `{market.get('average_return')}`，成交额 `{market.get('amount_yi')}` 亿",
                "- 强行业：" + " / ".join(f"{row.get('industry')}({_fmt_score(row.get('avg_ret'))})" for row in market.get("top_industries", [])[:3]),
            ]
        )
    else:
        lines.append(f"- 大盘快照暂不可用：`{market.get('reason') or 'unknown'}`")
    lines.extend(["", "## 主策略前排"])
    for row in main_rows:
        stock = fundamentals.get(str(row.get("code") or ""))
        lines.append(
            f"- #{row.get('rank')} {row.get('name')} `{row.get('code')}` 分数 `{_fmt_score(row.get('score'))}`；"
            f"{row.get('ml_observation_tag') or 'ML暂缺'}；{row.get('overhead_density_tag') or '兑现压力暂缺'}"
        )
        lines.append(f"  - 基本面：{_stock_fundamental_line(stock)}")
        lines.append(f"  - 量化理由：{_quant_reason(row, '主策略')}")
        lines.append(f"  - 操作逻辑：{_operation_logic(row, '主策略', risk)}")
    lines.extend(["", "## 弹性池"])
    for row in elastic_rows:
        stock = fundamentals.get(str(row.get("code") or ""))
        lines.append(f"- #{row.get('rank')} {row.get('name')} `{row.get('code')}` 分数 `{_fmt_score(row.get('score'))}`")
        lines.append(f"  - 基本面：{_stock_fundamental_line(stock)}")
        lines.append(f"  - 量化理由：{_quant_reason(row, '弹性池')}")
        lines.append(f"  - 操作逻辑：{_operation_logic(row, '弹性池', risk)}")
    lines.extend(["", "## 精选短线机会"])
    for row in short_rows:
        stock = fundamentals.get(str(row.get("code") or ""))
        lines.append(f"- #{row.get('rank')} {row.get('name')} `{row.get('code')}` 短线分 `{_fmt_score(row.get('shortline_score'))}`；风险 `{row.get('risk_level') or '暂缺'}`")
        lines.append(f"  - 基本面：{_stock_fundamental_line(stock)}")
        lines.append(f"  - 量化理由：{_quant_reason(row, '短线机会')}")
        lines.append(f"  - 操作逻辑：{_operation_logic(row, '短线机会', risk)}")
    lines.extend(
        [
            "",
            "## 本周板块轮动",
        ]
    )
    if sector_rotation.get("status") == "ok":
        lines.append(f"- 区间：`{sector_rotation.get('start_date')}` ~ `{sector_rotation.get('end_date')}`")
        lines.append("- 强板块：" + " / ".join(f"{row.get('industry')}({_pct(row.get('avg_ret'))})" for row in sector_rotation.get("top_industries", [])[:5]))
        lines.append("- 退潮板块：" + " / ".join(f"{row.get('industry')}({_pct(row.get('avg_ret'))})" for row in sector_rotation.get("weak_industries", [])[:5]))
    else:
        lines.append(f"- 板块轮动暂不可用：`{sector_rotation.get('reason') or 'unknown'}`")
    lines.extend(
        [
            "",
            "## 今日消息面",
        ]
    )
    if news_context.get("status") == "ok":
        for item in news_context.get("macro_items", [])[:3]:
            lines.append(f"- {item.get('title')}：{item.get('content')}")
        if news_context.get("tushare_limit_text"):
            lines.append(f"- 涨停情绪：{news_context.get('tushare_limit_text')}")
    else:
        lines.append(f"- 消息面暂不可用：`{news_context.get('reason') or 'unknown'}`")
    lines.extend(
        [
            "",
            "## 风控与候选池质量",
            f"- 风控：`{risk.get('status')}`；建议：`{risk.get('action_hint')}`；最大回撤：`{risk.get('max_drawdown')}`",
            f"- 看板：{facts.get('candidate_pool_quality', {}).get('headline') or '暂无候选池质量摘要'}",
        ]
    )
    return "\n".join(lines)


def _render_llm_text(facts: dict[str, Any], args: argparse.Namespace) -> tuple[str, str, str]:
    prompt_facts = _tiny_prompt_facts(facts)
    settings = resolve_llm_settings()
    try:
        content = call_openai_compatible_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是A股量化日报助手。只能基于JSON事实，输出中文分析。"
                        "请按固定小标题输出：大盘指数复盘、市场宽度、主策略意见、弹性策略意见、短线机会意见、本周板块轮动、今日消息面、日报总结。"
                        "每部分给出1到3句判断，控制在1200字以内。必须分别给出三大策略的意见，并结合个股基本面、量化理由和操作逻辑。"
                        "大模型分析区禁止输出任何阿拉伯数字、百分号、排名编号或具体数值；具体数字由下方模板表格展示。"
                        "可以写股票名、板块名和方向判断，但不要写新的数值。"
                        "如果news.status为ok，必须基于macro_items、tushare_limit_text、tushare_report_text总结消息面，禁止说消息面未提供。"
                        "大模型分析区禁止写任何具体股票名称，只能做大盘、板块和策略层判断；"
                        "每只推荐股的基本面、量化理由和操作逻辑由下方结构化明细展示。"
                    ),
                },
                {"role": "user", "content": compact_json(prompt_facts, limit=2500)},
            ],
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            model=settings["model"],
            timeout_seconds=args.llm_timeout_seconds,
            max_tokens=args.llm_max_tokens,
        )
        ungrounded = validate_numbers_grounded_in_json(content, facts)
        if ungrounded:
            return "LLM 输出包含未在结构化事实中登记的数字，已按硬校验规则降级。", "fallback_ungrounded_numbers", ",".join(ungrounded[:8])
        return content, "llm", ""
    except Exception as exc:
        return render_deterministic_report(facts), "fallback", str(exc)


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _call_text_agent(
    *,
    name: str,
    system_prompt: str,
    payload: dict[str, Any],
    args: argparse.Namespace,
    max_tokens: int,
) -> dict[str, Any]:
    settings = resolve_llm_settings()
    try:
        content = call_openai_compatible_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": compact_json(payload, limit=5000)},
            ],
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            model=settings["model"],
            timeout_seconds=args.llm_timeout_seconds,
            max_tokens=max_tokens,
        )
        return {"status": "ok", "agent": name, "content": content.strip()}
    except Exception as exc:
        return {"status": "fallback", "agent": name, "error": str(exc)[:300], "content": ""}


def _call_stock_agent(
    *,
    strategy_name: str,
    row: dict[str, Any],
    facts: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    code = str(row.get("code") or "")
    stock = (facts.get("fundamental_context", {}).get("stocks", {}) or {}).get(code, {})
    missing_inputs = _missing_required_stock_inputs(strategy_name, row, stock)
    if missing_inputs:
        reason = "缺少关键量化或基本面字段：" + "、".join(missing_inputs[:8])
        return {
            "status": "fallback_missing_required_inputs",
            "ai_fundamental": f"暂不可判定，{reason}。",
            "ai_quant_reason": f"暂不可判定，{reason}。",
            "ai_operation_logic": f"暂不可判定，{reason}。",
            "error": reason,
        }
    payload = {
        "strategy": strategy_name,
        "stock": {
            "rank": row.get("rank"),
            "code": code,
            "name": row.get("name"),
            "industry": row.get("industry"),
            "close_price": row.get("close"),
        },
        "quant_facts": {
            "score": row.get("score"),
            "ml_score": row.get("ml_score"),
            "ml_rank": row.get("ml_rank"),
            "shortline_score": row.get("shortline_score"),
            "ml_observation_tag": row.get("ml_observation_tag"),
            "industry_leader_follow_tag": row.get("industry_leader_follow_tag"),
            "industry_leader_follow_score": row.get("industry_leader_follow_score"),
            "industry_leader_follow_rank": row.get("industry_leader_follow_rank"),
            "overhead_density_tag": row.get("overhead_density_tag"),
            "overhead_density_score": row.get("overhead_density_score"),
            "overhead_density_rank": row.get("overhead_density_rank"),
            "market_cap": row.get("market_cap"),
            "amount": row.get("amount"),
            "risk_level": row.get("risk_level"),
            "current_streak": row.get("current_streak"),
            "limit_up_count_20d": row.get("limit_up_count_20d"),
            "open_board_count_20d": row.get("open_board_count_20d"),
            "turnover_ratio": row.get("turnover_ratio"),
            "fd_amount": row.get("fd_amount"),
            "recent_4d_path": row.get("recent_4d_path"),
            "operation_logic": row.get("operation_logic"),
            "buy_range": row.get("buy_range"),
            "stop_loss": row.get("stop_loss"),
        },
        "fundamental_facts": stock,
        "market_context": {
            "risk": facts.get("risk", {}),
            "market_snapshot": facts.get("market_snapshot", {}),
            "sector_rotation": facts.get("sector_rotation", {}),
        },
    }
    result = _call_text_agent(
        name=f"stock_agent:{strategy_name}:{code}",
        system_prompt=(
            "你是A股资深金融分析师，但大模型只做解释层，不做选股决策层。只基于JSON事实，输出严格JSON对象，不要markdown。"
            "字段必须为 ai_fundamental、ai_quant_reason、ai_operation_logic。"
            "三段都要由你分析生成，不能照抄输入原句。"
            "不要输出任何阿拉伯数字、百分号或新的具体数值；数字由模板区展示。"
            "如果JSON缺少判断所需字段，必须写“暂不可判定”，禁止脑补。"
            "只能分析当前stock里的这一只股票，禁止提及其他股票。"
            "ai_fundamental分析公司基本面质量、估值压力和财务亮点或隐患；"
            "ai_quant_reason解释为什么量化系统会选中它；"
            "ai_operation_logic给出观察、进出、仓位和风控动作。"
        ),
        payload=payload,
        args=args,
        max_tokens=args.stock_agent_max_tokens,
    )
    if result["status"] != "ok":
        return _fallback_stock_agent(
            status="fallback",
            row=row,
            strategy_name=strategy_name,
            stock=stock,
            risk=facts.get("risk", {}),
            error=result.get("error", ""),
        )
    try:
        parsed = json.loads(_strip_json_fence(result["content"]))
    except Exception as exc:
        return _fallback_stock_agent(
            status="fallback_parse",
            row=row,
            strategy_name=strategy_name,
            stock=stock,
            risk=facts.get("risk", {}),
            error=str(exc)[:200],
            raw=result["content"],
        )
    parsed_agent = {
        "status": "ok",
        "ai_fundamental": str(parsed.get("ai_fundamental") or "").strip(),
        "ai_quant_reason": str(parsed.get("ai_quant_reason") or "").strip(),
        "ai_operation_logic": str(parsed.get("ai_operation_logic") or "").strip(),
    }
    missing_fundamental = _missing_fundamental_inputs(stock)
    if missing_fundamental:
        parsed_agent["ai_fundamental"] = "暂不可判定，缺少关键 Tushare 基本面字段：" + "、".join(missing_fundamental[:8]) + "。"
    if any(not parsed_agent[key] for key in ("ai_fundamental", "ai_quant_reason", "ai_operation_logic")):
        return _fallback_stock_agent(
            status="fallback_empty_fields",
            row=row,
            strategy_name=strategy_name,
            stock=stock,
            risk=facts.get("risk", {}),
            error="empty_required_ai_fields",
            raw=result["content"],
        )
    return parsed_agent


def _run_multi_agent_analysis(facts: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    date_key = str(facts.get("date_key") or "").replace("-", "")
    allowed_codes = _allowed_stock_codes(facts)
    allowed_names = _allowed_stock_names(facts)
    master_names = _latest_master_name_map(args.master_data_path, date_key)
    market_agent = _call_text_agent(
        name="market_agent",
        system_prompt=(
            "你是A股市场复盘分析师。大模型只做解释层，不做选股决策层。只基于JSON事实，输出中文HTML片段，不要完整html文档。"
            "必须包含大盘指数、市场宽度、本周板块轮动、消息面、日报总结。"
            "禁止输出任何具体股票名和股票代码；可以引用JSON中已有数字，禁止编造新数字。"
        ),
        payload={
            "date": facts.get("selection_date"),
            "index_review": facts.get("index_review", {}),
            "market_snapshot": facts.get("market_snapshot", {}),
            "sector_rotation": facts.get("sector_rotation", {}),
            "news_context": facts.get("news_context", {}),
            "risk": facts.get("risk", {}),
        },
        args=args,
        max_tokens=args.llm_max_tokens,
    )
    market_errors = _validate_agent_content(
        agent=market_agent,
        facts=facts,
        allowed_codes=set(),
        allowed_names=set(),
        master_names=master_names,
        allow_numbers=True,
    )
    if market_agent.get("status") != "ok" or market_errors:
        market_agent = _fallback_market_agent(facts, ";".join(market_errors) or str(market_agent.get("error") or "market_agent_failed"))
    strategy_agents: dict[str, Any] = {}
    stock_agents: dict[str, Any] = {}
    for strategy_name, rows in _strategy_rows(facts).items():
        if not rows:
            strategy_agents[strategy_name] = {
                "status": "ok",
                "agent": f"strategy_agent:{strategy_name}",
                "content": (
                    "今日该策略没有符合当前规则的候选标的，日报只保留日期一致性占位，"
                    "不引用旧日期股票，也不生成交易建议。"
                ),
            }
            continue
        strategy_agents[strategy_name] = _call_text_agent(
            name=f"strategy_agent:{strategy_name}",
            system_prompt=(
                "你是A股策略组合分析师。大模型只做解释层，不做选股决策层。只基于JSON事实，输出两到三句中文HTML片段。"
                "只分析该策略当前候选池质量、适合的操作方式和主要风险。"
                "禁止输出具体股票名和股票代码；个股解释由个股卡片完成。可以引用JSON中已有数字，禁止编造新数字。"
            ),
            payload={
                "strategy": strategy_name,
                "rows": rows,
                "risk": facts.get("risk", {}),
                "quality": facts.get("candidate_pool_quality", {}),
                "market": facts.get("market_snapshot", {}),
            },
            args=args,
            max_tokens=420,
        )
        strategy_errors = _validate_agent_content(
            agent=strategy_agents[strategy_name],
            facts=facts,
            allowed_codes=set(),
            allowed_names=set(),
            master_names=master_names,
            allow_numbers=True,
        )
        if strategy_agents[strategy_name].get("status") != "ok" or strategy_errors:
            strategy_agents[strategy_name] = _fallback_strategy_agent(strategy_name, rows, facts, ";".join(strategy_errors) or str(strategy_agents[strategy_name].get("error") or "strategy_agent_failed"))
        for row in rows:
            code = str(row.get("code") or "")
            if not code:
                continue
            agent = _call_stock_agent(strategy_name=strategy_name, row=row, facts=facts, args=args)
            if agent.get("status") == "ok":
                stock_errors = _validate_agent_content(
                    agent=agent,
                    facts=facts,
                    allowed_codes={code},
                    allowed_names={str(row.get("name") or "").strip()},
                    master_names=master_names,
                    allow_numbers=False,
                )
                if stock_errors:
                    stock = (facts.get("fundamental_context", {}).get("stocks", {}) or {}).get(code, {})
                    agent = _fallback_stock_agent(
                        status="fallback_guardrail",
                        row=row,
                        strategy_name=strategy_name,
                        stock=stock,
                        risk=facts.get("risk", {}),
                        error=";".join(stock_errors),
                        raw=_agent_text(agent),
                    )
            stock_agents[code] = agent
            time.sleep(0.15)
    return {
        "market_agent": market_agent,
        "strategy_agents": strategy_agents,
        "stock_agents": stock_agents,
        "guardrails": {
            "allowed_codes": sorted(allowed_codes),
            "allowed_names": sorted(allowed_names),
            "master_name_count": len(master_names),
            "numbers_policy": "market/strategy/stock agent visible text must not contain Arabic numeric tokens; template renders JSON numbers",
            "unknown_stock_policy": "agent text must not mention stocks outside the structured JSON candidate set",
        },
    }


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _chip(text: Any, color: str = "#2563eb") -> str:
    return (
        f'<span style="display:inline-block;background:{color};color:#fff;'
        'padding:3px 8px;border-radius:999px;font-size:12px;font-weight:700;margin:2px 4px 2px 0;">'
        f"{_e(text)}</span>"
    )


def _section(title: str, body: str) -> str:
    return (
        '<section style="background:#ffffff;border:1px solid #e5e7eb;border-radius:16px;'
        'padding:16px;margin:14px 0;box-shadow:0 4px 14px rgba(15,23,42,0.06);">'
        f'<h2 style="font-size:22px;margin:0 0 12px;color:#111827;">{_e(title)}</h2>'
        f"{body}</section>"
    )


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f'<th style="border:1px solid #d1d5db;padding:8px;background:#f3f4f6;text-align:left;">{_e(h)}</th>' for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f'<td style="border:1px solid #d1d5db;padding:8px;">{_e(cell)}</td>' for cell in row) + "</tr>"
    return f'<table style="border-collapse:collapse;width:100%;font-size:13px;"> <thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _index_html(index_review: dict[str, Any]) -> str:
    if index_review.get("status") != "ok":
        return f'<p style="color:#6b7280;">指数复盘暂不可用：{_e(index_review.get("reason") or "unknown")}</p>'
    parts = []
    for name, item in (index_review.get("indices") or {}).items():
        if item.get("status") != "ok":
            parts.append(f'<p>{_e(name)}：暂不可用 {_e(item.get("reason"))}</p>')
            continue
        rows = [
            [row.get("date"), row.get("close"), f"{_safe_float(row.get('pct_chg_pct')):+.2f}%", f"{row.get('amount_yi')}亿", row.get("trend")]
            for row in item.get("rows", [])
        ]
        parts.append(f'<h3 style="margin:10px 0 8px;">{_e(name)}</h3>')
        parts.append(_html_table(["日期", "收盘", "涨跌%", "成交额", "趋势"], rows))
        parts.append(
            f'<p style="font-weight:700;">本周累计：{_safe_float(item.get("week_return_pct")):+.2f}%；'
            f'10日：{_safe_float(item.get("ret_10d_pct")):+.2f}%；'
            f'收盘/MA20：{_safe_float(item.get("close_vs_ma20_pct")):+.2f}%；'
            f'成交额/5日均值：{_safe_float(item.get("amount_vs_5d_pct")):+.2f}%；'
            f'最新趋势：{_e(item.get("latest_trend"))}</p>'
        )
    return "".join(parts)


def _market_width_html(market: dict[str, Any]) -> str:
    if market.get("status") != "ok":
        return f'<p style="color:#6b7280;">市场宽度暂不可用：{_e(market.get("reason") or "unknown")}</p>'
    chips = [
        _chip(f"上涨 {market.get('up_count')} 家", "#16a34a"),
        _chip(f"下跌 {market.get('down_count')} 家", "#dc2626"),
        _chip(f"上涨占比 {market.get('up_ratio_pct')}%", "#0f766e"),
        _chip(f"涨停估算 {market.get('limit_up_count')} 家", "#ea580c"),
        _chip(f"跌停估算 {market.get('limit_down_count')} 家", "#991b1b"),
        _chip(f"站上20日均线 {market.get('above_ma20_ratio_pct')}%", "#0369a1"),
        _chip(f"成交额 {market.get('amount_yi')} 亿", "#7c3aed"),
    ]
    top = " / ".join(f"{row.get('industry')}({_pct(row.get('avg_ret'))})" for row in market.get("top_industries", [])[:5])
    weak = " / ".join(f"{row.get('industry')}({_pct(row.get('avg_ret'))})" for row in market.get("weak_industries", [])[:3])
    return (
        "".join(chips)
        + f'<p style="margin-top:10px;"><b>强行业：</b>{_e(top)}</p>'
        + f'<p><b>弱行业：</b>{_e(weak)}</p>'
        + f'<p><b>强弱结构：</b>涨幅超过5%股票 {market.get("strong_up_count")} 家，跌幅超过5%股票 {market.get("strong_down_count")} 家；中位涨跌幅 {market.get("median_return_pct"):+.2f}%。</p>'
    )


def _sector_html(sector: dict[str, Any]) -> str:
    if sector.get("status") != "ok":
        return f'<p style="color:#6b7280;">板块轮动暂不可用：{_e(sector.get("reason") or "unknown")}</p>'
    top = " / ".join(
        f"{row.get('industry')}({_pct(row.get('avg_ret'))}, 上涨占比{row.get('up_ratio_pct')}%, 成交占比{row.get('amount_share_pct')}%)"
        for row in sector.get("top_industries", [])[:5]
    )
    weak = " / ".join(
        f"{row.get('industry')}({_pct(row.get('avg_ret'))}, 上涨占比{row.get('up_ratio_pct')}%)"
        for row in sector.get("weak_industries", [])[:5]
    )
    return (
        f'<p>区间：<b>{_e(sector.get("start_date"))}</b> 至 <b>{_e(sector.get("end_date"))}</b></p>'
        f'<p><b>覆盖行业：</b>{sector.get("industry_count")} 个；区间成交额 {sector.get("total_amount_yi")} 亿；前五强势行业成交占比 {sector.get("top5_amount_share_pct")}%。</p>'
        f'<p><b>强板块：</b>{_e(top)}</p>'
        f'<p><b>退潮板块：</b>{_e(weak)}</p>'
    )


def _news_html(news: dict[str, Any]) -> str:
    if news.get("status") != "ok":
        return f'<p style="color:#6b7280;">消息面暂不可用：{_e(news.get("reason") or "unknown")}</p>'
    morning_note = ""
    if news.get("morning_brief_status") and news.get("morning_brief_status") != "ok":
        morning_note = f'<p style="color:#92400e;"><b>晨报资讯：</b>{_e(news.get("morning_brief_status"))}，本节优先展示 Tushare 定量情绪。</p>'
    items = "".join(
        f'<li style="margin:8px 0;"><b>{_e(item.get("title"))}</b><br><span style="color:#4b5563;">{_e(item.get("content"))}</span></li>'
        for item in news.get("macro_items", [])[:3]
    )
    limit_text = f'<p><b>涨停情绪：</b>{_e(news.get("tushare_limit_text"))}</p>' if news.get("tushare_limit_text") else ""
    limit_stats = news.get("tushare_limit_stats") or {}
    stats_text = ""
    if limit_stats.get("status") == "ok":
        hot = " / ".join(f"{row.get('industry')}({row.get('count')})" for row in limit_stats.get("hot_industries", [])[:5])
        stats_text = (
            f'<p><b>Tushare 涨跌停统计：</b>样本 {limit_stats.get("rows")} 条，'
            f'涨停 {limit_stats.get("limit_up_count")} 家，跌停 {limit_stats.get("limit_down_count")} 家；'
            f'涨停集中行业：{_e(hot or "暂无行业字段")}。</p>'
        )
    else:
        stats_text = f'<p><b>Tushare 涨跌停统计：</b>暂不可用：{_e(limit_stats.get("reason") or "unknown")}</p>'
    return f'{morning_note}<ul style="padding-left:18px;margin:0;">{items}</ul>{limit_text}{stats_text}'


def _stock_card_html(
    *,
    row: dict[str, Any],
    strategy_name: str,
    stock_ai: dict[str, Any],
    stock: dict[str, Any] | None,
) -> str:
    code = row.get("code") or ""
    score = row.get("shortline_score") if strategy_name == "短线机会" else row.get("score")
    industry = row.get("industry") or (stock.get("industry") if stock else "") or ""
    subtitle = f"{_e(code)} · {_e(industry)}"
    fact_line = _stock_fundamental_line(stock)
    quant_fact = _quant_reason(row, strategy_name)
    price = _fmt_price(row.get("close"))
    ai_status = str(stock_ai.get("status") or "missing")
    return (
        '<div style="border:1px solid #dbeafe;background:#f8fafc;border-radius:14px;padding:14px;margin:12px 0;">'
        f'<div style="font-size:18px;font-weight:900;color:#111827;">#{_e(row.get("rank"))} {_e(row.get("name") or code)}</div>'
        f'<div style="color:#6b7280;margin:4px 0 10px;">{subtitle}</div>'
        f'{_chip(strategy_name, "#111827")}{_chip("单价 " + price, "#0f766e")}{_chip("分数 " + _fmt_score(score), "#2563eb")}{_chip("AI " + ai_status, "#16a34a" if ai_status == "ok" else "#dc2626")}'
        f'<p><b>基本面事实：</b>{_e(fact_line)}</p>'
        f'<p><b>AI 基本面分析：</b>{_e(stock_ai.get("ai_fundamental") or "暂不可判定")}</p>'
        f'<p><b>量化事实：</b>{_e(quant_fact)}</p>'
        f'<p><b>AI 量化理由：</b>{_e(stock_ai.get("ai_quant_reason") or "暂不可判定")}</p>'
        f'<p><b>AI 操作逻辑：</b>{_e(stock_ai.get("ai_operation_logic") or "暂不可判定")}</p>'
        '</div>'
    )


def _strategy_html(strategy_name: str, rows: list[dict[str, Any]], facts: dict[str, Any], agent_outputs: dict[str, Any]) -> str:
    fundamentals = facts.get("fundamental_context", {}).get("stocks", {}) or {}
    strategy_text = (agent_outputs.get("strategy_agents", {}).get(strategy_name, {}) or {}).get("content", "")
    summary_key = "main_strategy" if strategy_name == "主策略" else "elastic_pool" if strategy_name == "弹性池" else "shortline"
    summary = facts.get(summary_key, {}).get("summary", {}) or {}
    selected_count = summary.get("selected_count")
    candidate_count = summary.get("candidate_count")
    scope_line = (
        '<p style="color:#475569;font-weight:700;">'
        f'候选池总数：{_e(candidate_count if candidate_count is not None else "暂缺")}；'
        f'入选数量：{_e(selected_count if selected_count is not None else len(rows))}；'
        f'本报展示前排：{len(rows)}。'
        '</p>'
    )
    if not rows:
        summary = facts.get("shortline", {}).get("summary", {}) if strategy_name == "短线机会" else {}
        reason = summary.get("reason") or "今日无符合当前规则的候选标的。"
        return (
            scope_line
            + f'<div style="color:#374151;line-height:1.7;">{_e(strategy_text)}</div>'
            '<div style="border:1px solid #fed7aa;background:#fff7ed;border-radius:14px;padding:14px;margin:12px 0;">'
            f'<b>{_e(strategy_name)}：</b>{_e(reason)}'
            '</div>'
        )
    cards = []
    for row in rows:
        code = str(row.get("code") or "")
        cards.append(
            _stock_card_html(
                row=row,
                strategy_name=strategy_name,
                stock_ai=(agent_outputs.get("stock_agents", {}) or {}).get(code, {}),
                stock=fundamentals.get(code),
            )
        )
    return scope_line + f'<div style="color:#374151;line-height:1.7;">{strategy_text}</div>' + "".join(cards)


def _render_html_report(facts: dict[str, Any], *, agent_outputs: dict[str, Any], render_mode: str, fallback_reason: str) -> str:
    main_rows = facts.get("main_strategy", {}).get("top_rows", [])[:5]
    elastic_rows = facts.get("elastic_pool", {}).get("top_rows", [])[:3]
    short_rows = facts.get("shortline", {}).get("cards", [])[:3]
    market_text = (agent_outputs.get("market_agent", {}) or {}).get("content", "")
    status_line = (
        _chip(f"render_mode={render_mode}", "#16a34a" if render_mode == "llm" else "#dc2626")
        + _chip(f"日期校验={facts.get('validation', {}).get('date_consistent')}", "#0f766e")
        + _chip(f"主库={facts.get('master_freshness', {}).get('status', 'unknown')}", "#0f766e")
    )
    if fallback_reason:
        status_line += _chip(f"降级={fallback_reason}", "#dc2626")
    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;background:#f3f4f6;padding:14px;color:#111827;">'
        f'<h1 style="font-size:28px;margin:8px 0;">A股量化三合一日报 { _e(facts.get("selection_date") or facts.get("date_key")) }</h1>'
        f'<div>{status_line}</div>'
        + _section("大模型综合复盘", f'<div style="line-height:1.8;color:#374151;">{market_text}</div>')
        + _section("大盘指数复盘", _index_html(facts.get("index_review", {})))
        + _section("市场宽度", _market_width_html(facts.get("market_snapshot", {})))
        + _section("主策略候选", _strategy_html("主策略", main_rows, facts, agent_outputs))
        + _section("弹性策略候选", _strategy_html("弹性池", elastic_rows, facts, agent_outputs))
        + _section("精选短线机会", _strategy_html("短线机会", short_rows, facts, agent_outputs))
        + _section("本周板块轮动", _sector_html(facts.get("sector_rotation", {})))
        + _section("今日消息面", _news_html(facts.get("news_context", {})))
        + _section("风控与候选池质量", f'<p><b>风控：</b>{_e(facts.get("risk", {}).get("status"))} · {_e(facts.get("risk", {}).get("action_hint"))}</p><p><b>候选池看板：</b>{_e(facts.get("candidate_pool_quality", {}).get("headline") or "暂无")}</p>')
        + '</div>'
    )


def _post_pushplus(*, token: str, title: str, content: str, topic: str, channel: str) -> dict[str, Any]:
    payload = {
        "token": token,
        "title": title,
        "content": content,
        "template": "html",
        "channel": channel,
    }
    if topic:
        payload["topic"] = topic
    response = httpx.post("https://www.pushplus.plus/send", json=payload, timeout=30)
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return {"raw_text": response.text}


def _split_html_for_pushplus(content: str, *, max_chars: int = 18_000) -> list[str]:
    if len(content) <= max_chars:
        return [content]
    sections = re.findall(r"<section\b.*?</section>", content, flags=re.S)
    if not sections:
        return [content[i : i + max_chars] for i in range(0, len(content), max_chars)]
    first_section = content.find(sections[0])
    prefix = content[:first_section]
    chunks: list[str] = []
    current: list[str] = []

    def build(parts: list[str], part_no: int, total_hint: str = "") -> str:
        suffix = f" / {total_hint}" if total_hint else ""
        return (
            prefix
            + f'<p style="color:#6b7280;font-weight:700;">本报告因 PushPlus 长度限制自动分卷：第 {part_no}{suffix} 部分</p>'
            + "".join(parts)
            + "</div>"
        )

    part_no = 1
    for section in sections:
        candidate = build(current + [section], part_no)
        if current and len(candidate) > max_chars:
            chunks.append(build(current, part_no))
            part_no += 1
            current = [section]
        else:
            current.append(section)
    if current:
        chunks.append(build(current, part_no))
    total = str(len(chunks))
    return [chunk.replace(f"第 {idx} 部分", f"第 {idx} / {total} 部分") for idx, chunk in enumerate(chunks, start=1)]


def _post_pushplus_split_if_needed(*, token: str, title: str, content: str, topic: str, channel: str) -> dict[str, Any]:
    chunks = _split_html_for_pushplus(content)
    if len(chunks) == 1:
        return _post_pushplus(token=token, title=title, content=content, topic=topic, channel=channel)
    results = []
    for idx, chunk in enumerate(chunks, start=1):
        part_title = f"{title}（{idx}/{len(chunks)}）"
        results.append(_post_pushplus(token=token, title=part_title, content=chunk, topic=topic, channel=channel))
        time.sleep(1.2)
    return {"split": True, "parts": len(chunks), "results": results}


def main() -> None:
    args = build_parser().parse_args()
    token = args.token or __import__("os").environ.get("PUSHPLUS_TOKEN", "")
    topic = args.topic or __import__("os").environ.get("PUSHPLUS_TOPIC", "")
    channel = args.channel or __import__("os").environ.get("PUSHPLUS_CHANNEL", "wechat")
    master_freshness = _ensure_master_fresh(args)
    context = resolve_artifacts(args)
    errors = validate_report_context(context)
    preflight_repair: dict[str, Any] = {"status": "not_needed"}
    if errors:
        preflight_repair = _run_preflight_repair(args)
        context = resolve_artifacts(args)
        errors = validate_report_context(context)
    facts = build_structured_facts(context, args)
    facts["master_freshness"] = master_freshness
    facts["preflight_repair"] = preflight_repair
    facts["index_review"] = _load_index_review(str(context.get("date_key") or ""))
    facts["market_snapshot"] = _load_market_snapshot(args.master_data_path, str(context.get("date_key") or ""))
    facts["sector_rotation"] = _load_sector_rotation(args.master_data_path, str(context.get("date_key") or ""))
    facts["news_context"] = _load_news_context(str(context.get("date_key") or ""))
    facts["fundamental_context"] = _load_fundamental_context(str(context.get("date_key") or ""), facts)
    _add_derived_numeric_facts(facts)
    if errors:
        raise RuntimeError(
            json.dumps(
                {
                    "error": "preflight_validation_failed_refuse_to_push_stale_report",
                    "validation_errors": errors,
                    "master_freshness": master_freshness,
                    "preflight_repair": preflight_repair,
                },
                ensure_ascii=False,
            )
        )
    else:
        agent_outputs = _run_multi_agent_analysis(facts, args)
        statuses = [agent_outputs.get("market_agent", {}).get("status")]
        statuses.extend((item or {}).get("status") for item in (agent_outputs.get("strategy_agents") or {}).values())
        statuses.extend((item or {}).get("status") for item in (agent_outputs.get("stock_agents") or {}).values())
        render_mode = "llm" if statuses and all(status == "ok" for status in statuses) else "partial_llm"
        fallback_reason = "" if render_mode == "llm" else "some_agents_fallback"
    html_content = _render_html_report(facts, agent_outputs=agent_outputs, render_mode=render_mode, fallback_reason=fallback_reason)
    date_key = str(context.get("date_key") or "latest")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"llm_pushplus_evening_brief_{date_key}.md"
    html_path = output_dir / f"llm_pushplus_evening_brief_{date_key}.html"
    json_path = output_dir / f"llm_pushplus_evening_brief_{date_key}.json"
    push_result: dict[str, Any] = {"dry_run": True}
    if not args.dry_run:
        if not token:
            raise ValueError("missing PUSHPLUS_TOKEN")
        push_result = _post_pushplus_split_if_needed(
            token=token,
            title=f"LLM三合一量化日报试发 {facts.get('selection_date') or date_key}",
            content=html_content,
            topic=topic,
            channel=channel,
        )
    payload = {
        "date_key": date_key,
        "render_mode": render_mode,
        "fallback_reason": fallback_reason,
        "validation_errors": errors,
        "master_freshness": master_freshness,
        "preflight_repair": preflight_repair,
        "push_result": push_result,
        "agent_outputs": agent_outputs,
        "html": html_content,
    }
    md_path.write_text(html_content, encoding="utf-8")
    html_path.write_text(html_content, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"markdown_path": str(md_path), "html_path": str(html_path), "json_path": str(json_path), "render_mode": render_mode, "push_result": push_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
