from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_common import (
    ROOT,
    call_openai_compatible_chat,
    compact_json,
    latest_master_date,
    load_json,
    match_date_path,
    payload_date_key,
    resolve_llm_settings,
    tail_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM sidecar review for daily quant ops health.")
    parser.add_argument("--master-data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--main-monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_auto"))
    parser.add_argument("--elastic-monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_under20_elastic"))
    parser.add_argument("--shortline-dir", default=str(ROOT / "outputs" / "shortline_opportunities_live"))
    parser.add_argument("--risk-dir", default=str(ROOT / "outputs" / "risk_governor"))
    parser.add_argument("--ops-log-dir", default=str(ROOT / "outputs" / "ops_logs"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "llm_ops_review"))
    parser.add_argument("--date-key", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--no-llm", action="store_true")
    return parser


def _status_from_log(text: str, *, title_keyword: str = "") -> dict[str, Any]:
    if not text:
        return {"status": "missing_log", "code": None, "title": "", "raw_hint": ""}
    codes = re.findall(r'"code"\s*:\s*(\d+)', text)
    titles = re.findall(r'"title"\s*:\s*"([^"]+)"', text)
    title = ""
    if title_keyword:
        matched = [item for item in titles if title_keyword in item]
        title = matched[-1] if matched else (titles[-1] if titles else "")
    else:
        title = titles[-1] if titles else ""
    code = int(codes[-1]) if codes else None
    failed_markers = ("Traceback", "ERROR", "SystemExit", "发送失败", "missing", "stale")
    return {
        "status": "ok" if code == 200 and not any(marker in text[-4000:] for marker in failed_markers) else "warning",
        "code": code,
        "title": title,
        "raw_hint": text[-1000:],
    }


def _preflight_status(text: str) -> dict[str, Any]:
    if not text:
        return {"status": "missing_log", "ok": False, "target_date": "", "missing_after": []}
    ok_matches = re.findall(r'"ok"\s*:\s*(true|false)', text)
    date_matches = re.findall(r'"target_date"\s*:\s*"(20\d{2}-\d{2}-\d{2})"', text)
    missing_matches = re.findall(r'"missing_after"\s*:\s*(\[[^\]]*\])', text)
    missing_after: list[Any] = []
    if missing_matches:
        try:
            missing_after = json.loads(missing_matches[-1])
        except Exception:
            missing_after = ["parse_failed"]
    ok = bool(ok_matches and ok_matches[-1] == "true")
    return {
        "status": "ok" if ok else "failed",
        "ok": ok,
        "target_date": date_matches[-1] if date_matches else "",
        "missing_after": missing_after,
    }


def _tushare_status(log_texts: dict[str, str], master_date: str) -> dict[str, Any]:
    combined = "\n".join(log_texts.values())
    hard_markers = ("Set TUSHARE_TOKEN", "token is empty", "Invalid token", "权限", "积分不足")
    warn_markers = ("retry", "timeout", "ConnectionError", "ReadTimeout", "tushare optional")
    hard_hits = [marker for marker in hard_markers if marker in combined]
    warn_hits = [marker for marker in warn_markers if marker in combined]
    if hard_hits:
        status = "failed"
    elif warn_hits:
        status = "warning"
    elif master_date:
        status = "ok"
    else:
        status = "unknown"
    return {"status": status, "latest_master_date": master_date, "hard_hits": hard_hits, "warning_hits": warn_hits[:5]}


def _artifact_patterns(args: argparse.Namespace) -> dict[str, list[str]]:
    return {
        "main": [
            str(Path(args.main_monitor_dir) / "daily_monitor_*_picks.json"),
            str(ROOT / "outputs" / "daily_monitor_auto_today_sync" / "daily_monitor_*_picks.json"),
            str(ROOT / "outputs" / "daily_monitor_auto_today_sync_fast" / "daily_monitor_*_picks.json"),
        ],
        "elastic": [
            str(Path(args.elastic_monitor_dir) / "under20_elastic_*_picks.json"),
            str(ROOT / "outputs" / "daily_monitor_under20_elastic_today*" / "under20_elastic_*_picks.json"),
        ],
        "shortline": [
            str(Path(args.shortline_dir) / "shortline_opportunity_*_cards.json"),
            str(ROOT / "outputs" / "shortline_opportunities" / "shortline_opportunity_*_cards.json"),
            str(ROOT / "outputs" / "shortline_opportunities_live" / "shortline_opportunity_*_cards.json"),
        ],
        "risk": [str(Path(args.risk_dir) / "risk_gate_*.json")],
    }


def collect_ops_context(args: argparse.Namespace) -> dict[str, Any]:
    master_date = latest_master_date(Path(args.master_data_path))
    date_key = args.date_key or master_date.replace("-", "")
    artifacts: dict[str, Any] = {}
    for name, patterns in _artifact_patterns(args).items():
        path = match_date_path(patterns, date_key)
        payload = load_json(path)
        artifacts[name] = {
            "path": str(path) if path else "",
            "date_key": payload_date_key(payload, path),
            "exists": bool(path),
            "count": (
                len(payload.get("observation_pool", []) or payload.get("strategy_snapshot", []) or payload.get("cards", []))
                if isinstance(payload, dict)
                else 0
            ),
            "summary": payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {
                "selection_date": payload.get("selection_date", ""),
                "status": payload.get("status", ""),
                "action_hint": payload.get("action_hint", ""),
            },
        }

    ops_dir = Path(args.ops_log_dir)
    log_texts = {
        "evening": tail_text(ops_dir / "cron_evening_brief.log"),
        "morning": tail_text(ops_dir / "cron_morning_brief.log"),
        "preflight": tail_text(ops_dir / "cron_preflight_evening.log"),
        "post_close": tail_text(ops_dir / "cron_post_close.log"),
        "premium_sync": tail_text(ops_dir / "cron_premium_sync.log"),
    }
    pushplus = {
        "evening": _status_from_log(log_texts["evening"], title_keyword="量化三合一晚报"),
        "morning": _status_from_log(log_texts["morning"], title_keyword="量化热点晨报"),
    }
    preflight = _preflight_status(log_texts["preflight"])
    tushare = _tushare_status(log_texts, master_date)

    artifact_dates = {name: item["date_key"] for name, item in artifacts.items()}
    missing = [name for name, item in artifacts.items() if not item["exists"]]
    mismatched = [name for name, value in artifact_dates.items() if value and date_key and value != date_key]
    failures: list[str] = []
    warnings: list[str] = []
    if missing:
        failures.append("missing_artifacts:" + ",".join(missing))
    if mismatched:
        failures.append("date_mismatch:" + ",".join(mismatched))
    if preflight["status"] == "failed":
        failures.append("preflight_failed")
    if pushplus["evening"]["status"] != "ok":
        warnings.append("evening_pushplus_not_confirmed")
    if pushplus["morning"]["status"] != "ok":
        warnings.append("morning_pushplus_not_confirmed")
    if tushare["status"] == "failed":
        failures.append("tushare_failed")
    elif tushare["status"] in {"warning", "unknown"}:
        warnings.append(f"tushare_{tushare['status']}")

    suggested_commands = []
    if missing or mismatched or preflight["status"] == "failed":
        suggested_commands.append(".venv/bin/python scripts/preflight_evening_brief.py")
    if pushplus["evening"]["status"] != "ok":
        suggested_commands.append(".venv/bin/python scripts/send_pushplus_evening_brief.py --template html")
    if tushare["status"] == "failed":
        suggested_commands.append("检查 .localhome/post_close_monitor.env 中的 TUSHARE_TOKEN / TUSHARE_RATE_LIMIT_SECONDS")
    if "post_close" in log_texts and ("Traceback" in log_texts["post_close"] or "Killed" in log_texts["post_close"]):
        suggested_commands.append("bash scripts/run_cloud_post_close_stable.sh")

    risk_level = "high" if failures else "medium" if warnings else "low"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_date_key": date_key,
        "target_date": f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:]}" if re.fullmatch(r"20\d{6}", date_key) else master_date,
        "system_ok": not failures,
        "risk_level": risk_level,
        "abnormal_chains": failures + warnings,
        "needs_repair": bool(failures),
        "suggested_commands": suggested_commands,
        "artifacts": artifacts,
        "pushplus": pushplus,
        "preflight": preflight,
        "tushare": tushare,
    }


def render_fallback_review(context: dict[str, Any]) -> str:
    abnormal = context.get("abnormal_chains") or ["无"]
    commands = context.get("suggested_commands") or ["无需执行修复命令"]
    lines = [
        f"# LLM 日常运维审查 {context.get('target_date', '')}",
        "",
        f"- 系统是否正常：`{'是' if context.get('system_ok') else '否'}`",
        f"- 风险等级：`{context.get('risk_level')}`",
        f"- 是否需要修复：`{'是' if context.get('needs_repair') else '否'}`",
        f"- 异常链路：`{'; '.join(abnormal)}`",
        "",
        "## 建议命令",
        *[f"- `{cmd}`" for cmd in commands],
        "",
        "## 核心状态",
        f"- 产物日期：`{context.get('target_date_key')}`",
        f"- Preflight：`{(context.get('preflight') or {}).get('status')}`",
        f"- Tushare：`{(context.get('tushare') or {}).get('status')}`",
        f"- 晚报 PushPlus：`{((context.get('pushplus') or {}).get('evening') or {}).get('status')}`",
        f"- 晨报 PushPlus：`{((context.get('pushplus') or {}).get('morning') or {}).get('status')}`",
    ]
    return "\n".join(lines)


def render_llm_review(context: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    fallback = render_fallback_review(context)
    if args.no_llm:
        return fallback, "no_llm"
    settings = resolve_llm_settings(api_key=args.llm_api_key, base_url=args.llm_base_url, model=args.llm_model)
    try:
        content = call_openai_compatible_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是量化系统运维审查助手。只基于用户给出的 JSON 判断，不要编造状态。"
                        "输出中文 markdown，必须包含：系统是否正常、异常链路、是否需要修复、建议命令、风险等级。"
                    ),
                },
                {"role": "user", "content": compact_json(context, limit=18_000)},
            ],
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            model=settings["model"],
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
        )
        return content, "llm"
    except Exception as exc:
        return fallback + f"\n\n> LLM 降级原因：{exc}", "fallback"


def main() -> None:
    args = build_parser().parse_args()
    context = collect_ops_context(args)
    content, mode = render_llm_review(context, args)
    context["render_mode"] = mode
    context["review_markdown"] = content

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_key = str(context.get("target_date_key") or "latest")
    json_path = output_dir / f"llm_daily_ops_review_{date_key}.json"
    md_path = output_dir / f"llm_daily_ops_review_{date_key}.md"
    json_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(content, encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "markdown_path": str(md_path), "system_ok": context["system_ok"], "risk_level": context["risk_level"], "render_mode": mode}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
