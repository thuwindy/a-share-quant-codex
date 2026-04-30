from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import glob
import html
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.mx_bridge import (
    extract_mx_xuangu_rows,
    mx_data_query,
    mx_result_message,
    mx_xuangu_code_name_set,
    mx_xuangu_query,
    summarize_mx_data_result,
    summarize_mx_xuangu_rows,
)

MX_BASE_URL = "https://mkapi2.dfcfs.com/finskillshub/api/claw/news-search"
PREMIUM_DIR = ROOT / "data" / "premium_v22"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send 09:00 morning brief with MX hotspot search, sectors, and strategy picklists.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "research_morning_brief.json"))
    parser.add_argument("--token", default=os.environ.get("PUSHPLUS_TOKEN", ""))
    parser.add_argument("--topic", default=os.environ.get("PUSHPLUS_TOPIC", ""))
    parser.add_argument("--channel", default=os.environ.get("PUSHPLUS_CHANNEL", "wechat"))
    parser.add_argument("--template", choices=("html", "markdown", "txt"), default="html")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--mx-api-key", default=os.environ.get("MX_APIKEY", ""))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "morning_brief"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--use-latest-cache", action="store_true")
    return parser


def _latest_match(patterns: list[str]) -> Path | None:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(item) for item in glob.glob(pattern))
    if not found:
        return None

    def _score(path: Path) -> tuple[str, int, str]:
        match = re.search(r"(20\d{6})", path.name)
        date_key = match.group(1) if match else "00000000"
        priority = 1 if "today_sync_fast" in str(path) else 0
        return (date_key, -priority, str(path))

    return sorted(found, key=_score)[-1]


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_cache_payload(output_dir: Path, selection_date: str) -> tuple[dict[str, Any], str] | None:
    date_key = str(selection_date or "").replace("-", "")
    if not date_key:
        return None
    json_path = output_dir / f"morning_brief_{date_key}.json"
    html_path = output_dir / f"morning_brief_{date_key}.html"
    if not json_path.exists() or not html_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8")), html_path.read_text(encoding="utf-8")


def _path_date_key(path: Path | None) -> str:
    if path is None:
        return ""
    match = re.search(r"(20\d{6})", path.name)
    return match.group(1) if match else ""


def _payload_selection_date(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("summary", "profile"):
        block = payload.get(key, {})
        if isinstance(block, dict):
            value = str(block.get("selection_date") or "").strip()
            if value:
                return value
    return ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _fmt_money_yi(value: Any) -> str:
    return f"{_safe_float(value) / 100000000.0:.2f}亿"


def _industry_counter(items: list[dict[str, Any]], limit: int) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in items[:limit]:
        industry = str(item.get("industry") or "未知")
        counter[industry] += 1
    return counter


def _pick_focus_sectors(
    main_cards: list[dict[str, Any]],
    elastic_cards: list[dict[str, Any]],
    short_cards: list[dict[str, Any]],
    sector_limit: int,
) -> list[str]:
    counter: Counter[str] = Counter()
    counter.update(_industry_counter(main_cards, limit=5))
    counter.update(_industry_counter(elastic_cards, limit=5))
    counter.update(_industry_counter(short_cards, limit=5))
    return [name for name, _ in counter.most_common(sector_limit)]


def _pick_focus_stocks(
    main_cards: list[dict[str, Any]],
    elastic_cards: list[dict[str, Any]],
    short_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item, source in (
        (main_cards[0] if main_cards else None, "主策略"),
        (elastic_cards[0] if elastic_cards else None, "低价弹性"),
        (short_cards[0] if short_cards else None, "短线机会"),
    ):
        if item is None:
            continue
        row = dict(item)
        row["source_bucket"] = source
        out.append(row)
    return out


def _mx_search(query: str, api_key: str, timeout_seconds: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "apikey": api_key}
    payload = {"query": query}
    response = httpx.post(MX_BASE_URL, headers=headers, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def _search_message(result: dict[str, Any]) -> str:
    return mx_result_message(result)


def _is_mx_rate_limit_message(message: str) -> bool:
    text = str(message or "").strip()
    if not text:
        return False
    markers = (
        "调用次数已达到上限",
        "今日调用次数已达到上限",
        "上限50次",
        "免费版用户",
    )
    return any(marker in text for marker in markers)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _extract_news_items(result: dict[str, Any], limit: int, content_char_limit: int) -> list[dict[str, str]]:
    data = result.get("data", {}) if isinstance(result, dict) else {}
    inner = data.get("data", {}) if isinstance(data, dict) else {}
    llm = inner.get("llmSearchResponse", {}) if isinstance(inner, dict) else {}
    items = llm.get("data", []) if isinstance(llm, dict) else []
    out: list[dict[str, str]] = []
    for item in items[:limit]:
        content = str(item.get("content") or "").strip()
        if len(content) > content_char_limit:
            content = content[: content_char_limit - 1].rstrip() + "…"
        out.append(
            {
                "title": str(item.get("title") or ""),
                "date": str(item.get("date") or ""),
                "source": str(item.get("insName") or item.get("entityFullName") or ""),
                "type": str(item.get("informationType") or ""),
                "content": content,
            }
        )
    return out


def _extract_result_text(result: dict[str, Any], char_limit: int) -> str:
    def _inner(raw: Any) -> str:
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, dict):
            for key in ("data", "result"):
                wrapped = raw.get(key)
                text = _inner(wrapped)
                if text:
                    return text
            for key in ("llmSearchResponse", "searchResponse", "content", "answer", "summary"):
                value = raw.get(key)
                text = _inner(value)
                if text:
                    return text
        return ""

    text = _inner(result)
    if text:
        return text[:char_limit]
    items = _extract_news_items(result, limit=4, content_char_limit=max(80, min(char_limit, 180)))
    lines = []
    for item in items:
        title = item.get("title", "")
        content = item.get("content", "")
        lines.append(f"{title}：{content}".strip("："))
    merged = "\n".join(line for line in lines if line)
    return merged[:char_limit]


def _build_reason_from_stock(item: dict[str, Any]) -> str:
    industry = str(item.get("industry") or "未知行业")
    name = str(item.get("name") or item.get("code") or "-")
    source_bucket = str(item.get("source_bucket") or "候选池")
    if "shortline_score" in item:
        return f"{name} 来自{source_bucket}，处于 {industry} 热点方向，具备短线强度与事件催化共振。"
    score = _safe_float(item.get("score"))
    return f"{name} 来自{source_bucket}，位于 {industry} 方向，量化评分 {score:.4f}，适合作为今日重点观察标的。"


def _build_fallback_macro_text(focus_sectors: list[str], focus_stocks: list[dict[str, Any]]) -> str:
    sector_text = "、".join(focus_sectors) if focus_sectors else "暂无明确共振板块"
    stock_lines = []
    for row in focus_stocks[:3]:
        stock_lines.append(
            f"{row.get('name', row.get('code', '-'))}（{row.get('source_bucket', '候选池')} / {row.get('industry', '未知行业')}）"
        )
    stock_text = "；".join(stock_lines) if stock_lines else "暂无重点股票"
    return (
        "妙想实时资讯暂不可用，晨报先回退到量化候选摘要。"
        f"\n今日量化聚焦板块：{sector_text}"
        f"\n重点观察股票：{stock_text}"
    )


def _build_fallback_validation_text(focus_stocks: list[dict[str, Any]]) -> str:
    if not focus_stocks:
        return "妙想盘前校验暂不可用，当前也没有可用候选。"
    lines = ["妙想盘前校验暂不可用，先看量化候选本身："]
    for row in focus_stocks[:3]:
        name = str(row.get("name") or row.get("code") or "-")
        bucket = str(row.get("source_bucket") or "候选池")
        industry = str(row.get("industry") or "未知行业")
        close = _safe_float(row.get("close"))
        reason = _build_reason_from_stock(row)
        lines.append(f"{name} / {bucket} / {industry} / 收盘 {close:.2f}")
        lines.append(reason)
    return "\n".join(lines)


def _latest_limit_sentiment_text(selection_date: str, focus_sectors: list[str]) -> str:
    path = PREMIUM_DIR / "tushare_limit_sentiment_daily.csv"
    if not path.exists():
        return ""
    rows: list[dict[str, str]] = []
    latest_date = ""
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            trade_date = str(row.get("date") or "").strip()
            if not trade_date:
                continue
            if selection_date and trade_date > selection_date:
                continue
            if trade_date >= latest_date:
                if trade_date > latest_date:
                    rows = []
                    latest_date = trade_date
                rows.append(row)
    if not rows:
        return ""
    up_count = 0
    open_count = 0
    industry_counter: Counter[str] = Counter()
    for row in rows:
        limit_flag = str(row.get("limit") or "").strip().upper()
        open_times = _safe_float(row.get("open_times"))
        if limit_flag == "U":
            up_count += 1
        if open_times > 0:
            open_count += 1
        industry_counter[str(row.get("industry") or "未知行业")] += 1
    top_industries = "、".join(name for name, _ in industry_counter.most_common(3))
    sector_hint = "、".join(focus_sectors) if focus_sectors else top_industries
    return (
        f"Tushare 涨停情绪（{latest_date}）：涨停样本 {up_count} 家，炸板 {open_count} 家，"
        f"热门行业集中在 {top_industries or '暂无统计'}。"
        f" 当前量化聚焦方向：{sector_hint or '暂无明显共振'}。"
    )


def _recent_report_rc_text(selection_date: str, focus_stocks: list[dict[str, Any]]) -> str:
    path = PREMIUM_DIR / "tushare_report_rc_events.csv"
    if not path.exists() or not focus_stocks:
        return ""
    focus_codes = {str(row.get("code") or "").strip().upper() for row in focus_stocks if str(row.get("code") or "").strip()}
    latest_by_code: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            code = str(row.get("code") or "").strip().upper()
            if code not in focus_codes:
                continue
            report_date = str(row.get("report_date") or "").strip()
            if selection_date and report_date and report_date > selection_date:
                continue
            prev = latest_by_code.get(code)
            if prev is None or report_date >= str(prev.get("report_date") or ""):
                latest_by_code[code] = row
    lines: list[str] = []
    for row in focus_stocks[:3]:
        code = str(row.get("code") or "").strip().upper()
        payload = latest_by_code.get(code)
        if not payload:
            continue
        target_max = str(payload.get("max_price") or "").strip()
        target_min = str(payload.get("min_price") or "").strip()
        target_text = ""
        if target_min or target_max:
            target_text = f"，目标价区间 {target_min or '-'} ~ {target_max or '-'}"
        lines.append(
            f"{row.get('name', code)}：{payload.get('report_date', '-')}"
            f" 研报评级 {payload.get('rating', '-')}"
            f"（{payload.get('org_name', '-') }）{target_text}"
        )
    return "\n".join(lines)


def _recent_hk_hold_text(selection_date: str, focus_stocks: list[dict[str, Any]]) -> str:
    path = PREMIUM_DIR / "tushare_hk_hold_daily.csv"
    if not path.exists() or not focus_stocks:
        return ""
    focus_codes = {str(row.get("code") or "").strip().upper() for row in focus_stocks if str(row.get("code") or "").strip()}
    latest_by_code: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "code" not in reader.fieldnames:
            return ""
        for row in reader:
            code = str(row.get("code") or "").strip().upper()
            if code not in focus_codes:
                continue
            trade_date = str(row.get("date") or "").strip()
            if selection_date and trade_date and trade_date > selection_date:
                continue
            prev = latest_by_code.get(code)
            if prev is None or trade_date >= str(prev.get("date") or ""):
                latest_by_code[code] = row
    lines: list[str] = []
    for row in focus_stocks[:3]:
        code = str(row.get("code") or "").strip().upper()
        payload = latest_by_code.get(code)
        if not payload:
            continue
        ratio = str(payload.get("ratio") or payload.get("hold_ratio") or payload.get("vol_ratio") or "").strip()
        lines.append(f"{row.get('name', code)}：北向持股数据更新至 {payload.get('date', '-') }，持股占比 {ratio or '-'}")
    return "\n".join(lines)


def _normalize_identity_set(rows: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        code = str(row.get("code") or "").strip().upper()
        name = str(row.get("name") or "").strip().upper()
        if code:
            out.add(code)
        if name:
            out.add(name)
    return out


def _render_news_block(title: str, items: list[dict[str, str]]) -> str:
    if not items:
        return (
            '<div style="background:#141a2a;border:1px solid #26324a;border-radius:16px;'
            'padding:16px 18px;margin:12px 0;">'
            f'<div style="font-size:16px;font-weight:800;color:#ffffff;margin-bottom:8px;">{html.escape(title)}</div>'
            '<div style="color:#cbd5e1;font-size:13px;">暂无可用资讯。</div></div>'
        )
    lines = []
    for item in items:
        meta = " / ".join(x for x in [item["date"], item["source"], item["type"]] if x)
        lines.append(
            '<div style="padding:10px 0;border-top:1px solid #26324a;">'
            f'<div style="color:#ffffff;font-size:14px;font-weight:700;">{html.escape(item["title"])}</div>'
            f'<div style="color:#8ea2c8;font-size:12px;margin-top:4px;">{html.escape(meta)}</div>'
            f'<div style="color:#d8e3ff;font-size:13px;line-height:1.6;margin-top:6px;">{html.escape(item["content"])}</div>'
            '</div>'
        )
    return (
        '<div style="background:#141a2a;border:1px solid #26324a;border-radius:16px;'
        'padding:16px 18px;margin:12px 0;box-shadow:0 6px 20px rgba(0,0,0,0.18);">'
        f'<div style="font-size:16px;font-weight:800;color:#ffffff;margin-bottom:4px;">{html.escape(title)}</div>'
        + "".join(lines)
        + "</div>"
    )


def _render_notice_block(title: str, message: str) -> str:
    return (
        '<div style="background:#2a1b14;border:1px solid #5a3326;border-radius:16px;'
        'padding:16px 18px;margin:12px 0;">'
        f'<div style="font-size:16px;font-weight:800;color:#ffd3b6;margin-bottom:8px;">{html.escape(title)}</div>'
        f'<div style="color:#ffe7d6;font-size:13px;line-height:1.7;">{html.escape(message)}</div>'
        '</div>'
    )


def _render_text_panel(title: str, content: str) -> str:
    safe = html.escape(content.strip() or "暂无可用资讯。")
    safe = safe.replace("\n", "<br>")
    return (
        '<div style="background:#141a2a;border:1px solid #26324a;border-radius:16px;'
        'padding:16px 18px;margin:12px 0;">'
        f'<div style="font-size:16px;font-weight:800;color:#ffffff;margin-bottom:8px;">{html.escape(title)}</div>'
        f'<div style="color:#d8e3ff;font-size:13px;line-height:1.7;">{safe}</div>'
        '</div>'
    )


def _render_match_list(title: str, items: list[dict[str, Any]], empty_text: str) -> str:
    if not items:
        return _render_text_panel(title, empty_text)
    lines = []
    for row in items:
        code = str(row.get("code") or "-")
        name = str(row.get("name") or "-")
        industry = str(row.get("industry") or "-")
        score = _safe_float(row.get("score"))
        close = _safe_float(row.get("close"))
        lines.append(f"• {name}({code}) / {industry} / 收盘 {close:.2f} / 量化分 {score:.4f}")
    return _render_text_panel(title, "\n".join(lines))


def _render_pick_card(item: dict[str, Any]) -> str:
    close = _safe_float(item.get("close"))
    score = _safe_float(item.get("score", item.get("shortline_score")))
    reason = _build_reason_from_stock(item)
    bucket = str(item.get("source_bucket") or "候选池")
    return (
        '<div style="background:#171d2e;border:1px solid #2b3b5a;border-radius:18px;padding:16px;'
        'margin:10px 0;box-shadow:0 8px 22px rgba(0,0,0,0.2);">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<div style="font-size:18px;font-weight:800;color:#ffffff;">{html.escape(str(item.get("name", "-")))}</div>'
        f'<span style="display:inline-block;padding:4px 10px;border-radius:999px;background:#2f7df6;color:#fff;font-size:12px;font-weight:700;">{html.escape(bucket)}</span></div>'
        f'<div style="color:#8ea2c8;font-size:13px;margin-bottom:10px;">{html.escape(str(item.get("code", "-")))} · {html.escape(str(item.get("industry", "-")))}</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px 16px;">'
        f'<div><div style="color:#8ea2c8;font-size:12px;">收盘价</div><div style="color:#ff7575;font-size:20px;font-weight:800;">{close:.2f}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">评分</div><div style="color:#7ee787;font-size:20px;font-weight:800;">{score:.4f}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">成交额</div><div style="color:#d8e3ff;font-size:15px;font-weight:700;">{_fmt_money_yi(item.get("amount", item.get("fd_amount")))}</div></div>'
        '</div>'
        f'<div style="margin-top:8px;color:#d8e3ff;font-size:13px;line-height:1.7;">{html.escape(reason)}</div>'
        '</div>'
    )


def _post_pushplus(token: str, title: str, content: str, template: str, topic: str, channel: str, timeout_seconds: float) -> dict[str, Any]:
    payload = {"token": token, "title": title, "content": content, "template": template, "channel": channel}
    if topic:
        payload["topic"] = topic
    response = httpx.post("https://www.pushplus.plus/send", json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = build_parser().parse_args()
    cfg = _load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.mx_api_key.strip():
        raise ValueError("MX_APIKEY is required.")

    main_path = _latest_match([
        str(ROOT / "outputs" / "daily_monitor_auto" / "daily_monitor_*_picks.json"),
        str(ROOT / "outputs" / "daily_monitor_auto_today_sync_fast" / "daily_monitor_*_picks.json"),
        str(ROOT / "outputs" / "daily_monitor_auto_today_sync" / "daily_monitor_*_picks.json"),
        str(ROOT / "outputs" / "daily_monitor_auto_direct" / "daily_monitor_*_picks.json"),
    ])
    elastic_path = _latest_match([
        str(ROOT / "outputs" / "daily_monitor_under20_elastic" / "under20_elastic_*_picks.json"),
        str(ROOT / "outputs" / "daily_monitor_under20_elastic_today_v2" / "under20_elastic_*_picks.json"),
        str(ROOT / "outputs" / "daily_monitor_under20_elastic_today*" / "under20_elastic_*_picks.json"),
    ])
    shortline_path = _latest_match([
        str(ROOT / "outputs" / "shortline_opportunities" / "shortline_opportunity_*_cards.json"),
        str(ROOT / "outputs" / "shortline_opportunities_live" / "shortline_opportunity_*_cards.json"),
    ])

    main_payload = _load_json(main_path)
    elastic_payload = _load_json(elastic_path)
    shortline_payload = _load_json(shortline_path)

    main_date = _payload_selection_date(main_payload) or (
        f"{_path_date_key(main_path)[:4]}-{_path_date_key(main_path)[4:6]}-{_path_date_key(main_path)[6:8]}"
        if _path_date_key(main_path)
        else ""
    )
    elastic_date = _payload_selection_date(elastic_payload) or (
        f"{_path_date_key(elastic_path)[:4]}-{_path_date_key(elastic_path)[4:6]}-{_path_date_key(elastic_path)[6:8]}"
        if _path_date_key(elastic_path)
        else ""
    )
    shortline_date = _payload_selection_date(shortline_payload) or (
        f"{_path_date_key(shortline_path)[:4]}-{_path_date_key(shortline_path)[4:6]}-{_path_date_key(shortline_path)[6:8]}"
        if _path_date_key(shortline_path)
        else ""
    )
    selection_date = max([x for x in [main_date, elastic_date, shortline_date] if x], default="")

    main_cards = list(main_payload.get("observation_pool", [])[:5]) if main_date == selection_date else []
    elastic_cards = list(elastic_payload.get("observation_pool", [])[:5]) if elastic_date == selection_date else []
    short_cards = list(shortline_payload.get("cards", [])[:5]) if shortline_date == selection_date else []
    date_key = selection_date.replace("-", "")
    today_key = datetime.now().strftime("%Y-%m-%d")
    title = f"量化热点晨报 {today_key}".strip()
    if selection_date and selection_date != today_key:
        title = f"{title}（基于 {selection_date} 收盘）"

    if args.use_latest_cache:
        cached = _load_cache_payload(output_dir, selection_date)
        if cached is not None:
            cached_result, cached_content = cached
            cached_result["title"] = title
            cached_result["template"] = args.template
            cached_result["channel"] = args.channel
            cached_result["selection_date"] = selection_date
            cached_result["dry_run"] = bool(args.dry_run)
            if args.dry_run or args.cache_only:
                cached_result["preview"] = cached_content[:6000]
                print(json.dumps(cached_result, ensure_ascii=False, indent=2))
                return

            token = str(args.token).strip()
            if not token:
                raise ValueError("PUSHPLUS_TOKEN is required.")
            cached_result["pushplus_result"] = _post_pushplus(
                token=token,
                title=title,
                content=cached_content,
                template=args.template,
                topic=str(args.topic).strip(),
                channel=str(args.channel).strip(),
                timeout_seconds=float(args.timeout_seconds),
            )
            save_json = output_dir / f"morning_brief_{date_key or 'latest'}.json"
            save_json.write_text(json.dumps(cached_result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(cached_result, ensure_ascii=False, indent=2))
            return

    focus_sectors = _pick_focus_sectors(main_cards, elastic_cards, short_cards, sector_limit=_safe_int(cfg.get("sector_limit"), 2))
    focus_stocks = _pick_focus_stocks(main_cards, elastic_cards, short_cards)
    focus_stock_names = "、".join(
        str(row.get("name", row.get("code", "")))
        for row in focus_stocks[: _safe_int(cfg.get("stock_limit_per_section"), 3)]
    )
    validation_stock_names = "、".join(
        str(row.get("name", row.get("code", "")))
        for row in (main_cards[:2] + elastic_cards[:2])[: _safe_int(cfg.get("mx_data_stock_limit"), 4)]
    )

    notices: list[str] = []
    mx_rate_limit_sources: list[str] = []
    macro_items: list[dict[str, str]] = []
    macro_text = ""
    company_text = ""
    mx_data_text = ""
    mx_data_query_text = ""
    mx_xuangu_query_text = ""
    mx_xuangu_rows: list[dict[str, Any]] = []
    elastic_secondary_matches: list[dict[str, Any]] = []
    sector_blocks: list[tuple[str, list[dict[str, str]], str]] = []
    company_blocks: list[tuple[str, list[dict[str, str]], str]] = []

    query_mode = str(cfg.get("mx_query_mode", "compact_dual")).strip()
    if query_mode == "compact_dual":
        sector_query = str(cfg.get("combined_sector_query_template", cfg.get("macro_query", ""))).format(
            sectors="、".join(focus_sectors) if focus_sectors else "暂无明确热点板块"
        )
        macro_result = _mx_search(sector_query, args.mx_api_key, args.timeout_seconds)
        macro_message = _search_message(macro_result)
        if macro_message:
            if _is_mx_rate_limit_message(macro_message):
                mx_rate_limit_sources.append("政策/行业热点")
            else:
                notices.append(f"政策/行业热点：{macro_message}")
        macro_items = _extract_news_items(
            macro_result,
            limit=_safe_int(cfg.get("news_items_per_query"), 2),
            content_char_limit=_safe_int(cfg.get("content_char_limit"), 140),
        )
        macro_text = _extract_result_text(macro_result, 1500)

        company_query = str(cfg.get("combined_company_query_template", "{stocks}")).format(stocks=focus_stock_names or "暂无重点股票")
        company_result = _mx_search(company_query, args.mx_api_key, args.timeout_seconds)
        company_message = _search_message(company_result)
        if company_message:
            if _is_mx_rate_limit_message(company_message):
                mx_rate_limit_sources.append("公司突发/公告/研报")
            else:
                notices.append(f"公司突发/公告/研报：{company_message}")
        company_text = _extract_result_text(company_result, 2600)
    else:
        macro_result = _mx_search(str(cfg.get("macro_query", "")).strip(), args.mx_api_key, args.timeout_seconds)
        macro_items = _extract_news_items(
            macro_result,
            limit=_safe_int(cfg.get("news_items_per_query"), 2),
            content_char_limit=_safe_int(cfg.get("content_char_limit"), 140),
        )
        macro_message = _search_message(macro_result)
        if macro_message:
            if _is_mx_rate_limit_message(macro_message):
                mx_rate_limit_sources.append("政策快讯与热点摘要")
            else:
                notices.append(f"政策快讯与热点摘要：{macro_message}")

        sector_tpl = str(cfg.get("sector_query_template", "{sector}板块最新新闻")).strip()
        for sector in focus_sectors:
            query = sector_tpl.format(sector=sector)
            result = _mx_search(query, args.mx_api_key, args.timeout_seconds)
            items = _extract_news_items(
                result,
                limit=_safe_int(cfg.get("news_items_per_query"), 2),
                content_char_limit=_safe_int(cfg.get("content_char_limit"), 140),
            )
            message = _search_message(result)
            if message:
                if _is_mx_rate_limit_message(message):
                    mx_rate_limit_sources.append(f"{sector}板块内部新闻")
                else:
                    notices.append(f"{sector} 板块内部新闻：{message}")
            sector_blocks.append((sector, items, query))

        company_tpl = str(cfg.get("company_query_template", "{name}最新公告")).strip()
        for row in focus_stocks[: _safe_int(cfg.get("stock_limit_per_section"), 3)]:
            query = company_tpl.format(name=row.get("name", row.get("code", "")))
            result = _mx_search(query, args.mx_api_key, args.timeout_seconds)
            items = _extract_news_items(
                result,
                limit=_safe_int(cfg.get("news_items_per_query"), 2),
                content_char_limit=_safe_int(cfg.get("content_char_limit"), 140),
            )
            message = _search_message(result)
            if message:
                if _is_mx_rate_limit_message(message):
                    mx_rate_limit_sources.append(f"{row.get('name', '-') }公司突发/公告/研报")
                else:
                    notices.append(f"{row.get('name', '-') } 公司突发/公告/研报：{message}")
            company_blocks.append((str(row.get("name", "-")), items, query))

    if validation_stock_names:
        mx_data_query_text = str(cfg.get("mx_data_query_template", "{stocks}")).format(stocks=validation_stock_names)
        mx_data_result = mx_data_query(mx_data_query_text, args.mx_api_key, args.timeout_seconds)
        mx_data_message = mx_result_message(mx_data_result)
        if mx_data_message:
            if _is_mx_rate_limit_message(mx_data_message):
                mx_rate_limit_sources.append("盘前数据校验")
            else:
                notices.append(f"盘前数据校验：{mx_data_message}")
        mx_data_text = summarize_mx_data_result(mx_data_result, table_limit=3, row_limit=2, value_limit=4)

    if elastic_cards:
        elastic_sectors = "、".join(
            name for name, _count in _industry_counter(elastic_cards, limit=8).most_common(_safe_int(cfg.get("sector_limit"), 2))
        ) or "通信设备、电气设备"
        mx_xuangu_query_text = str(cfg.get("mx_xuangu_query_template", "")).format(sectors=elastic_sectors)
        mx_xuangu_result = mx_xuangu_query(mx_xuangu_query_text, args.mx_api_key, args.timeout_seconds)
        mx_xuangu_message = mx_result_message(mx_xuangu_result)
        if mx_xuangu_message:
            if _is_mx_rate_limit_message(mx_xuangu_message):
                mx_rate_limit_sources.append("低价弹性二次筛选")
            else:
                notices.append(f"低价弹性二次筛选：{mx_xuangu_message}")
        mx_xuangu_rows = extract_mx_xuangu_rows(
            mx_xuangu_result,
            row_limit=_safe_int(cfg.get("mx_xuangu_elastic_limit"), 8),
        )
        identity_set = mx_xuangu_code_name_set(mx_xuangu_rows)
        elastic_secondary_matches = [
            row
            for row in elastic_cards
            if str(row.get("code", "")).strip().upper() in identity_set
            or str(row.get("name", "")).strip().upper() in identity_set
        ][: _safe_int(cfg.get("mx_xuangu_display_limit"), 5)]

    if sector_blocks:
        conclusion = cfg.get("morning_conclusion_templates", {}).get("strong", "热点较集中。")
    elif macro_items:
        conclusion = cfg.get("morning_conclusion_templates", {}).get("mixed", "热点分化。")
    else:
        conclusion = cfg.get("morning_conclusion_templates", {}).get("weak", "消息面较平静。")

    notices = _dedupe_keep_order(notices)
    rate_limit_sources = _dedupe_keep_order(mx_rate_limit_sources)
    stale_modules: list[str] = []
    if main_date and main_date != selection_date:
        stale_modules.append(f"主策略停留在 {main_date}")
    if elastic_date and elastic_date != selection_date:
        stale_modules.append(f"低价弹性停留在 {elastic_date}")
    if shortline_date and shortline_date != selection_date:
        stale_modules.append(f"短线机会停留在 {shortline_date}")
    if stale_modules:
        notices.insert(0, "部分支线未更新到同一天，已自动剔除旧日期内容：" + "；".join(stale_modules))
    if rate_limit_sources:
        notices.insert(
            0,
            "妙想今日免费额度已用尽，"
            + "、".join(rate_limit_sources)
            + "未返回实时内容；晨报已回退为量化候选与本地摘要。",
        )

    fallback_macro_text = _build_fallback_macro_text(focus_sectors, focus_stocks)
    tushare_limit_text = _latest_limit_sentiment_text(selection_date, focus_sectors)
    tushare_report_text = _recent_report_rc_text(selection_date, focus_stocks)
    tushare_hk_text = _recent_hk_hold_text(selection_date, focus_stocks)
    if not macro_items and not macro_text:
        macro_parts = [part for part in [tushare_limit_text, fallback_macro_text] if part]
        macro_text = "\n".join(macro_parts)
    if not mx_data_text:
        validation_parts = [_build_fallback_validation_text(focus_stocks)]
        if tushare_report_text:
            validation_parts.append("Tushare 研报跟踪：\n" + tushare_report_text)
        if tushare_hk_text:
            validation_parts.append("Tushare 北向跟踪：\n" + tushare_hk_text)
        mx_data_text = "\n\n".join(part for part in validation_parts if part)
    low_price_empty_text = "当前二次筛选未与量化低价池形成明确交集，建议保持观察。"
    if rate_limit_sources and "低价弹性二次筛选" in rate_limit_sources:
        low_price_empty_text = "妙想低价二次筛选今日额度已用尽，先以量化低价弹性池原始候选为准。"
    if query_mode == "compact_dual" and not company_text:
        company_text = "妙想公司资讯查询暂不可用，先以量化候选与公告复核为准。"

    html_parts = [
        '<div style="background:#0b1020;padding:18px 14px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">',
        '<div style="max-width:900px;margin:0 auto;">',
        '<div style="background:linear-gradient(135deg,#1b2b4b,#0f172a);border-radius:22px;padding:22px 20px;margin-bottom:16px;border:1px solid #2b3b5a;">',
        '<div style="font-size:24px;font-weight:900;color:#fff;">量化热点晨报</div>',
        f'<div style="margin-top:8px;color:#a8b8d8;font-size:14px;">推送日期：{html.escape(today_key)} · 基于 {html.escape(selection_date or today_key)} 收盘结果与妙想资讯搜索</div>',
        f'<div style="margin-top:10px;color:#d8e3ff;font-size:14px;line-height:1.7;">晨报结论：<b>{html.escape(str(conclusion))}</b></div>',
        '</div>',
    ]
    html_parts.append(
        '<div style="background:#141a2a;border:1px solid #26324a;border-radius:16px;padding:16px 18px;margin:12px 0;">'
        '<div style="font-size:16px;font-weight:800;color:#ffffff;margin-bottom:8px;">重点相关股票</div>'
    )
    html_parts.extend(_render_pick_card(row) for row in focus_stocks[: _safe_int(cfg.get("stock_limit_per_section"), 3)])
    html_parts.append("</div>")
    if notices:
        html_parts.append(_render_notice_block("妙想资讯接口状态", "；".join(notices[:3])))
    if macro_items:
        html_parts.append(_render_news_block("政策快讯与热点摘要", macro_items))
    else:
        html_parts.append(_render_text_panel("政策快讯与热点摘要", macro_text[:3000]))
    html_parts.append(_render_text_panel("盘前数据校验（mx-data）", mx_data_text[:3000]))
    if query_mode != "compact_dual":
        for sector, items, _query in sector_blocks:
            html_parts.append(_render_news_block(f"{sector} 板块内部新闻", items))
    html_parts.append(
        _render_match_list(
            "低价弹性池二次筛选（mx-xuangu 交集）",
            elastic_secondary_matches,
            low_price_empty_text,
        )
    )
    if mx_xuangu_rows:
        html_parts.append(_render_text_panel("低价方向外部筛选候选", summarize_mx_xuangu_rows(mx_xuangu_rows, row_limit=5)))
    if query_mode == "compact_dual":
        html_parts.append(_render_text_panel("重点股票公司突发/公告/研报（合并查询）", company_text[:3000]))
    else:
        for name, items, _query in company_blocks:
            html_parts.append(_render_news_block(f"{name} 公司突发/公告/研报", items))
    html_parts.append("</div></div>")
    content = "".join(html_parts)

    result = {
        "title": title,
        "template": args.template,
        "channel": args.channel,
        "selection_date": selection_date,
        "main_selection_date": main_date,
        "elastic_selection_date": elastic_date,
        "shortline_selection_date": shortline_date,
        "main_path": str(main_path) if main_path else "",
        "elastic_path": str(elastic_path) if elastic_path else "",
        "shortline_path": str(shortline_path) if shortline_path else "",
        "focus_sectors": focus_sectors,
        "focus_stocks": [row.get("name", row.get("code", "")) for row in focus_stocks],
        "macro_items": macro_items,
        "mx_data_query": mx_data_query_text,
        "mx_data_text": mx_data_text,
        "mx_xuangu_query": mx_xuangu_query_text,
        "mx_xuangu_rows": mx_xuangu_rows[: _safe_int(cfg.get("mx_xuangu_display_limit"), 5)],
        "elastic_secondary_matches": [row.get("code", "") for row in elastic_secondary_matches],
        "tushare_limit_text": tushare_limit_text,
        "tushare_report_text": tushare_report_text,
        "tushare_hk_text": tushare_hk_text,
        "mx_rate_limit_sources": rate_limit_sources,
        "dry_run": bool(args.dry_run),
    }

    save_json = output_dir / f"morning_brief_{date_key or 'latest'}.json"
    save_html = output_dir / f"morning_brief_{date_key or 'latest'}.html"
    save_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_html.write_text(content, encoding="utf-8")

    if args.dry_run or args.cache_only:
        result["preview"] = content[:6000]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    token = str(args.token).strip()
    if not token:
        raise ValueError("PUSHPLUS_TOKEN is required.")
    result["pushplus_result"] = _post_pushplus(
        token=token,
        title=title,
        content=content,
        template=args.template,
        topic=str(args.topic).strip(),
        channel=str(args.channel).strip(),
        timeout_seconds=float(args.timeout_seconds),
    )
    save_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
