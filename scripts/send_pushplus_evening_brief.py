from __future__ import annotations

import argparse
from collections import Counter
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
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_PUSHPLUS_CONTENT_LIMIT = 18000
PUSHPLUS_OVERSIZE_MARKERS = ("发送内容过大", "不能超过2万字", "超过2万字")

from ashare_quant.analysis.latest_picks import _annotate_premium_observation_layers, build_review_question_pack  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send evening brief with main strategy, under-20 elastic pool, and shortline opportunities.")
    parser.add_argument("--token", default=os.environ.get("PUSHPLUS_TOKEN", ""))
    parser.add_argument("--topic", default=os.environ.get("PUSHPLUS_TOPIC", ""))
    parser.add_argument("--channel", default=os.environ.get("PUSHPLUS_CHANNEL", "wechat"))
    parser.add_argument("--template", choices=("html", "markdown", "txt"), default="html")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--master-data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--content-limit", type=int, default=DEFAULT_PUSHPLUS_CONTENT_LIMIT)
    parser.add_argument("--dry-run", action="store_true")
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
        priority = 1 if "auto_direct" in str(path) else 0
        return (date_key, -priority, str(path))
    return sorted(found, key=_score)[-1]


def _match_date_path(patterns: list[str], date_key: str) -> Path | None:
    if not date_key:
        return None
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(item) for item in glob.glob(pattern))
    matched = [path for path in found if date_key in path.name]
    if not matched:
        return None
    return sorted(matched)[-1]


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _path_date_key(path: Path | None) -> str:
    if path is None:
        return ""
    match = re.search(r"(20\d{6})", path.name)
    return match.group(1) if match else ""


def _payload_date_key(payload: dict[str, Any], path: Path | None) -> str:
    selection_date = str((payload or {}).get("summary", {}).get("selection_date") or "").strip()
    if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", selection_date):
        return selection_date.replace("-", "")
    return _path_date_key(path)


def _read_tail_date(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        chunk = 4096
        data = b""
        pos = size
        while pos > 0 and b"\n" not in data:
            step = min(chunk, pos)
            pos -= step
            fh.seek(pos)
            data = fh.read(step) + data
        lines = [line for line in data.decode("utf-8", errors="ignore").splitlines() if line.strip()]
        if not lines:
            return ""
        return lines[-1].split(",", 1)[0].strip()


def _validate_report_dates(
    *,
    expected_date_key: str,
    main_date_key: str,
    elastic_date_key: str,
    shortline_date_key: str,
    risk_date_key: str,
) -> None:
    date_map = {
        "main": main_date_key,
        "elastic": elastic_date_key,
        "shortline": shortline_date_key,
        "risk": risk_date_key,
    }
    missing = [name for name, value in date_map.items() if not value]
    if missing:
        raise ValueError(f"missing report date for: {', '.join(missing)}")
    unique_dates = sorted(set(date_map.values()))
    if len(unique_dates) != 1:
        raise ValueError(
            "report date mismatch: "
            + ", ".join(f"{name}={value}" for name, value in date_map.items())
        )
    if expected_date_key and unique_dates[0] != expected_date_key:
        raise ValueError(
            f"report date stale: expected {expected_date_key}, got {unique_dates[0]}"
        )


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


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:4px 10px;border-radius:999px;'
        f'background:{color};color:#fff;font-size:12px;font-weight:700;">{html.escape(text)}</span>'
    )


def _summary_card(title: str, lines: list[str]) -> str:
    body = "".join(f'<div style="margin:6px 0;color:#d8e3ff;font-size:13px;line-height:1.6;">{line}</div>' for line in lines)
    return (
        '<div style="background:#141a2a;border:1px solid #26324a;border-radius:16px;'
        'padding:16px 18px;margin:12px 0;box-shadow:0 6px 20px rgba(0,0,0,0.18);">'
        f'<div style="font-size:16px;font-weight:800;color:#ffffff;margin-bottom:8px;">{html.escape(title)}</div>'
        f"{body}</div>"
    )


def _compact_line(lines: list[str], default: str = "暂无") -> str:
    cleaned = [str(line).strip() for line in lines if str(line).strip()]
    return cleaned[0] if cleaned else default


def _observation_summary(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "ml_observation_tag",
        "industry_breadth_tag",
        "industry_leader_follow_tag",
        "overhead_density_tag",
        "moneyflow_confirmation_tag",
        "chip_overhead_tag",
        "fundamental_context_tag",
    ):
        value = str(item.get(key) or "").strip()
        if value:
            parts.append(value)
    return " / ".join(parts[:4]) if parts else "观察层暂无额外标签"


def _top_names_line(items: list[dict[str, Any]], *, score_key: str, limit: int = 3) -> str:
    parts: list[str] = []
    for item in items[:limit]:
        name = str(item.get("name") or item.get("code") or "-")
        score = _safe_float(item.get(score_key))
        parts.append(f"{html.escape(name)}({score:.4f})")
    return " / ".join(parts) if parts else "暂无"


def _fmt_signed(value: float) -> str:
    return f"{value:+.4f}"


def _factor_label(name: str) -> str:
    mapping = {
        "momentum_20_neu": "20日动量",
        "momentum_60_neu": "60日动量",
        "smart_money_inflow_20_neu": "20日聪明钱流入",
        "volatility_20_neu": "20日波动",
        "turnover_20_neu": "20日换手",
    }
    return mapping.get(name, name)


def _score_reason_pack(item: dict[str, Any], factor_weights: dict[str, Any]) -> dict[str, list[str] | str]:
    contributions: list[tuple[str, float, float, float]] = []
    neutral_lines: list[str] = []
    for factor, weight_raw in factor_weights.items():
        value = item.get(factor)
        if value is None:
            continue
        factor_value = _safe_float(value)
        factor_weight = _safe_float(weight_raw)
        contribution = factor_weight * factor_value
        if abs(contribution) < 1e-9:
            neutral_lines.append(
                f"{_factor_label(factor)}={factor_value:.4f}，当前权重={_fmt_signed(factor_weight)}，本期贡献接近 0"
            )
            continue
        contributions.append((factor, factor_value, factor_weight, contribution))

    positive = [item for item in contributions if item[3] > 0]
    negative = [item for item in contributions if item[3] < 0]
    positive.sort(key=lambda row: row[3], reverse=True)
    negative.sort(key=lambda row: row[3])

    plus_lines = [
        f"{_factor_label(name)}={value:.4f}，权重={_fmt_signed(weight)}，贡献={_fmt_signed(contribution)}"
        for name, value, weight, contribution in positive[:3]
    ]
    minus_lines = [
        f"{_factor_label(name)}={value:.4f}，权重={_fmt_signed(weight)}，贡献={_fmt_signed(contribution)}"
        for name, value, weight, contribution in negative[:3]
    ]

    passed = [
        f"主策略排序第 {_safe_int(item.get('rank'))} 名，已进入主池前 20",
        f"目标权重 {_safe_float(item.get('target_weight')) * 100:.2f}%，说明组合愿意给仓位",
    ]
    failed: list[str] = []

    ml_rank = _safe_int(item.get("ml_rank"))
    ml_score = _safe_float(item.get("ml_score"))
    ml_tag = str(item.get("ml_observation_tag") or "").strip()
    breadth_score = _safe_float(item.get("industry_breadth_score"))
    breadth_rank = _safe_int(item.get("industry_breadth_rank"))
    breadth_tag = str(item.get("industry_breadth_tag") or "").strip()
    breadth_detail = str(item.get("industry_breadth_detail") or "").strip()
    leader_follow_score = _safe_float(item.get("industry_leader_follow_score"))
    leader_follow_rank = _safe_int(item.get("industry_leader_follow_rank"))
    leader_follow_tag = str(item.get("industry_leader_follow_tag") or "").strip()
    leader_follow_detail = str(item.get("industry_leader_follow_detail") or "").strip()
    overhead_density_score = _safe_float(item.get("overhead_density_score"))
    overhead_density_rank = _safe_int(item.get("overhead_density_rank"))
    overhead_density_tag = str(item.get("overhead_density_tag") or "").strip()
    overhead_density_detail = str(item.get("overhead_density_detail") or "").strip()
    moneyflow_score = _safe_float(item.get("moneyflow_confirmation_score"))
    moneyflow_rank = _safe_int(item.get("moneyflow_confirmation_rank"))
    moneyflow_tag = str(item.get("moneyflow_confirmation_tag") or "").strip()
    moneyflow_detail = str(item.get("moneyflow_confirmation_detail") or "").strip()
    if ml_tag:
        passed.append(f"ML观察分={ml_score:.4f}，ML排序第 {ml_rank} 名")
        if "ml_score低" in ml_tag:
            failed.append("未进入 ML 前 10，机器学习不共振")
        elif "异常高" in ml_tag:
            passed.append("虽然主排序不在最前，但 ML 认为它有短线爆发特征")
        elif "ml_score高" in ml_tag:
            passed.append("进入 ML 前 10，主策略与 ML 共振")
    if breadth_tag:
        passed.append(f"板块确认分={breadth_score:.4f}，板块排序第 {breadth_rank} 名")
        if breadth_detail:
            passed.append(breadth_detail)
        if breadth_tag == "板块共振强":
            passed.append("所属行业同步走强，不只是单票独立上涨")
        elif breadth_tag == "板块确认一般":
            passed.append("所属行业有一定跟随，但还不是最强扩散")
        elif breadth_tag == "板块确认弱":
            failed.append("板块联动偏弱，更像个股独立走强")
    if leader_follow_tag:
        passed.append(f"龙头扩散分={leader_follow_score:.4f}，扩散排序第 {leader_follow_rank} 名")
        if leader_follow_detail:
            passed.append(leader_follow_detail)
        if leader_follow_tag == "龙头扩散强":
            passed.append("行业龙头已确认，个股处在板块跟随扩散段")
        elif leader_follow_tag == "龙头扩散一般":
            passed.append("行业有龙头带动，但跟随扩散还不算充分")
        elif leader_follow_tag == "龙头扩散弱":
            failed.append("缺少龙头带动的跟随扩散，不像板块二次发酵票")
    if overhead_density_tag:
        passed.append(f"兑现压力分={overhead_density_score:.4f}，主池压力排序第 {overhead_density_rank} 名")
        if overhead_density_detail:
            passed.append(overhead_density_detail)
        if overhead_density_tag == "兑现压力轻":
            passed.append("最近20日上方兑现压力偏轻，强势更容易保住利润")
        elif overhead_density_tag == "兑现压力一般":
            passed.append("最近20日有一定兑现压力，冲高后还要继续看承接")
        elif overhead_density_tag == "兑现压力大":
            failed.append("最近20日兑现压力偏大，强势容易被上方抛压吃掉")
    if moneyflow_tag:
        passed.append(f"资金确认分={moneyflow_score:.4f}，主池资金排序第 {moneyflow_rank} 名")
        if moneyflow_detail:
            passed.append(moneyflow_detail)
        if moneyflow_tag == "资金确认强":
            passed.append("资金配合较强，不只是价格自己走高")
        elif moneyflow_tag == "资金确认一般":
            passed.append("有一定资金配合，但不是最强抢筹")
        elif moneyflow_tag == "资金确认弱":
            failed.append("资金确认偏弱，当前更像价格先行")
    chip_tag = str(item.get("chip_overhead_tag") or "").strip()
    chip_detail = str(item.get("chip_overhead_detail") or "").strip()
    if chip_tag:
        if chip_detail:
            passed.append(chip_detail)
        if chip_tag == "筹码压力轻":
            passed.append("上方套牢盘较轻，冲高时阻力更小")
        elif chip_tag == "筹码结构中性":
            passed.append("筹码结构一般，向上空间还要继续看量")
        elif chip_tag == "筹码压力偏大":
            failed.append("上方筹码压力偏大，冲高容易受压")
    fundamental_tag = str(item.get("fundamental_context_tag") or "").strip()
    fundamental_detail = str(item.get("fundamental_context_detail") or "").strip()
    if fundamental_tag:
        if fundamental_detail:
            passed.append(fundamental_detail)
        if fundamental_tag == "公告/预警偏正面":
            passed.append("最近公告/预告偏正面，解释层更完整")
        elif fundamental_tag == "公告/预警偏弱":
            failed.append("最近公告/预警偏弱，需要更谨慎控仓")

    if not plus_lines and neutral_lines:
        plus_lines.append(neutral_lines[0])
    if not minus_lines:
        if neutral_lines:
            minus_lines.append(neutral_lines[-1])
        else:
            minus_lines.append("当前展示因子里没有明显减分项")

    logic_parts: list[str] = []
    if positive:
        top = positive[0]
        logic_parts.append(f"{_factor_label(top[0])}是主加分项")
    if len(positive) > 1:
        logic_parts.append(f"{_factor_label(positive[1][0])}也在抬分")
    if negative:
        logic_parts.append(f"{_factor_label(negative[0][0])}是主要拖累")
    if ml_tag:
        if "ml_score低" in ml_tag:
            logic_parts.append("适合当主策略候选观察，不算双确认票")
        elif "ml_score高" in ml_tag:
            logic_parts.append("属于主策略和 ML 共振票，可以优先观察")
        elif "异常高" in ml_tag:
            logic_parts.append("主分一般，但短线概率分有抬头")
    if breadth_tag == "板块共振强":
        logic_parts.append("所属行业是一起走强，不只是单票冲高")
    elif breadth_tag == "板块确认一般":
        logic_parts.append("板块有配合，但强度中等")
    elif breadth_tag == "板块确认弱":
        logic_parts.append("更偏个股强势，板块确认不够")
    if leader_follow_tag == "龙头扩散强":
        logic_parts.append("行业龙头已经先走出来，这只票更像二次扩散跟随")
    elif leader_follow_tag == "龙头扩散一般":
        logic_parts.append("有龙头带动迹象，但扩散还在半路")
    elif leader_follow_tag == "龙头扩散弱":
        logic_parts.append("缺少龙头带动的扩散确认，更像单票自发走强")
    if overhead_density_tag == "兑现压力轻":
        logic_parts.append("最近20日兑现压力轻，冲高后利润保留难度相对更低")
    elif overhead_density_tag == "兑现压力一般":
        logic_parts.append("上方兑现盘不算轻，继续往上要看承接质量")
    elif overhead_density_tag == "兑现压力大":
        logic_parts.append("最近20日上方兑现压力大，强势容易被抛压吃掉")
    if moneyflow_tag == "资金确认强":
        logic_parts.append("资金面也在配合，短线承接更扎实")
    elif moneyflow_tag == "资金确认一般":
        logic_parts.append("资金面有跟随，但不是最强确认")
    elif moneyflow_tag == "资金确认弱":
        logic_parts.append("资金面没跟上，追高要更谨慎")
    chip_tag = str(item.get("chip_overhead_tag") or "").strip()
    if chip_tag == "筹码压力轻":
        logic_parts.append("上方筹码压力轻，冲高阻力相对更小")
    elif chip_tag == "筹码结构中性":
        logic_parts.append("筹码结构一般，往上还要继续看量")
    elif chip_tag == "筹码压力偏大":
        logic_parts.append("上方筹码压力偏大，冲高后更容易受压")
    fundamental_tag = str(item.get("fundamental_context_tag") or "").strip()
    if fundamental_tag == "公告/预警偏正面":
        logic_parts.append("最近公告偏正面，基本解释层更完整")
    elif fundamental_tag == "公告/预警偏弱":
        logic_parts.append("最近公告/预警偏弱，仓位要更克制")

    return {
        "plus": plus_lines,
        "minus": minus_lines,
        "passed": passed,
        "failed": failed or ["没有明显未满足项，但仍需结合风控和仓位"],
        "logic": "；".join(logic_parts) if logic_parts else "当前主要是相对排序入选，建议结合行业环境继续确认。",
    }


def _detail_block(title: str, lines: list[str], color: str) -> str:
    body = "".join(
        f'<li style="margin:4px 0;">{html.escape(line)}</li>'
        for line in (lines or ["-"])
    )
    return (
        '<div style="margin-top:10px;padding:10px 12px;border-radius:12px;'
        f'background:{color};">'
        f'<div style="color:#ffffff;font-size:13px;font-weight:800;margin-bottom:6px;">{html.escape(title)}</div>'
        f'<ul style="margin:0;padding-left:18px;color:#d8e3ff;font-size:12px;line-height:1.6;">{body}</ul>'
        '</div>'
    )


def _moneyflow_block(item: dict[str, Any]) -> str:
    moneyflow_tag = str(item.get("moneyflow_confirmation_tag") or "").strip()
    if not moneyflow_tag:
        return ""
    detail = str(item.get("moneyflow_confirmation_detail") or "").strip()
    lines = [detail] if detail else [f"资金确认标签：{moneyflow_tag}"]
    color = "#173126" if moneyflow_tag == "资金确认强" else "#1c2940" if moneyflow_tag == "资金确认一般" else "#3a2f17"
    return _detail_block("资金确认", lines, color)


def _leader_follow_block(item: dict[str, Any]) -> str:
    tag = str(item.get("industry_leader_follow_tag") or "").strip()
    if not tag:
        return ""
    detail = str(item.get("industry_leader_follow_detail") or "").strip()
    lines = [detail] if detail else [f"龙头扩散标签：{tag}"]
    color = "#173126" if tag == "龙头扩散强" else "#1c2940" if tag == "龙头扩散一般" else "#3a2f17"
    return _detail_block("龙头扩散", lines, color)


def _overhead_density_block(item: dict[str, Any]) -> str:
    tag = str(item.get("overhead_density_tag") or "").strip()
    if not tag:
        return ""
    detail = str(item.get("overhead_density_detail") or "").strip()
    lines = [detail] if detail else [f"兑现压力标签：{tag}"]
    color = "#173126" if tag == "兑现压力轻" else "#1c2940" if tag == "兑现压力一般" else "#3a2f17"
    return _detail_block("兑现压力", lines, color)


def _chip_block(item: dict[str, Any]) -> str:
    chip_tag = str(item.get("chip_overhead_tag") or "").strip()
    if not chip_tag:
        return ""
    detail = str(item.get("chip_overhead_detail") or "").strip()
    lines = [detail] if detail else [f"筹码状态：{chip_tag}"]
    color = "#173126" if chip_tag == "筹码压力轻" else "#1c2940" if chip_tag == "筹码结构中性" else "#3a2f17"
    return _detail_block("筹码/压力", lines, color)


def _fundamental_block(item: dict[str, Any]) -> str:
    tag = str(item.get("fundamental_context_tag") or "").strip()
    if not tag:
        return ""
    detail = str(item.get("fundamental_context_detail") or "").strip()
    lines = [detail] if detail else [f"公告状态：{tag}"]
    color = "#173126" if tag == "公告/预警偏正面" else "#1c2940" if tag == "公告/预警中性" else "#342225"
    return _detail_block("公告/预警", lines, color)


def _stable_cycle_summary_card(main_payload: dict[str, Any]) -> str:
    research_summary = (main_payload or {}).get("research_summary", {}) or {}
    lines = [
        f"system_mode：<b>{html.escape(str(research_summary.get('system_mode') or 'stable_observation_cycle'))}</b>",
        f"主分冻结：<b>{html.escape(str(research_summary.get('main_score_frozen', True)))}</b>；执行层冻结：<b>{html.escape(str(research_summary.get('execution_layer_frozen', True)))}</b>；观察层冻结：<b>{html.escape(str(research_summary.get('observation_layer_frozen', True)))}</b>",
        f"新增研究门槛：<b>{html.escape(str(research_summary.get('new_research_gate_passed', False)))}</b>；{html.escape(str(research_summary.get('new_research_gate_reason') or 'does not improve candidate pool quality or action clarity'))}",
    ]
    return _summary_card("当前运行状态", lines)


def _review_question_block(item: dict[str, Any]) -> str:
    review_pack = build_review_question_pack(item)
    return _detail_block(
        "复盘三问",
        [
            f"它是因为主分高被选中，还是因为观察层更值得关注？{review_pack['selected_because']}",
            f"它属于哪种观察层组合？{review_pack['label_combo']}",
            f"最后没做好，是排序问题还是兑现路径问题？{review_pack['failure_attribution']}",
        ],
        "#141a2a",
    )


def _annotate_main_cards_moneyflow(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not cards:
        return cards
    if any(str(item.get("moneyflow_confirmation_tag") or "").strip() for item in cards):
        return cards

    scored = []
    for idx, item in enumerate(cards):
        value = _safe_float(item.get("smart_money_inflow_20_neu"))
        scored.append((idx, value))
    ranks = {
        idx: rank
        for rank, (idx, _value) in enumerate(sorted(scored, key=lambda row: row[1], reverse=True), start=1)
    }

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(cards):
        row = dict(item)
        value = _safe_float(row.get("smart_money_inflow_20_neu"))
        rank = int(ranks.get(idx, 0))
        if value > 0 and rank <= 5:
            tag = "资金确认强"
        elif value > 0 and rank <= 10:
            tag = "资金确认一般"
        else:
            tag = "资金确认弱"
        row["moneyflow_confirmation_score"] = value
        row["moneyflow_confirmation_rank"] = rank
        row["moneyflow_confirmation_tag"] = tag
        row["moneyflow_confirmation_detail"] = (
            f"20日聪明钱流入={value:.4f}，主池资金排序第 {rank} 名，{tag}"
        )
        out.append(row)
    return out


def _annotate_main_cards_premium(cards: list[dict[str, Any]], selection_date: str) -> list[dict[str, Any]]:
    if not cards or not selection_date:
        return cards
    if all(
        str(item.get("chip_overhead_tag") or "").strip() or str(item.get("fundamental_context_tag") or "").strip()
        for item in cards
    ):
        return cards
    try:
        import pandas as pd

        frame = pd.DataFrame(cards)
        enriched = _annotate_premium_observation_layers(frame, pd.Timestamp(selection_date))
        return json.loads(enriched.to_json(orient="records", date_format="iso", force_ascii=False))
    except Exception:
        return cards


def _industry_counter(items: list[dict[str, Any]], limit: int) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in items[:limit]:
        industry = str(item.get("industry") or "未知")
        counter[industry] += 1
    return counter


def _format_top_industries(counter: Counter[str], top_n: int = 3) -> str:
    if not counter:
        return "暂无"
    total = sum(counter.values()) or 1
    parts = []
    for industry, count in counter.most_common(top_n):
        parts.append(f"{html.escape(industry)} {count}只({count / total:.0%})")
    return " / ".join(parts)


def _build_market_summary_lines(
    main_summary: dict[str, Any],
    elastic_summary: dict[str, Any],
    short_cards: list[dict[str, Any]],
    risk_payload: dict[str, Any],
) -> list[str]:
    risk_status = str(risk_payload.get("status") or risk_payload.get("risk_status") or "N/A").upper()
    if risk_status == "PASS":
        stance = "环境偏顺风，可优先看主策略，再小仓试弹性。"
    elif risk_status == "WARN":
        stance = "环境中性偏谨慎，主策略优先，弹性池降仓。"
    elif risk_status == "BLOCK":
        stance = "环境偏逆风，晚报以观察和复盘为主，不宜激进追高。"
    else:
        stance = "风控状态未刷新，建议先观察后执行。"

    total_fd_amount = sum(_safe_float(item.get("fd_amount")) for item in short_cards)
    avg_turnover = (
        sum(_safe_float(item.get("turnover_ratio")) for item in short_cards) / max(len(short_cards), 1)
        if short_cards
        else 0.0
    )
    max_streak = max((_safe_int(item.get("current_streak")) for item in short_cards), default=0)
    return [
        f"风控状态：<b>{html.escape(risk_status)}</b>；动作建议：<b>{html.escape(str(risk_payload.get('action_hint', 'review only')))}</b>",
        f"主策略候选 <b>{_safe_int(main_summary.get('selected_count'))}</b> 只；低价弹性池候选 <b>{_safe_int(elastic_summary.get('selected_count'))}</b> 只；短线卡片 <b>{len(short_cards)}</b> 只。",
        f"短线情绪：前五封单合计 <b>{_fmt_money_yi(total_fd_amount)}</b>，平均换手 <b>{avg_turnover:.2f}%</b>，最高连板 <b>{max_streak}</b> 板。",
        f"环境结论：<b>{stance}</b>",
    ]


def _build_sector_summary_lines(
    main_cards: list[dict[str, Any]],
    elastic_cards: list[dict[str, Any]],
    short_cards: list[dict[str, Any]],
) -> list[str]:
    main_counter = _industry_counter(main_cards, limit=5)
    elastic_counter = _industry_counter(elastic_cards, limit=5)
    short_counter = _industry_counter(short_cards, limit=5)
    overlap = [name for name, _ in (main_counter & elastic_counter).most_common(3)]
    overlap_text = " / ".join(html.escape(x) for x in overlap) if overlap else "暂无明显重叠主线"
    return [
        f"主策略前五集中：<b>{_format_top_industries(main_counter)}</b>",
        f"低价弹性前五集中：<b>{_format_top_industries(elastic_counter)}</b>",
        f"短线前五集中：<b>{_format_top_industries(short_counter)}</b>",
        f"共振主线：<b>{overlap_text}</b>",
    ]


def _main_card(item: dict[str, Any], factor_weights: dict[str, Any], *, compact: bool = False) -> str:
    ml_tag = str(item.get("ml_observation_tag") or "").strip()
    ml_meta = ""
    if ml_tag:
        ml_meta = (
            f'<div style="margin-top:10px;color:#ffd866;font-size:13px;line-height:1.6;">'
            f'观察标签：<b>{html.escape(ml_tag)}</b>'
            f'；ml_score：<b>{_safe_float(item.get("ml_score")):.4f}</b>'
            f'；ml_rank：<b>{_safe_int(item.get("ml_rank"))}</b></div>'
        )
    reason_pack = _score_reason_pack(item, factor_weights)
    if compact:
        return (
            '<div style="background:#171d2e;border:1px solid #2b3b5a;border-radius:18px;padding:16px;'
            'margin:10px 0;box-shadow:0 8px 22px rgba(0,0,0,0.2);">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<div style="font-size:18px;font-weight:800;color:#ffffff;">{html.escape(str(item.get("name", "-")))}</div>'
            f'{_badge("#" + str(item.get("rank", "-")), "#2f7df6")}</div>'
            f'<div style="color:#8ea2c8;font-size:13px;margin-bottom:10px;">{html.escape(str(item.get("code", "-")))} · {html.escape(str(item.get("industry", "-")))}</div>'
            f'<div style="color:#d8e3ff;font-size:13px;line-height:1.7;">收盘价 <b>{_safe_float(item.get("close")):.2f}</b>；评分 <b>{_safe_float(item.get("score")):.4f}</b>；目标权重 <b>{_safe_float(item.get("target_weight"))*100:.2f}%</b>；市值 <b>{_fmt_money_yi(item.get("market_cap"))}</b></div>'
            f'<div style="margin-top:8px;color:#ffd866;font-size:12px;line-height:1.6;">观察层：{html.escape(_observation_summary(item))}</div>'
            f'<div style="margin-top:8px;color:#7ee787;font-size:12px;line-height:1.6;">主加分项：{html.escape(_compact_line(list(reason_pack["plus"])))}</div>'
            f'<div style="margin-top:4px;color:#ff9b9b;font-size:12px;line-height:1.6;">主要隐患：{html.escape(_compact_line(list(reason_pack["minus"])))}</div>'
            f'<div style="margin-top:8px;color:#d8e3ff;font-size:12px;line-height:1.7;">交易逻辑：{html.escape(str(reason_pack["logic"]))}</div>'
            '</div>'
        )
    return (
        '<div style="background:#171d2e;border:1px solid #2b3b5a;border-radius:18px;padding:16px;'
        'margin:10px 0;box-shadow:0 8px 22px rgba(0,0,0,0.2);">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<div style="font-size:18px;font-weight:800;color:#ffffff;">{html.escape(str(item.get("name", "-")))}</div>'
        f'{_badge("#" + str(item.get("rank", "-")), "#2f7df6")}</div>'
        f'<div style="color:#8ea2c8;font-size:13px;margin-bottom:10px;">{html.escape(str(item.get("code", "-")))} · {html.escape(str(item.get("industry", "-")))}</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px 16px;">'
        f'<div><div style="color:#8ea2c8;font-size:12px;">收盘价</div><div style="color:#ff7575;font-size:22px;font-weight:800;">{_safe_float(item.get("close")):.2f}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">评分</div><div style="color:#7ee787;font-size:22px;font-weight:800;">{_safe_float(item.get("score")):.4f}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">目标权重</div><div style="color:#d8e3ff;font-size:16px;font-weight:700;">{_safe_float(item.get("target_weight"))*100:.2f}%</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">市值</div><div style="color:#d8e3ff;font-size:16px;font-weight:700;">{_fmt_money_yi(item.get("market_cap"))}</div></div>'
        f'</div>{ml_meta}'
        f'{_leader_follow_block(item)}'
        f'{_overhead_density_block(item)}'
        f'{_moneyflow_block(item)}'
        f'{_chip_block(item)}'
        f'{_fundamental_block(item)}'
        f'{_detail_block("加分项", list(reason_pack["plus"]), "#173126")}'
        f'{_detail_block("减分项", list(reason_pack["minus"]), "#342225")}'
        f'{_detail_block("通过条件", list(reason_pack["passed"]), "#1c2940")}'
        f'{_detail_block("未满足条件", list(reason_pack["failed"]), "#3a2f17")}'
        f'{_review_question_block(item)}'
        f'<div style="margin-top:10px;padding:10px 12px;border-radius:12px;background:#141a2a;">'
        f'<div style="color:#ffffff;font-size:13px;font-weight:800;margin-bottom:6px;">交易逻辑</div>'
        f'<div style="color:#d8e3ff;font-size:12px;line-height:1.7;">{html.escape(str(reason_pack["logic"]))}</div>'
        '</div></div>'
    )


def _elastic_card(item: dict[str, Any], *, compact: bool = False) -> str:
    if compact:
        return (
            '<div style="background:#171d2e;border:1px solid #32482f;border-radius:18px;padding:16px;'
            'margin:10px 0;box-shadow:0 8px 22px rgba(0,0,0,0.2);">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<div style="font-size:18px;font-weight:800;color:#ffffff;">{html.escape(str(item.get("name", "-")))}</div>'
            f'{_badge("#" + str(item.get("rank", "-")), "#1ea362")}</div>'
            f'<div style="color:#8ea2c8;font-size:13px;line-height:1.6;">{html.escape(str(item.get("code", "-")))} · {html.escape(str(item.get("industry", "-")))}</div>'
            f'<div style="margin-top:8px;color:#d8e3ff;font-size:12px;line-height:1.7;">收盘价 <b>{_safe_float(item.get("close")):.2f}</b>；评分 <b>{_safe_float(item.get("score")):.4f}</b>；成交额 <b>{_fmt_money_yi(item.get("amount"))}</b>；市值 <b>{_fmt_money_yi(item.get("market_cap"))}</b></div>'
            '</div>'
        )
    return (
        '<div style="background:#171d2e;border:1px solid #32482f;border-radius:18px;padding:16px;'
        'margin:10px 0;box-shadow:0 8px 22px rgba(0,0,0,0.2);">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<div style="font-size:18px;font-weight:800;color:#ffffff;">{html.escape(str(item.get("name", "-")))}</div>'
        f'{_badge("#" + str(item.get("rank", "-")), "#1ea362")}</div>'
        f'<div style="color:#8ea2c8;font-size:13px;margin-bottom:10px;">{html.escape(str(item.get("code", "-")))} · {html.escape(str(item.get("industry", "-")))}</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px 16px;">'
        f'<div><div style="color:#8ea2c8;font-size:12px;">收盘价</div><div style="color:#ffb86b;font-size:22px;font-weight:800;">{_safe_float(item.get("close")):.2f}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">评分</div><div style="color:#7ee787;font-size:22px;font-weight:800;">{_safe_float(item.get("score")):.4f}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">成交额</div><div style="color:#d8e3ff;font-size:16px;font-weight:700;">{_fmt_money_yi(item.get("amount"))}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">市值</div><div style="color:#d8e3ff;font-size:16px;font-weight:700;">{_fmt_money_yi(item.get("market_cap"))}</div></div>'
        '</div></div>'
    )


def _shortline_card(item: dict[str, Any], *, compact: bool = False) -> str:
    if compact:
        return (
            '<div style="background:#171d2e;border:1px solid #4a3526;border-radius:18px;padding:16px;'
            'margin:10px 0;box-shadow:0 8px 22px rgba(0,0,0,0.2);">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<div style="font-size:18px;font-weight:800;color:#ffffff;">{html.escape(str(item.get("name", "-")))}</div>'
            f'{_badge("#" + str(item.get("rank", "-")), "#ef7d3c")}</div>'
            f'<div style="color:#8ea2c8;font-size:13px;line-height:1.6;">{html.escape(str(item.get("code", "-")))} · {html.escape(str(item.get("industry", "-")))}</div>'
            f'<div style="margin-top:8px;color:#d8e3ff;font-size:12px;line-height:1.7;">评分 <b>{_safe_float(item.get("shortline_score")):.4f}</b>；当前连板 <b>{_safe_int(item.get("current_streak"))}</b>；买入区间 <b>{html.escape(str(item.get("buy_range", "-")))}</b></div>'
            f'<div style="margin-top:4px;color:#ffd866;font-size:12px;line-height:1.6;">目标价 <b>{html.escape(str(item.get("target_price", "-")))}</b>；止损线 <b>{html.escape(str(item.get("stop_loss", "-")))}</b></div>'
            '</div>'
        )
    return (
        '<div style="background:#171d2e;border:1px solid #4a3526;border-radius:18px;padding:16px;'
        'margin:10px 0;box-shadow:0 8px 22px rgba(0,0,0,0.2);">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<div style="font-size:18px;font-weight:800;color:#ffffff;">{html.escape(str(item.get("name", "-")))}</div>'
        f'{_badge("#" + str(item.get("rank", "-")), "#ef7d3c")}</div>'
        f'<div style="color:#8ea2c8;font-size:13px;margin-bottom:10px;">{html.escape(str(item.get("code", "-")))} · {html.escape(str(item.get("industry", "-")))}</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px 16px;">'
        f'<div><div style="color:#8ea2c8;font-size:12px;">收盘价</div><div style="color:#ff7575;font-size:20px;font-weight:800;">{_safe_float(item.get("close")):.2f}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">评分</div><div style="color:#7ee787;font-size:20px;font-weight:800;">{_safe_float(item.get("shortline_score")):.4f}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">当前连板</div><div style="color:#ffd866;font-size:20px;font-weight:800;">{_safe_int(item.get("current_streak"))}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">近月连板</div><div style="color:#d8e3ff;font-size:15px;font-weight:700;">{_safe_int(item.get("limit_up_count_20d"))}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">炸板次数</div><div style="color:#d8e3ff;font-size:15px;font-weight:700;">{_safe_int(item.get("open_board_count_20d"))}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">换手率</div><div style="color:#d8e3ff;font-size:15px;font-weight:700;">{_safe_float(item.get("turnover_ratio")):.2f}%</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">封单金额</div><div style="color:#d8e3ff;font-size:15px;font-weight:700;">{_fmt_money_yi(item.get("fd_amount"))}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">流通市值</div><div style="color:#d8e3ff;font-size:15px;font-weight:700;">{_fmt_money_yi(item.get("float_mv"))}</div></div>'
        f'<div><div style="color:#8ea2c8;font-size:12px;">档位</div><div style="color:#d8e3ff;font-size:15px;font-weight:700;">{html.escape(str(item.get("grade", "-")))} / {html.escape(str(item.get("style", "-")))}</div></div>'
        '</div>'
        f'<div style="margin-top:10px;color:#c7d4ee;font-size:13px;line-height:1.6;">近4日走势：<b>{html.escape(str(item.get("recent_4d_path", "-")))}</b></div>'
        f'<div style="margin-top:8px;color:#7ee787;font-size:14px;line-height:1.6;">明日买入区间：<b>{html.escape(str(item.get("buy_range", "-")))}</b></div>'
        f'<div style="margin-top:4px;color:#ffd866;font-size:14px;line-height:1.6;">目标价：<b>{html.escape(str(item.get("target_price", "-")))}</b></div>'
        f'<div style="margin-top:4px;color:#ff7575;font-size:14px;line-height:1.6;">止损线：<b>{html.escape(str(item.get("stop_loss", "-")))}</b></div>'
        f'<div style="margin-top:4px;color:#8be9fd;font-size:14px;line-height:1.6;">建议仓位：<b>{html.escape(str(item.get("suggested_position", "-")))}</b></div>'
        f'<div style="margin-top:8px;color:#d8e3ff;font-size:13px;line-height:1.7;">{html.escape(str(item.get("operation_logic", "-")))}</div>'
        '</div>'
    )


def _render_html(
    main_payload: dict[str, Any],
    elastic_payload: dict[str, Any],
    shortline_payload: dict[str, Any],
    risk_payload: dict[str, Any],
    *,
    main_limit: int = 5,
    elastic_limit: int = 5,
    shortline_limit: int = 5,
    compact: bool = False,
) -> tuple[str, str]:
    main_summary = main_payload.get("summary", {})
    main_factor_weights = main_payload.get("factor_weights", {})
    elastic_summary = elastic_payload.get("summary", {})
    short_summary = shortline_payload.get("summary", {})
    selection_date = str(short_summary.get("selection_date") or elastic_summary.get("selection_date") or main_summary.get("selection_date") or "")
    title = f"量化三合一晚报 {selection_date}".strip()

    main_cards = _annotate_main_cards_moneyflow((main_payload.get("observation_pool", [])[:main_limit]))
    main_cards = _annotate_main_cards_premium(main_cards, str(main_summary.get("selection_date") or ""))
    elastic_cards = elastic_payload.get("observation_pool", [])[:elastic_limit]
    short_cards = shortline_payload.get("cards", [])[:shortline_limit]
    risk_status = str(risk_payload.get("status") or risk_payload.get("risk_status") or "n/a").upper()
    action_hint = str(risk_payload.get("action_hint", "review only"))
    reasons = risk_payload.get("reasons", []) or []
    market_lines = _build_market_summary_lines(main_summary, elastic_summary, short_cards, risk_payload)
    sector_lines = _build_sector_summary_lines(main_cards, elastic_cards, short_cards)

    html_body = [
        '<div style="background:#0b1020;padding:18px 14px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">',
        '<div style="max-width:900px;margin:0 auto;">',
        '<div style="background:linear-gradient(135deg,#16213f,#0f172a);border-radius:22px;padding:22px 20px;margin-bottom:16px;border:1px solid #2b3b5a;">',
        '<div style="font-size:24px;font-weight:900;color:#fff;">量化三合一晚报</div>',
        f'<div style="margin-top:8px;color:#a8b8d8;font-size:14px;">选股日期：{html.escape(selection_date)} · 风控状态：{html.escape(risk_status)} · 动作建议：{html.escape(action_hint)}</div>',
        '</div>',
    ]
    html_body.append(_summary_card("大盘环境摘要", market_lines))
    html_body.append(_summary_card("板块分析", sector_lines))
    html_body.append(_stable_cycle_summary_card(main_payload))
    if reasons:
        html_body.append(_summary_card("风控摘要", [html.escape(str(item)) for item in reasons[:3]]))

    html_body.append(_summary_card("主策略（稳健候选池）", [
        f"selection_date：<b>{html.escape(str(main_summary.get('selection_date', '-')))}</b>",
        f"selected_count：<b>{_safe_int(main_summary.get('selected_count'))}</b>",
        f"说明：偏中低频、低换手、可长期跟踪。当前展示前 <b>{len(main_cards)}</b> 只。",
    ]))
    html_body.extend(_main_card(item, main_factor_weights, compact=compact) for item in main_cards)

    html_body.append(_summary_card("20元以下弹性池", [
        f"selection_date：<b>{html.escape(str(elastic_summary.get('selection_date', '-')))}</b>",
        f"selected_count：<b>{_safe_int(elastic_summary.get('selected_count'))}</b>",
        f"说明：高风险高收益，只适合小仓试错。当前展示前 <b>{len(elastic_cards)}</b> 只。",
    ]))
    html_body.extend(_elastic_card(item, compact=compact) for item in elastic_cards)

    html_body.append(_summary_card("精选短线机会（前五名）", [
        f"selection_date：<b>{html.escape(str(short_summary.get('selection_date', '-')))}</b>",
        f"selected_count：<b>{_safe_int(short_summary.get('selected_count'))}</b>",
        f"说明：按 shortline_score 排序。当前展示前 <b>{len(short_cards)}</b> 只。",
    ]))
    html_body.extend(_shortline_card(item, compact=compact) for item in short_cards)
    html_body.append('</div></div>')
    return title, "".join(html_body)


def _render_summary_fallback(
    main_payload: dict[str, Any],
    elastic_payload: dict[str, Any],
    shortline_payload: dict[str, Any],
    risk_payload: dict[str, Any],
) -> tuple[str, str]:
    main_summary = main_payload.get("summary", {})
    elastic_summary = elastic_payload.get("summary", {})
    short_summary = shortline_payload.get("summary", {})
    selection_date = str(short_summary.get("selection_date") or elastic_summary.get("selection_date") or main_summary.get("selection_date") or "")
    title = f"量化三合一晚报 {selection_date}".strip()
    main_cards = _annotate_main_cards_moneyflow((main_payload.get("observation_pool", [])[:3]))
    main_cards = _annotate_main_cards_premium(main_cards, str(main_summary.get("selection_date") or ""))
    elastic_cards = elastic_payload.get("observation_pool", [])[:3]
    short_cards = shortline_payload.get("cards", [])[:3]
    risk_status = str(risk_payload.get("status") or risk_payload.get("risk_status") or "n/a").upper()
    action_hint = str(risk_payload.get("action_hint", "review only"))
    body = [
        '<div style="background:#0b1020;padding:18px 14px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">',
        '<div style="max-width:900px;margin:0 auto;">',
        '<div style="background:linear-gradient(135deg,#16213f,#0f172a);border-radius:22px;padding:22px 20px;margin-bottom:16px;border:1px solid #2b3b5a;">',
        '<div style="font-size:24px;font-weight:900;color:#fff;">量化三合一晚报</div>',
        f'<div style="margin-top:8px;color:#a8b8d8;font-size:14px;">选股日期：{html.escape(selection_date)} · 风控状态：{html.escape(risk_status)} · 动作建议：{html.escape(action_hint)}</div>',
        '</div>',
        _summary_card("主策略摘要", [
            f"主策略前列：<b>{_top_names_line(main_cards, score_key='score', limit=3)}</b>",
            f"观察层重点：<b>{html.escape(_observation_summary(main_cards[0]) if main_cards else '暂无')}</b>",
        ]),
        _summary_card("弹性池摘要", [
            f"弹性池前列：<b>{_top_names_line(elastic_cards, score_key='score', limit=3)}</b>",
        ]),
        _summary_card("短线摘要", [
            f"短线前列：<b>{_top_names_line(short_cards, score_key='shortline_score', limit=3)}</b>",
        ]),
        _summary_card("发送说明", [
            "本次自动启用了精简版晚报，以保证 PushPlus 在字数上限内稳定送达。",
            "完整素材仍保留在云端输出目录，可继续用于复盘和归档。",
        ]),
        '</div></div>',
    ]
    return title, "".join(body)


def _render_within_pushplus_limit(
    main_payload: dict[str, Any],
    elastic_payload: dict[str, Any],
    shortline_payload: dict[str, Any],
    risk_payload: dict[str, Any],
    *,
    content_limit: int,
) -> tuple[str, str, str, list[dict[str, Any]]]:
    profiles = [
        {"name": "full", "main_limit": 5, "elastic_limit": 5, "shortline_limit": 5, "compact": False},
        {"name": "balanced", "main_limit": 4, "elastic_limit": 3, "shortline_limit": 3, "compact": True},
        {"name": "compact", "main_limit": 3, "elastic_limit": 2, "shortline_limit": 2, "compact": True},
    ]
    attempts: list[dict[str, Any]] = []
    for profile in profiles:
        title, content = _render_html(
            main_payload,
            elastic_payload,
            shortline_payload,
            risk_payload,
            main_limit=int(profile["main_limit"]),
            elastic_limit=int(profile["elastic_limit"]),
            shortline_limit=int(profile["shortline_limit"]),
            compact=bool(profile["compact"]),
        )
        attempts.append({"profile": profile["name"], "content_length": len(content)})
        if len(content) <= content_limit:
            return title, content, str(profile["name"]), attempts
    title, content = _render_summary_fallback(main_payload, elastic_payload, shortline_payload, risk_payload)
    attempts.append({"profile": "summary", "content_length": len(content)})
    return title, content, "summary", attempts


def _pushplus_content_too_large(result: dict[str, Any]) -> bool:
    text = " ".join(str(result.get(key) or "") for key in ("msg", "data", "raw_text"))
    return any(marker in text for marker in PUSHPLUS_OVERSIZE_MARKERS)


def _post_pushplus(token: str, title: str, content: str, template: str, topic: str, channel: str, timeout_seconds: float) -> dict[str, Any]:
    payload = {
        "token": token,
        "title": title,
        "content": content,
        "template": template,
        "channel": channel,
    }
    if topic:
        payload["topic"] = topic
    response = httpx.post("https://www.pushplus.plus/send", json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return {"raw_text": response.text}


def main() -> None:
    args = build_parser().parse_args()

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
    selection_date = str(
        shortline_payload.get("summary", {}).get("selection_date")
        or elastic_payload.get("summary", {}).get("selection_date")
        or main_payload.get("summary", {}).get("selection_date")
        or ""
    )
    selection_key = selection_date.replace("-", "")
    risk_patterns = [
        str(ROOT / "outputs" / "risk_governor" / "risk_gate_*.json"),
        str(ROOT / "outputs" / "risk_governor_direct" / "risk_gate_*.json"),
    ]
    risk_path = _match_date_path(risk_patterns, selection_key) or _latest_match(risk_patterns)
    risk_payload = _load_json(risk_path)
    expected_date_key = _read_tail_date(Path(args.master_data_path))
    main_date_key = _payload_date_key(main_payload, main_path)
    elastic_date_key = _payload_date_key(elastic_payload, elastic_path)
    shortline_date_key = _payload_date_key(shortline_payload, shortline_path)
    risk_date_key = _payload_date_key({"summary": {"selection_date": str(risk_payload.get("selection_date") or "")}}, risk_path)
    _validate_report_dates(
        expected_date_key=expected_date_key.replace("-", ""),
        main_date_key=main_date_key,
        elastic_date_key=elastic_date_key,
        shortline_date_key=shortline_date_key,
        risk_date_key=risk_date_key,
    )
    title, content, content_profile, render_attempts = _render_within_pushplus_limit(
        main_payload,
        elastic_payload,
        shortline_payload,
        risk_payload,
        content_limit=max(2000, int(args.content_limit)),
    )

    result = {
        "title": title,
        "template": args.template,
        "channel": args.channel,
        "main_path": str(main_path) if main_path else "",
        "elastic_path": str(elastic_path) if elastic_path else "",
        "shortline_path": str(shortline_path) if shortline_path else "",
        "risk_path": str(risk_path) if risk_path else "",
        "expected_date_key": expected_date_key.replace("-", ""),
        "main_date_key": main_date_key,
        "elastic_date_key": elastic_date_key,
        "shortline_date_key": shortline_date_key,
        "risk_date_key": risk_date_key,
        "content_limit": int(args.content_limit),
        "content_length": len(content),
        "content_profile": content_profile,
        "render_attempts": render_attempts,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        result["preview"] = content[:5000]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    token = str(args.token).strip()
    if not token:
        raise ValueError("PUSHPLUS_TOKEN is required.")
    pushplus_result = _post_pushplus(
        token=token,
        title=title,
        content=content,
        template=args.template,
        topic=str(args.topic).strip(),
        channel=str(args.channel).strip(),
        timeout_seconds=float(args.timeout_seconds),
    )
    if _pushplus_content_too_large(pushplus_result):
        retry_title, retry_content = _render_summary_fallback(
            main_payload,
            elastic_payload,
            shortline_payload,
            risk_payload,
        )
        retry_result = _post_pushplus(
            token=token,
            title=retry_title,
            content=retry_content,
            template=args.template,
            topic=str(args.topic).strip(),
            channel=str(args.channel).strip(),
            timeout_seconds=float(args.timeout_seconds),
        )
        result["pushplus_retry"] = {
            "reason": "content_too_large",
            "retry_profile": "summary",
            "retry_content_length": len(retry_content),
            "initial_pushplus_result": pushplus_result,
        }
        pushplus_result = retry_result
        result["content_profile"] = "summary"
        result["content_length"] = len(retry_content)
    result["pushplus_result"] = pushplus_result
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
