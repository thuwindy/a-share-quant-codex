from __future__ import annotations

import argparse
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
    latest_match,
    load_json,
    match_date_path,
    payload_date_key,
    resolve_llm_settings,
)


DEFAULT_CONTENT_LIMIT = 18_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM sidecar renderer for structured daily quant JSON.")
    parser.add_argument("--master-data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--main-monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_auto"))
    parser.add_argument("--elastic-monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_under20_elastic"))
    parser.add_argument("--shortline-dir", default=str(ROOT / "outputs" / "shortline_opportunities_live"))
    parser.add_argument("--risk-dir", default=str(ROOT / "outputs" / "risk_governor"))
    parser.add_argument("--candidate-quality-root", default=str(ROOT / "outputs"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "llm_report_renderer"))
    parser.add_argument("--date-key", default="")
    parser.add_argument("--top-main", type=int, default=10)
    parser.add_argument("--top-elastic", type=int, default=5)
    parser.add_argument("--top-shortline", type=int, default=5)
    parser.add_argument("--content-limit", type=int, default=DEFAULT_CONTENT_LIMIT)
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--no-llm", action="store_true")
    return parser


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pick_fields(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields if field in row and row.get(field) not in (None, "")}


def _compact_main_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    fields = [
        "rank",
        "code",
        "name",
        "industry",
        "close",
        "score",
        "target_weight",
        "ml_score",
        "ml_rank",
        "ml_observation_tag",
        "industry_leader_follow_score",
        "industry_leader_follow_rank",
        "industry_leader_follow_tag",
        "industry_leader_follow_detail",
        "overhead_density_score",
        "overhead_density_rank",
        "overhead_density_tag",
        "overhead_density_detail",
        "moneyflow_confirmation_tag",
        "chip_overhead_tag",
        "fundamental_context_tag",
    ]
    return [_pick_fields(row, fields) for row in rows[:limit]]


def _compact_elastic_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    fields = ["rank", "code", "name", "industry", "close", "score", "target_weight", "amount", "market_cap"]
    return [_pick_fields(row, fields) for row in rows[:limit]]


def _compact_shortline_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    fields = [
        "rank",
        "code",
        "name",
        "industry",
        "close",
        "shortline_score",
        "grade",
        "risk_level",
        "current_streak",
        "limit_up_count_20d",
        "open_board_count_20d",
        "turnover_ratio",
        "fd_amount",
        "recent_4d_path",
        "buy_range",
        "target_price",
        "stop_loss",
        "operation_logic",
    ]
    return [_pick_fields(row, fields) for row in rows[:limit]]


def _latest_candidate_quality(root: Path, date_key: str) -> dict[str, Any]:
    dated = root / f"candidate_pool_quality_dashboard_{date_key}" / "candidate_pool_quality_dashboard.json"
    if dated.exists():
        return load_json(dated)
    fallback = latest_match([str(root / "candidate_pool_quality_dashboard_*" / "candidate_pool_quality_dashboard.json")])
    return load_json(fallback)


def resolve_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    master_date = latest_master_date(Path(args.master_data_path))
    date_key = args.date_key or master_date.replace("-", "")
    paths: dict[str, Path | None] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, patterns in _artifact_patterns(args).items():
        paths[name] = match_date_path(patterns, date_key)
        payloads[name] = load_json(paths[name])
    quality = _latest_candidate_quality(Path(args.candidate_quality_root), date_key)
    return {"master_date": master_date, "date_key": date_key, "paths": paths, "payloads": payloads, "candidate_quality": quality}


def validate_report_context(context: dict[str, Any]) -> list[str]:
    date_key = str(context.get("date_key") or "")
    payloads: dict[str, dict[str, Any]] = context.get("payloads") or {}
    paths: dict[str, Path | None] = context.get("paths") or {}
    errors: list[str] = []
    for name in ("main", "elastic", "shortline", "risk"):
        if not paths.get(name):
            errors.append(f"missing_{name}_artifact")
            continue
        got = payload_date_key(payloads.get(name, {}), paths.get(name))
        if date_key and got != date_key:
            errors.append(f"{name}_date_mismatch:{got}!={date_key}")
    return errors


def build_structured_facts(context: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payloads: dict[str, dict[str, Any]] = context["payloads"]
    main = payloads.get("main", {})
    elastic = payloads.get("elastic", {})
    shortline = payloads.get("shortline", {})
    risk = payloads.get("risk", {})
    quality = context.get("candidate_quality") or {}
    main_rows = main.get("observation_pool") or main.get("strategy_snapshot") or []
    elastic_rows = elastic.get("observation_pool") or elastic.get("strategy_snapshot") or []
    short_rows = shortline.get("cards") or []
    return {
        "date_key": context.get("date_key"),
        "selection_date": (main.get("summary") or {}).get("selection_date") or risk.get("selection_date"),
        "validation": {"date_consistent": not validate_report_context(context)},
        "main_strategy": {
            "summary": main.get("summary", {}),
            "factor_weights": main.get("factor_weights", {}),
            "top_rows": _compact_main_rows(main_rows, args.top_main),
        },
        "elastic_pool": {
            "summary": elastic.get("summary", {}),
            "top_rows": _compact_elastic_rows(elastic_rows, args.top_elastic),
        },
        "shortline": {
            "summary": shortline.get("summary", {}),
            "cards": _compact_shortline_rows(short_rows, args.top_shortline),
        },
        "risk": {
            "selection_date": risk.get("selection_date", ""),
            "status": risk.get("status", ""),
            "action_hint": risk.get("action_hint", ""),
            "annual_return": risk.get("annual_return"),
            "sharpe": risk.get("sharpe"),
            "max_drawdown": risk.get("max_drawdown"),
            "avg_turnover": risk.get("avg_turnover"),
            "selected_count": risk.get("selected_count"),
            "checks": risk.get("checks", []),
            "reasons": risk.get("reasons", []),
        },
        "candidate_pool_quality": {
            "headline": quality.get("headline", ""),
            "keep_frozen": (quality.get("stable_observation_cycle_verdict") or {}).get("keep_frozen"),
            "summary": (quality.get("summary_rows") or quality.get("candidate_pool_quality") or [])[:12],
            "observation_tag_paths": (quality.get("observation_tag_paths") or [])[:12],
            "failure_attribution": (quality.get("failure_attribution") or [])[:12],
        },
    }


def render_deterministic_report(facts: dict[str, Any], *, reason: str = "") -> str:
    main_top = facts["main_strategy"]["top_rows"][:5]
    elastic_top = facts["elastic_pool"]["top_rows"][:3]
    short_top = facts["shortline"]["cards"][:3]
    risk = facts["risk"]

    def _names(rows: list[dict[str, Any]], score_key: str) -> str:
        parts = []
        for row in rows:
            name = row.get("name") or row.get("code") or "-"
            score = _safe_float(row.get(score_key))
            parts.append(f"{name}({score:.4f})")
        return " / ".join(parts) if parts else "暂无"

    lines = [
        f"# 量化日报解释 {facts.get('selection_date') or facts.get('date_key')}",
        "",
        "## 数据校验",
        f"- 日期一致：`{facts.get('validation', {}).get('date_consistent')}`",
        f"- 降级原因：`{reason or 'template_fallback'}`",
        "",
        "## 主策略",
        f"- 前排候选：{_names(main_top, 'score')}",
        "- 观察层重点：",
    ]
    for row in main_top:
        lines.append(
            f"  - {row.get('name') or row.get('code')}: "
            f"{row.get('industry_leader_follow_tag', '龙头扩散暂缺')}；"
            f"{row.get('overhead_density_tag', '兑现压力暂缺')}；"
            f"{row.get('ml_observation_tag', 'ML观察暂缺')}"
        )
    lines.extend(
        [
            "",
            "## 弹性池",
            f"- 前排候选：{_names(elastic_top, 'score')}",
            "",
            "## 精选短线机会",
            f"- 前排候选：{_names(short_top, 'shortline_score')}",
            "",
            "## 风控",
            f"- 状态：`{risk.get('status', '')}`；建议：`{risk.get('action_hint', '')}`；最大回撤：`{risk.get('max_drawdown')}`",
            "",
            "## 候选池质量",
            f"- 看板摘要：{facts.get('candidate_pool_quality', {}).get('headline') or '暂无候选池质量摘要'}",
            f"- keep_frozen：`{facts.get('candidate_pool_quality', {}).get('keep_frozen')}`",
        ]
    )
    return "\n".join(lines)


def _short_text(value: Any, limit: int = 80) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def build_llm_prompt_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """Keep the LLM input small; full facts are still saved to JSON for audit."""
    main_rows = facts.get("main_strategy", {}).get("top_rows", [])[:5]
    elastic_rows = facts.get("elastic_pool", {}).get("top_rows", [])[:3]
    short_rows = facts.get("shortline", {}).get("cards", [])[:3]
    return {
        "selection_date": facts.get("selection_date"),
        "validation": facts.get("validation"),
        "main_top": [
            {
                "rank": row.get("rank"),
                "code": row.get("code"),
                "name": row.get("name"),
                "industry": row.get("industry"),
                "score": row.get("score"),
                "target_weight": row.get("target_weight"),
                "ml_observation_tag": row.get("ml_observation_tag"),
                "industry_leader_follow_tag": row.get("industry_leader_follow_tag"),
                "industry_leader_follow_detail": _short_text(row.get("industry_leader_follow_detail")),
                "overhead_density_tag": row.get("overhead_density_tag"),
                "overhead_density_detail": _short_text(row.get("overhead_density_detail")),
            }
            for row in main_rows
        ],
        "elastic_top": [
            {
                "rank": row.get("rank"),
                "code": row.get("code"),
                "name": row.get("name"),
                "industry": row.get("industry"),
                "score": row.get("score"),
            }
            for row in elastic_rows
        ],
        "shortline_top": [
            {
                "rank": row.get("rank"),
                "code": row.get("code"),
                "name": row.get("name"),
                "shortline_score": row.get("shortline_score"),
                "risk_level": row.get("risk_level"),
                "operation_logic": _short_text(row.get("operation_logic")),
            }
            for row in short_rows
        ],
        "risk": {
            "status": facts.get("risk", {}).get("status"),
            "action_hint": facts.get("risk", {}).get("action_hint"),
            "max_drawdown": facts.get("risk", {}).get("max_drawdown"),
        },
        "candidate_pool_quality": {
            "headline": facts.get("candidate_pool_quality", {}).get("headline"),
            "keep_frozen": facts.get("candidate_pool_quality", {}).get("keep_frozen"),
        },
    }


def _numeric_tokens(text: str) -> set[str]:
    tokens = re.findall(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?%?", text)
    normalized: set[str] = set()
    for token in tokens:
        cleaned = token.rstrip("%")
        normalized.add(cleaned)
        if "." in cleaned:
            normalized.add(cleaned.rstrip("0").rstrip("."))
    return {item for item in normalized if item}


def _source_numeric_tokens(text: str) -> set[str]:
    tokens = _numeric_tokens(text)
    expanded = set(tokens)
    for token in tokens:
        try:
            value = float(token)
        except Exception:
            continue
        for digits in range(0, 7):
            rounded = f"{value:.{digits}f}"
            expanded.add(rounded)
            expanded.add(rounded.rstrip("0").rstrip("."))
    return {item for item in expanded if item}


def validate_numbers_grounded_in_json(markdown: str, facts: dict[str, Any]) -> list[str]:
    """Hard guard: every numeric token in LLM text must be present in the JSON facts."""
    source = compact_json(facts, limit=200_000)
    source_numbers = _source_numeric_tokens(source)
    output_numbers = _numeric_tokens(markdown)
    return sorted(token for token in output_numbers if token not in source_numbers)


def render_llm_report(facts: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    fallback = render_deterministic_report(facts)
    if args.no_llm:
        return fallback, "no_llm"
    prompt_facts = build_llm_prompt_facts(facts)
    fact_text = compact_json(prompt_facts, limit=min(args.content_limit, 6000))
    if len(fact_text) > args.content_limit:
        return render_deterministic_report(facts, reason="fact_payload_too_long"), "fallback_too_long"
    settings = resolve_llm_settings(api_key=args.llm_api_key, base_url=args.llm_base_url, model=args.llm_model)
    try:
        content = call_openai_compatible_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 A 股量化日报解释助手。只能使用用户 JSON 中出现的日期、股票、分数、收益、风控数字；"
                        "不得编造任何数字、排名或事件。输出中文 markdown，控制在 500 字以内。"
                        "结构包括：主策略、弹性池、短线、风控、候选池质量、观察层解释、今日使用建议。"
                        "如果信息不足，明确写“暂不可判定”。不要输出 JSON。"
                    ),
                },
                {"role": "user", "content": fact_text},
            ],
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            model=settings["model"],
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
        )
        if len(content) > args.content_limit:
            return render_deterministic_report(facts, reason="llm_output_too_long"), "fallback_output_too_long"
        ungrounded_numbers = validate_numbers_grounded_in_json(content, facts)
        if ungrounded_numbers:
            reason = "ungrounded_numbers:" + ",".join(ungrounded_numbers[:10])
            return render_deterministic_report(facts, reason=reason), "fallback_ungrounded_numbers"
        return content, "llm"
    except Exception as exc:
        return render_deterministic_report(facts, reason=str(exc)), "fallback"


def main() -> None:
    args = build_parser().parse_args()
    context = resolve_artifacts(args)
    validation_errors = validate_report_context(context)
    facts = build_structured_facts(context, args)
    if validation_errors:
        facts["validation"] = {"date_consistent": False, "errors": validation_errors}
        content = render_deterministic_report(facts, reason=";".join(validation_errors))
        render_mode = "fallback_validation_failed"
    else:
        content, render_mode = render_llm_report(facts, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_key = str(context.get("date_key") or "latest")
    json_path = output_dir / f"llm_report_{date_key}.json"
    md_path = output_dir / f"llm_report_{date_key}.md"
    payload = {
        "date_key": date_key,
        "render_mode": render_mode,
        "validation_errors": validation_errors,
        "facts": facts,
        "markdown": content,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(content, encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "markdown_path": str(md_path), "render_mode": render_mode, "validation_errors": validation_errors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
