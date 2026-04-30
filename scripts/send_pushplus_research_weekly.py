from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import os
from pathlib import Path
import re
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send weekly research summary.")
    parser.add_argument("--token", default=os.environ.get("PUSHPLUS_TOKEN", ""))
    parser.add_argument("--topic", default=os.environ.get("PUSHPLUS_TOPIC", ""))
    parser.add_argument("--channel", default=os.environ.get("PUSHPLUS_CHANNEL", "wechat"))
    parser.add_argument("--template", choices=("html", "markdown", "txt"), default="html")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _latest_path(patterns: list[str]) -> Path | None:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(item) for item in glob.glob(pattern, recursive=True))
    if not found:
        return None

    def _key(path: Path) -> tuple[str, str]:
        match = re.search(r"(20\d{6})", path.name)
        return (match.group(1) if match else "00000000", str(path))

    return sorted(found, key=_key)[-1]


def _latest_candidate_pool_dashboard() -> Path | None:
    found = sorted(
        ROOT.glob("outputs/candidate_pool_quality_dashboard_20??????/candidate_pool_quality_dashboard.json")
    )
    return found[-1] if found else None


def _summary_card(title: str, lines: list[str]) -> str:
    body = "".join(f'<div style="margin:6px 0;color:#d8e3ff;font-size:13px;line-height:1.6;">{line}</div>' for line in lines)
    return (
        '<div style="background:#141a2a;border:1px solid #26324a;border-radius:16px;'
        'padding:16px 18px;margin:12px 0;box-shadow:0 6px 20px rgba(0,0,0,0.18);">'
        f'<div style="font-size:16px;font-weight:800;color:#ffffff;margin-bottom:8px;">{html.escape(title)}</div>'
        f"{body}</div>"
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_csv_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _style_snapshot() -> tuple[list[str], str]:
    style_csv = _latest_path(
        [
            str(ROOT / "outputs" / "style_compare_cloud" / "**" / "*.csv"),
        ]
    )
    rows = _load_csv_rows(style_csv)
    if not rows:
        return (["最近一周没有拿到新的 style compare 结果。"], "")

    normalized: list[dict[str, Any]] = []
    for row in rows:
        style = str(row.get("style") or row.get("strategy") or row.get("name") or "").strip()
        if not style:
            continue
        sharpe = _safe_float(row.get("sharpe") or row.get("annual_sharpe"))
        annual_return = _safe_float(row.get("annual_return") or row.get("annualized_return"))
        mdd = _safe_float(row.get("max_drawdown") or row.get("drawdown"))
        normalized.append(
            {
                "style": style,
                "sharpe": sharpe,
                "annual_return": annual_return,
                "max_drawdown": mdd,
            }
        )
    if not normalized:
        return (["style compare 产物存在，但字段不完整，暂时无法自动排序。"], str(style_csv))

    ranked = sorted(normalized, key=lambda row: (row["sharpe"], row["annual_return"]), reverse=True)
    strongest = ranked[0]
    weakest = ranked[-1]
    lines = [
        f"最近最强风格：<b>{html.escape(strongest['style'])}</b>，Sharpe {strongest['sharpe']:.3f}，年化 {strongest['annual_return']:.2%}。",
        f"最近最弱风格：<b>{html.escape(weakest['style'])}</b>，Sharpe {weakest['sharpe']:.3f}，年化 {weakest['annual_return']:.2%}。",
    ]
    if len(ranked) > 2:
        runner = ranked[1]
        lines.append(
            f"次强风格：<b>{html.escape(runner['style'])}</b>，Sharpe {runner['sharpe']:.3f}，说明轮动并非单一风格独大。"
        )
    return lines, str(style_csv)


def _benchmark_snapshot() -> tuple[list[str], str]:
    bench_json = _latest_path(
        [
            str(ROOT / "outputs" / "locked_benchmark_2025" / "**" / "*.json"),
            str(ROOT / "outputs" / "formal_fusion_validation_*" / "formal_fusion_validation_summary.json"),
        ]
    )
    payload = _load_json(bench_json)
    if not payload:
        return (["最近一周没有拿到新的 benchmark 产物，暂时无法判断是否漂移。"], "")

    lines: list[str] = []
    if isinstance(payload.get("rows"), list) and payload["rows"]:
        rows = payload["rows"]
        baseline = next((row for row in rows if str(row.get("scenario")) == "baseline"), rows[0])
        fusion = next((row for row in rows if str(row.get("scenario")) in {"main", "steady"}), rows[-1])
        base_sharpe = _safe_float(baseline.get("sharpe"))
        fusion_sharpe = _safe_float(fusion.get("sharpe"))
        drift = fusion_sharpe - base_sharpe
        drift_text = "未明显漂移"
        if drift <= -0.10:
            drift_text = "出现负向漂移"
        elif drift >= 0.10:
            drift_text = "有正向改善"
        lines.append(
            f"最近 benchmark：基线 Sharpe {base_sharpe:.3f}，{html.escape(str(fusion.get('scenario')))} Sharpe {fusion_sharpe:.3f}，结论：<b>{drift_text}</b>。"
        )
        lines.append(
            f"基线年化 {(_safe_float(baseline.get('annual_return'))):.2%}，{html.escape(str(fusion.get('scenario')))} 年化 {(_safe_float(fusion.get('annual_return'))):.2%}。"
        )
        return lines, str(bench_json)
    if "scenarios" in payload:
        scenarios = payload.get("scenarios", {})
        baseline = scenarios.get("baseline", {})
        fusion = scenarios.get("main", {}) or scenarios.get("main_version", {})
        base_sharpe = _safe_float(baseline.get("sharpe"))
        fusion_sharpe = _safe_float(fusion.get("sharpe"))
        drift = fusion_sharpe - base_sharpe
        drift_text = "未明显漂移"
        if drift <= -0.10:
            drift_text = "出现负向漂移"
        elif drift >= 0.10:
            drift_text = "有正向改善"
        lines.append(
            f"最近 benchmark：基线 Sharpe {base_sharpe:.3f}，融合方案 Sharpe {fusion_sharpe:.3f}，结论：<b>{drift_text}</b>。"
        )
        lines.append(
            f"基线年化 {(_safe_float(baseline.get('annual_return'))):.2%}，融合年化 {(_safe_float(fusion.get('annual_return'))):.2%}。"
        )
        return lines, str(bench_json)

    annual_return = _safe_float(payload.get("annual_return"))
    sharpe = _safe_float(payload.get("sharpe"))
    mdd = _safe_float(payload.get("max_drawdown"))
    lines.append(
        f"最近 benchmark 摘要：年化 {annual_return:.2%}，Sharpe {sharpe:.3f}，最大回撤 {mdd:.2%}。"
    )
    lines.append("当前只有单份 benchmark 结果，暂时只能做快照，不能判断漂移方向。")
    return lines, str(bench_json)


def _candidate_pool_snapshot() -> tuple[list[str], str]:
    dashboard_json = _latest_candidate_pool_dashboard()
    payload = _load_json(dashboard_json)
    if not payload:
        return (["最近一周没有新的候选池质量看板，暂时只能维持冻结观察。"], "")
    lines = [
        f"冻结结论：<b>{'继续冻结' if bool(payload.get('keep_frozen', True)) else '可讨论解冻'}</b>。",
        f"原因：{html.escape(str(payload.get('keep_frozen_reason') or '暂无结论。'))}",
    ]
    summary_rows = payload.get("summary") or []
    for row in summary_rows[:3]:
        lines.append(
            "top{top_n}：平均收益 {avg_return:.2%}，命中率 {hit_rate:.2%}，胜率 {win_rate:.2%}，平均回撤 {avg_drawdown:.2%}。".format(
                top_n=int(row.get("top_n", 0) or 0),
                avg_return=_safe_float(row.get("avg_return")),
                hit_rate=_safe_float(row.get("hit_rate")),
                win_rate=_safe_float(row.get("win_rate")),
                avg_drawdown=_safe_float(row.get("avg_drawdown")),
            )
        )
    tag_rows = payload.get("tag_paths") or []
    realizable_row = next((row for row in tag_rows if str(row.get("path_quality")) == "可兑现收益"), None)
    spike_only_row = next((row for row in tag_rows if str(row.get("path_quality")) == "会冲一下"), None)
    if realizable_row:
        lines.append(
            f"更接近可兑现收益的标签：<b>{html.escape(str(realizable_row.get('group_value')))}</b>，样本 {int(realizable_row.get('sample_count', 0) or 0)}。"
        )
    if spike_only_row:
        lines.append(
            f"更像会冲一下的标签：<b>{html.escape(str(spike_only_row.get('group_value')))}</b>，样本 {int(spike_only_row.get('sample_count', 0) or 0)}。"
        )
    failure_rows = payload.get("failure_attribution") or []
    if failure_rows:
        ranked = sorted(
            failure_rows,
            key=lambda row: (
                _safe_float(row.get("path_error_ratio") or row.get("path_error_rate")),
                _safe_float(row.get("ranking_error_ratio") or row.get("ranking_error_rate")),
            ),
            reverse=True,
        )
        top_issue = ranked[0]
        path_error = _safe_float(top_issue.get("path_error_ratio") or top_issue.get("path_error_rate"))
        ranking_error = _safe_float(top_issue.get("ranking_error_ratio") or top_issue.get("ranking_error_rate"))
        lines.append(
            "最近最需要警惕的失败归因：top{top_n} 更像 <b>{dominant}</b>，路径错配占比 {path_rate:.2%}，排序错配占比 {rank_rate:.2%}。".format(
                top_n=int(top_issue.get("top_n", 0) or 0),
                dominant="兑现路径问题"
                if path_error >= ranking_error
                else "排序问题",
                path_rate=path_error,
                rank_rate=ranking_error,
            )
        )
    return lines, str(dashboard_json)


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
    style_lines, style_path = _style_snapshot()
    bench_lines, bench_path = _benchmark_snapshot()
    pool_lines, pool_path = _candidate_pool_snapshot()
    title = "量化观察期周检"
    content = (
        '<div style="background:#0b1020;padding:18px 14px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">'
        '<div style="max-width:860px;margin:0 auto;">'
        '<div style="background:linear-gradient(135deg,#16213f,#0f172a);border-radius:22px;padding:22px 20px;margin-bottom:16px;border:1px solid #2b3b5a;">'
        '<div style="font-size:24px;font-weight:900;color:#fff;">量化观察期周检</div>'
        '<div style="margin-top:8px;color:#a8b8d8;font-size:14px;">每周五晚推送一次，重点检查候选池质量变化、失败归因和 keep_frozen 结论。</div>'
        '</div>'
        + _summary_card("候选池质量与冻结结论", pool_lines)
        + _summary_card("最近风格强弱", style_lines)
        + _summary_card("Benchmark 漂移检查", bench_lines)
        + '</div></div>'
    )
    result = {
        "title": title,
        "candidate_pool_source": pool_path,
        "style_source": style_path,
        "benchmark_source": bench_path,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        result["preview"] = content[:5000]
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
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
