from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _latest_picks_json(directory: Path) -> Path | None:
    files = sorted(directory.glob("*_picks.json"))
    if not files:
        return None
    best_file = files[0]
    best_date = ""
    for candidate in files:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            date = str(payload.get("summary", {}).get("selection_date", "")).strip()
        except Exception:
            date = ""
        if date > best_date:
            best_date = date
            best_file = candidate
    if best_date:
        return best_file
    return max(files, key=lambda p: p.stat().st_mtime)


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_badge(status: str) -> str:
    status_upper = str(status or "").upper()
    if status_upper == "PASS":
        return "GREEN"
    if status_upper == "WARN":
        return "YELLOW"
    if status_upper == "BLOCK":
        return "RED"
    return "GRAY"


def _load_payload_from_directory(directory: Path) -> tuple[dict[str, Any], str]:
    picks_path = _latest_picks_json(directory)
    if picks_path is not None:
        return _load_payload(picks_path), str(picks_path)

    summary_path = directory / "summary.json"
    observation_path = directory / "observation_pool.csv"
    strategy_path = directory / "strategy_snapshot.csv"
    if not summary_path.exists() or not observation_path.exists():
        raise FileNotFoundError(
            f"No *_picks.json and no summary/observation fallback files in {directory}"
        )

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = summary_payload.get("strategy_summary") or summary_payload.get("observation_summary") or summary_payload
    observation_df = pd.read_csv(observation_path)
    strategy_df = pd.read_csv(strategy_path) if strategy_path.exists() else pd.DataFrame()
    payload = {
        "summary": summary,
        "factor_weights": summary_payload.get("factor_weights", {}),
        "observation_pool": json.loads(observation_df.to_json(orient="records", force_ascii=False)),
        "strategy_snapshot": json.loads(strategy_df.to_json(orient="records", force_ascii=False)),
    }
    return payload, str(directory)


def _select_display_table(payload: dict[str, Any]) -> pd.DataFrame:
    strategy = pd.DataFrame(payload.get("strategy_snapshot", []))
    if not strategy.empty:
        return strategy.copy()
    observation = pd.DataFrame(payload.get("observation_pool", []))
    return observation.copy()


def _to_lines(df: pd.DataFrame, limit: int) -> list[str]:
    if df.empty:
        return ["| - | - | - | - | - | - |"]
    cols = ["rank", "code", "name", "close", "industry", "score"]
    data = df.copy()
    for col in cols:
        if col not in data.columns:
            data[col] = None
    lines: list[str] = []
    for row in data[cols].head(limit).itertuples(index=False):
        rank = "-" if pd.isna(row.rank) else int(row.rank)
        close = "-" if pd.isna(row.close) else f"{float(row.close):.2f}"
        score = "-" if pd.isna(row.score) else f"{float(row.score):.4f}"
        lines.append(
            f"| {rank} | {row.code or '-'} | {row.name or '-'} | {close} | {row.industry or '-'} | {score} |"
        )
    return lines


def _build_daily_checklist(
    *,
    main_summary: dict[str, Any],
    elastic_summary: dict[str, Any],
    main_payload: dict[str, Any],
    elastic_payload: dict[str, Any],
    compare_csv: Path,
) -> list[dict[str, str]]:
    main_date = str(main_summary.get("selection_date", "") or "")
    elastic_date = str(elastic_summary.get("selection_date", "") or "")
    main_selected = int(main_summary.get("selected_count", 0) or 0)
    elastic_selected = int(elastic_summary.get("selected_count", 0) or 0)
    main_strategy_rows = len(pd.DataFrame(main_payload.get("strategy_snapshot", [])))
    elastic_table_rows = len(_select_display_table(elastic_payload))
    items = [
        {
            "check": "main_date_present",
            "status": "PASS" if bool(main_date) else "WARN",
            "detail": main_date or "missing",
        },
        {
            "check": "elastic_date_present",
            "status": "PASS" if bool(elastic_date) else "WARN",
            "detail": elastic_date or "missing",
        },
        {
            "check": "same_selection_date",
            "status": "PASS" if main_date and elastic_date and main_date == elastic_date else "WARN",
            "detail": f"main={main_date or 'n/a'}, elastic={elastic_date or 'n/a'}",
        },
        {
            "check": "main_selected_nonzero",
            "status": "PASS" if main_selected > 0 else "WARN",
            "detail": str(main_selected),
        },
        {
            "check": "elastic_selected_nonzero",
            "status": "PASS" if elastic_selected > 0 else "WARN",
            "detail": str(elastic_selected),
        },
        {
            "check": "main_strategy_snapshot_present",
            "status": "PASS" if main_strategy_rows > 0 else "WARN",
            "detail": str(main_strategy_rows),
        },
        {
            "check": "elastic_display_table_present",
            "status": "PASS" if elastic_table_rows > 0 else "WARN",
            "detail": str(elastic_table_rows),
        },
        {
            "check": "compare_csv_written",
            "status": "PASS" if compare_csv.exists() else "WARN",
            "detail": str(compare_csv),
        },
    ]
    return items


def _load_matching_risk_summary(risk_dir: Path, selection_date: str) -> dict[str, Any]:
    if not risk_dir.exists():
        return {}
    if selection_date:
        candidate = risk_dir / f"risk_gate_{selection_date.replace('-', '')}.json"
        if candidate.exists():
            try:
                return _load_payload(candidate)
            except Exception:
                pass
    files = sorted(risk_dir.glob("risk_gate_*.json"))
    if not files:
        return {}
    for candidate in reversed(files):
        try:
            return _load_payload(candidate)
        except Exception:
            continue
    return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a dual-board daily monitor report.")
    parser.add_argument("--main-dir", default=str(ROOT / "outputs" / "daily_monitor_auto"))
    parser.add_argument("--elastic-dir", default=str(ROOT / "outputs" / "daily_monitor_under20_elastic"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "daily_monitor_dual"))
    parser.add_argument("--output-prefix", default="dual_monitor")
    parser.add_argument("--risk-dir", default=str(ROOT / "outputs" / "risk_governor"))
    parser.add_argument("--main-title", default="主策略（稳健赚钱）")
    parser.add_argument("--elastic-title", default="涨停弹性池（高风险高收益，20元以内）")
    parser.add_argument("--main-top", type=int, default=12)
    parser.add_argument("--elastic-top", type=int, default=20)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main_dir = Path(args.main_dir)
    elastic_dir = Path(args.elastic_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_payload, main_path = _load_payload_from_directory(main_dir)
    elastic_payload, elastic_path = _load_payload_from_directory(elastic_dir)

    main_summary = main_payload.get("summary", {})
    elastic_summary = elastic_payload.get("summary", {})
    risk_summary = _load_matching_risk_summary(Path(args.risk_dir), str(main_summary.get("selection_date", "") or ""))
    main_date = str(main_summary.get("selection_date", ""))
    elastic_date = str(elastic_summary.get("selection_date", ""))
    selection_date = min([d for d in [main_date, elastic_date] if d], default="")
    date_suffix = selection_date.replace("-", "") if selection_date else "latest"

    main_df = _select_display_table(main_payload)
    elastic_df = _select_display_table(elastic_payload)

    main_rows = main_df.copy()
    main_rows["board"] = args.main_title
    elastic_rows = elastic_df.copy()
    elastic_rows["board"] = args.elastic_title

    compare_cols = ["board", "rank", "code", "name", "close", "industry", "score", "target_weight"]
    for col in compare_cols:
        if col not in main_rows.columns:
            main_rows[col] = None
        if col not in elastic_rows.columns:
            elastic_rows[col] = None
    compare_df = pd.concat([main_rows[compare_cols], elastic_rows[compare_cols]], ignore_index=True)

    compare_csv = output_dir / f"{args.output_prefix}_{date_suffix}_compare.csv"
    compare_df.to_csv(compare_csv, index=False)
    checklist = _build_daily_checklist(
        main_summary=main_summary,
        elastic_summary=elastic_summary,
        main_payload=main_payload,
        elastic_payload=elastic_payload,
        compare_csv=compare_csv,
    )

    report_path = output_dir / f"{args.output_prefix}_{date_suffix}_report.md"
    risk_status = str(risk_summary.get("status", "n/a"))
    risk_badge = _status_badge(risk_status)
    main_classification = str(main_payload.get("strategy_assessment", {}).get("classification", "n/a"))
    action_hint = "review only"
    if risk_status == "PASS" and main_classification == "tradable prototype":
        action_hint = "main strategy paper/live eligible"
    elif risk_status == "WARN":
        action_hint = "main strategy caution; elastic pool observation only"
    elif risk_status == "BLOCK":
        action_hint = "main strategy blocked; keep dual report as monitor only"
    lines = [
        f"# 双轨选股日报 ({selection_date or 'latest'})",
        "",
        "## Desk Snapshot",
        "",
        f"- main_classification: `{main_classification}`",
        f"- risk_status: `{risk_status}`",
        f"- status_badge: `{risk_badge}`",
        f"- action_hint: `{action_hint}`",
        f"- main_selected_count: `{main_summary.get('selected_count', 0)}`",
        f"- elastic_selected_count: `{elastic_summary.get('selected_count', 0)}`",
        f"- main_top_industry_weight: `{float(risk_summary.get('top_industry_weight', 0.0) or 0.0):.4f}`",
        "",
        "## Decision Panel",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| main_classification | `{main_classification}` |",
        f"| risk_status | `{risk_status}` |",
        f"| status_badge | `{risk_badge}` |",
        f"| action_hint | `{action_hint}` |",
        f"| same_selection_date | `{'yes' if main_date and elastic_date and main_date == elastic_date else 'no'}` |",
        f"| compare_csv | `{compare_csv.name}` |",
        "",
        "## 运行摘要",
        "",
        f"- 主策略日期: `{main_date or 'n/a'}`",
        f"- 弹性池日期: `{elastic_date or 'n/a'}`",
        f"- 主策略候选数: `{main_summary.get('candidate_count', 0)}`，入选数: `{main_summary.get('selected_count', 0)}`",
        f"- 弹性池候选数: `{elastic_summary.get('candidate_count', 0)}`，入选数: `{elastic_summary.get('selected_count', 0)}`",
        f"- 主策略来源: `{main_path}`",
        f"- 弹性池来源: `{elastic_path}`",
        "",
        "## 日报 Checklist",
        "",
        "| 检查项 | 状态 | 说明 |",
        "| --- | --- | --- |",
        *(
            f"| {item['check']} | {item['status']} | {item['detail']} |"
            for item in checklist
        ),
        "",
        f"## {args.main_title}",
        "",
        "| 排名 | 代码 | 名称 | 收盘价 | 行业 | 分数 |",
        "| --- | --- | --- | --- | --- | --- |",
        *_to_lines(main_df, args.main_top),
        "",
        f"## {args.elastic_title}",
        "",
        "| 排名 | 代码 | 名称 | 收盘价 | 行业 | 分数 |",
        "| --- | --- | --- | --- | --- | --- |",
        *_to_lines(elastic_df, args.elastic_top),
        "",
        "## 输出文件",
        "",
        f"- 对比CSV: `{compare_csv}`",
        f"- 本报告: `{report_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "selection_date": selection_date,
                "main_selected_count": int(main_summary.get("selected_count", 0) or 0),
                "elastic_selected_count": int(elastic_summary.get("selected_count", 0) or 0),
                "compare_csv": str(compare_csv),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
