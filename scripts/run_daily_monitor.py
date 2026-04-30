from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.latest_picks import (
    LatestPickSummary,
    build_review_question_pack,
    build_latest_strategy_table,
    build_latest_picks_table,
)
from ashare_quant.analysis.metric_pack import (
    build_standard_metric_pack,
    metric_pack_dataframe,
    metric_pack_markdown,
)
from ashare_quant.analysis.research_report import extract_display_weights
from ashare_quant.data.research_slice import build_research_slice
from ashare_quant.data.tushare_sync import enrich_local_history_fundamentals
from ashare_quant.data.tushare_sync import update_local_history_with_tushare
from ashare_quant.pipeline import load_json, run_research_pipeline
from ashare_quant.strategy_classifier import assessment_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a daily end-of-day A-share monitoring workflow: optional Tushare update, latest picks, and optional backtest summary."
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--slice-path", default=str(ROOT / "data" / "daily_monitor_slice.csv"))
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--max-codes", type=int, default=None, help="Use 0 for the full eligible universe.")
    parser.add_argument("--suffixes", default="SH,SZ")
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--research-config", default=str(ROOT / "configs" / "research_production_default.json"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_production_managed_15bps.json"))
    parser.add_argument("--prediction-date", default="")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "daily_monitor"))
    parser.add_argument("--output-prefix", default="daily_monitor")
    parser.add_argument("--skip-update", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument(
        "--refresh-fundamentals",
        action="store_true",
        help="After building the research slice, pull Tushare announcement fundamentals into the slice for veto/filter use.",
    )
    parser.add_argument("--http-url", default=None)
    parser.add_argument("--proxy-url", default=None)
    parser.add_argument("--bypass-system-proxy", action="store_true")
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    return parser


def _top_weight_lines(weights: dict[str, float], limit: int = 5) -> list[str]:
    ranked = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
    return [f"- `{name}`: {value:.4f}" for name, value in ranked[: min(limit, len(ranked))]]


def _status_badge(status: str) -> str:
    status_upper = str(status or "").upper()
    if status_upper == "PASS":
        return "GREEN"
    if status_upper == "WARN":
        return "YELLOW"
    if status_upper == "BLOCK":
        return "RED"
    return "GRAY"


def _candidate_pool_quality_lines(metrics: dict | None) -> list[str]:
    quality = (metrics or {}).get("candidate_pool_quality") or {}
    rows = quality.get("summary_rows") or []
    if not rows:
        return ["- candidate pool quality not available"]
    lines = [f"- {quality.get('headline', '').strip() or '候选池质量摘要可用'}"]
    for row in rows:
        lines.append(
            "- {pool}: avg_return={avg_return:.4f}, hit_rate={hit_rate:.2%}, win_rate={win_rate:.2%}, avg_drawdown={avg_drawdown:.4f}, payoff_ratio={payoff_ratio:.4f}".format(
                pool=row.get("pool_name", ""),
                avg_return=float(row.get("avg_return", 0.0) or 0.0),
                hit_rate=float(row.get("hit_rate", 0.0) or 0.0),
                win_rate=float(row.get("win_rate", 0.0) or 0.0),
                avg_drawdown=float(row.get("avg_drawdown", 0.0) or 0.0),
                payoff_ratio=float(row.get("payoff_ratio", 0.0) or 0.0),
            )
        )
    failure_rows = quality.get("failure_attribution_rows") or []
    for row in failure_rows:
        lines.append(
            "- {pool}: 排序错占比={ranking_error_ratio:.2%}，兑现路径错占比={path_error_ratio:.2%}，可兑现占比={realized_ratio:.2%}；{comment}".format(
                pool=row.get("pool_name", ""),
                ranking_error_ratio=float(row.get("ranking_error_ratio", 0.0) or 0.0),
                path_error_ratio=float(row.get("path_error_ratio", 0.0) or 0.0),
                realized_ratio=float(row.get("realized_ratio", 0.0) or 0.0),
                comment=row.get("failure_comment", "") or "暂不可判定",
            )
        )
    return lines


def _stable_cycle_lines(summary: LatestPickSummary, research_summary: dict | None) -> list[str]:
    research_summary = research_summary or {}
    return [
        f"- system_mode: `{research_summary.get('system_mode', summary.system_mode)}`",
        f"- main_score_frozen: `{research_summary.get('main_score_frozen', summary.main_score_frozen)}`",
        f"- execution_layer_frozen: `{research_summary.get('execution_layer_frozen', summary.execution_layer_frozen)}`",
        f"- observation_layer_frozen: `{research_summary.get('observation_layer_frozen', summary.observation_layer_frozen)}`",
        "- freeze_message: `当前只允许 bugfix、监控、质量看板和报告结构增强，不进行新因子升级，不进行执行规则升级。`",
        f"- new_research_gate_passed: `{research_summary.get('new_research_gate_passed', summary.new_research_gate_passed)}`",
        f"- why_not: `{research_summary.get('new_research_gate_reason', summary.new_research_gate_reason)}`",
    ]


def _review_question_lines(observation_table: pd.DataFrame | None, limit: int = 5) -> list[str]:
    if observation_table is None or observation_table.empty:
        return ["- 暂不可判定"]
    lines: list[str] = []
    for row in observation_table.head(limit).itertuples(index=False):
        review_pack = build_review_question_pack(row._asdict())
        lines.append(
            "- {code} {name}: 1) {selected_because} 2) {label_combo} 3) {failure_attribution}".format(
                code=getattr(row, "code", ""),
                name=getattr(row, "name", "") or "",
                selected_because=review_pack["selected_because"],
                label_combo=review_pack["label_combo"],
                failure_attribution=review_pack["failure_attribution"],
            )
        )
    return lines


def build_daily_monitor_markdown(
    summary: LatestPickSummary,
    picks_json_path: Path,
    observation_table,
    strategy_table,
    factor_weights: dict[str, float],
    metrics: dict | None,
    update_info: dict | None,
    strategy_assessment: dict | None,
    research_summary: dict | None,
    risk_summary: dict | None = None,
    performance_artifacts: dict[str, str] | None = None,
) -> str:
    observation_lines = []
    observation_has_ml = "ml_observation_tag" in observation_table.columns if observation_table is not None else False
    observation_has_leader = "industry_leader_follow_tag" in observation_table.columns if observation_table is not None else False
    observation_has_pressure = "overhead_density_tag" in observation_table.columns if observation_table is not None else False
    observation_detail_lines = []
    for row in observation_table.itertuples(index=False):
        if observation_has_ml or observation_has_leader or observation_has_pressure:
            observation_lines.append(
                "| {rank} | {code} | {name} | {weight:.3f} | {score:.4f} | {ml_tag} | {leader_tag} | {pressure_tag} | {industry} |".format(
                    rank=int(getattr(row, "rank", 0)),
                    code=getattr(row, "code", ""),
                    name=getattr(row, "name", "") or "",
                    weight=float(getattr(row, "target_weight", 0.0)),
                    score=float(getattr(row, "score", 0.0)),
                    ml_tag=getattr(row, "ml_observation_tag", "") or "无",
                    leader_tag=getattr(row, "industry_leader_follow_tag", "") or "无",
                    pressure_tag=getattr(row, "overhead_density_tag", "") or "无",
                    industry=getattr(row, "industry", "") or "Unknown",
                )
            )
        else:
            observation_lines.append(
                "| {rank} | {code} | {name} | {weight:.3f} | {score:.4f} | {industry} |".format(
                    rank=int(getattr(row, "rank", 0)),
                    code=getattr(row, "code", ""),
                    name=getattr(row, "name", "") or "",
                    weight=float(getattr(row, "target_weight", 0.0)),
                    score=float(getattr(row, "score", 0.0)),
                    industry=getattr(row, "industry", "") or "Unknown",
                )
            )
        observation_detail_lines.append(
            "- {code} {name}: 龙头扩散 {leader_score:.4f}/#{leader_rank} {leader_tag}；兑现压力 {pressure_score:.4f}/#{pressure_rank} {pressure_tag}；{leader_detail}；{pressure_detail}".format(
                code=getattr(row, "code", ""),
                name=getattr(row, "name", "") or "",
                leader_score=float(getattr(row, "industry_leader_follow_score", 0.0) or 0.0),
                leader_rank=int(getattr(row, "industry_leader_follow_rank", 0) or 0),
                leader_tag=getattr(row, "industry_leader_follow_tag", "") or "无",
                pressure_score=float(getattr(row, "overhead_density_score", 0.0) or 0.0),
                pressure_rank=int(getattr(row, "overhead_density_rank", 0) or 0),
                pressure_tag=getattr(row, "overhead_density_tag", "") or "无",
                leader_detail=getattr(row, "industry_leader_follow_detail", "") or "无龙头扩散说明",
                pressure_detail=getattr(row, "overhead_density_detail", "") or "无兑现压力说明",
            )
        )
    strategy_lines = []
    if strategy_table is not None and not strategy_table.empty:
        for row in strategy_table.itertuples(index=False):
            strategy_lines.append(
                "| {rank} | {code} | {name} | {weight:.3f} | {score:.4f} | {industry} | {reason} | {risk_state} |".format(
                    rank=int(getattr(row, "rank", 0)),
                    code=getattr(row, "code", ""),
                    name=getattr(row, "name", "") or "",
                    weight=float(getattr(row, "target_weight", 0.0)),
                    score=float(getattr(row, "score", 0.0)),
                    industry=getattr(row, "industry", "") or "Unknown",
                    reason=getattr(row, "trade_reason", "") or "hold",
                    risk_state=getattr(row, "risk_state", "") or "full_risk",
                )
            )
    update_block = [
        f"- updated: `{bool(update_info)}`",
    ]
    if update_info:
        update_block.extend(
            [
                f"- update rows: `{update_info.get('rows', 0)}`",
                f"- pull start: `{update_info.get('pull_start', '')}`",
                f"- pull end: `{update_info.get('pull_end', '')}`",
                f"- new rows: `{update_info.get('new_rows', 0)}`",
            ]
        )

    metric_block = ["- backtest skipped"] if metrics is None else [
        f"- strategy_classification: `{(strategy_assessment or {}).get('classification', 'n/a')}`",
        f"- gross_annual_return: `{metrics.get('gross_annual_return', 0.0):.4f}`",
        f"- annual_return: `{metrics.get('annual_return', 0.0):.4f}`",
        f"- sharpe: `{metrics.get('sharpe', 0.0):.4f}`",
        f"- max_drawdown: `{metrics.get('max_drawdown', 0.0):.4f}`",
        f"- avg_turnover: `{metrics.get('avg_turnover', 0.0):.4f}`",
        f"- after_cost_return_drag: `{metrics.get('after_cost_return_drag', 0.0):.4f}`",
        f"- payoff_ratio: `{metrics.get('payoff_ratio', 0.0):.4f}`",
        f"- signal_time: `{metrics.get('signal_time', '')}`",
        f"- execution_time: `{metrics.get('execution_time', '')}`",
    ]
    candidate_pool_quality_block = _candidate_pool_quality_lines(metrics)
    stable_cycle_block = _stable_cycle_lines(summary, research_summary)
    review_question_block = _review_question_lines(observation_table)

    weight_lines = _top_weight_lines(factor_weights) or ["- no weights"]
    risk_status = str((risk_summary or {}).get("status", "n/a"))
    risk_badge = _status_badge(risk_status)
    strategy_classification = (strategy_assessment or {}).get("classification", "n/a")
    action_hint = "review only"
    if risk_status == "PASS" and strategy_classification == "tradable prototype":
        action_hint = "paper/live eligible"
    elif risk_status == "WARN":
        action_hint = "trade with caution"
    elif risk_status == "BLOCK":
        action_hint = "no new paper/live trades"
    risk_block = [
        f"- status: `{risk_summary.get('status', 'n/a')}`",
        f"- annual_return: `{float(risk_summary.get('annual_return', 0.0) or 0.0):.4f}`",
        f"- sharpe: `{float(risk_summary.get('sharpe', 0.0) or 0.0):.4f}`",
        f"- max_drawdown: `{float(risk_summary.get('max_drawdown', 0.0) or 0.0):.4f}`",
        f"- avg_turnover: `{float(risk_summary.get('avg_turnover', 0.0) or 0.0):.4f}`",
        f"- selected_count: `{int(risk_summary.get('selected_count', 0) or 0)}`",
        f"- top_industry_weight: `{float(risk_summary.get('top_industry_weight', 0.0) or 0.0):.4f}`",
        f"- max_single_weight: `{float(risk_summary.get('max_single_weight', 0.0) or 0.0):.4f}`",
    ] if risk_summary else ["- risk governor not refreshed yet"]
    lines = [
        "# Daily Monitor",
        "",
        "## Desk Snapshot",
        "",
        f"- strategy_classification: `{strategy_classification}`",
        f"- risk_status: `{risk_status}`",
        f"- status_badge: `{risk_badge}`",
        f"- action_hint: `{action_hint}`",
        f"- selection_date: `{summary.selection_date}`",
        f"- selected_count: `{summary.selected_count}`",
        f"- candidate_count: `{summary.candidate_count}`",
        f"- annual_return: `{float((metrics or {}).get('annual_return', 0.0) or 0.0):.4f}`",
        f"- sharpe: `{float((metrics or {}).get('sharpe', 0.0) or 0.0):.4f}`",
        f"- max_drawdown: `{float((metrics or {}).get('max_drawdown', 0.0) or 0.0):.4f}`",
        f"- avg_turnover: `{float((metrics or {}).get('avg_turnover', 0.0) or 0.0):.4f}`",
        "",
        "## Stable Observation Cycle",
        "",
        *stable_cycle_block,
        "",
        "## Decision Panel",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| classification | `{strategy_classification}` |",
        f"| risk_status | `{risk_status}` |",
        f"| status_badge | `{risk_badge}` |",
        f"| action_hint | `{action_hint}` |",
        f"| execution_time | `{summary.execution_time}` |",
        f"| top_factor_1 | `{next(iter(sorted(factor_weights.items(), key=lambda item: abs(item[1]), reverse=True)), ('-', 0.0))[0] if factor_weights else '-'}` |",
        "",
        "## Update",
        "",
        *update_block,
        "",
        "## Selection",
        "",
        f"- layer: `{summary.layer}`",
        f"- selection date: `{summary.selection_date}`",
        f"- candidate count: `{summary.candidate_count}`",
        f"- selected count: `{summary.selected_count}`",
        f"- label type: `{summary.label_type}`",
        f"- signal time: `{summary.signal_time}`",
        f"- execution time: `{summary.execution_time}`",
        f"- picks json: `{picks_json_path}`",
        "",
        "## Recent Backtest",
        "",
        *metric_block,
        "",
        "## Candidate Pool Quality",
        "",
        *candidate_pool_quality_block,
        "",
        "## Performance Artifacts",
        "",
        *(f"- {key}: `{value}`" for key, value in (performance_artifacts or {}).items()),
        "",
        "## Top Weights",
        "",
        *weight_lines,
        "",
        "## Strategy Assessment",
        "",
        f"- analyst_view: `{(strategy_assessment or {}).get('analyst_view', 'n/a')}`",
        f"- trader_view: `{(strategy_assessment or {}).get('trader_view', 'n/a')}`",
        f"- exposure_warning: `{(strategy_assessment or {}).get('exposure_warning', 'n/a')}`",
        *(f"- risk: `{risk}`" for risk in (strategy_assessment or {}).get("key_risks", [])[:5]),
        "",
        "## Risk Governor",
        "",
        *risk_block,
        "",
        "## Observation Pool",
        "",
        *(
            [
                "| rank | code | name | weight | score | ML观察 | 龙头扩散 | 兑现压力 | industry |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            if observation_has_ml or observation_has_leader or observation_has_pressure
            else [
                "| rank | code | name | weight | score | industry |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        ),
        *(observation_lines if observation_lines else (["| - | - | - | - | - | - | - | - | - |"] if observation_has_ml or observation_has_leader or observation_has_pressure else ["| - | - | - | - | - | - |"])),
        "",
        "## Observation Layer Detail",
        "",
        *(observation_detail_lines[:10] if observation_detail_lines else ["- observation detail not available"]),
        "",
        "## 复盘三问",
        "",
        *review_question_block,
        "",
        "## Tradable Strategy Snapshot",
        "",
        "| rank | code | name | weight | score | industry | trade_reason | risk_state |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *(strategy_lines if strategy_lines else ["| - | - | - | - | - | - | - | - |"]),
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    args = build_parser().parse_args()
    suffixes = tuple(part.strip().upper() for part in args.suffixes.split(",") if part.strip())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    update_info = None
    data_path = Path(args.data_path)
    research_cfg = load_json(args.research_config)
    if not args.skip_update:
        _updated_path, total_rows, source = update_local_history_with_tushare(
            existing_path=data_path,
            http_url=args.http_url,
            proxy_url=args.proxy_url,
            bypass_system_proxy=args.bypass_system_proxy,
            pause_seconds=args.pause_seconds,
        )
        latest_summary = source.last_summary_ if source is not None else None
        update_info = {
            "rows": total_rows,
            "pull_start": latest_summary.start_date if latest_summary is not None else "",
            "pull_end": latest_summary.end_date if latest_summary is not None else "",
            "new_rows": latest_summary.rows if latest_summary is not None else 0,
        }

    profile = build_research_slice(
        input_path=data_path,
        output_path=args.slice_path,
        start_date=args.start_date,
        end_date=args.end_date,
        include_suffixes=suffixes,
        max_codes=args.max_codes if args.max_codes is not None else int(research_cfg.get("universe_max_codes", 500)),
    )
    if args.refresh_fundamentals:
        enrich_local_history_fundamentals(
            existing_path=args.slice_path,
            output_path=args.slice_path,
            http_url=args.http_url,
            proxy_url=args.proxy_url,
            bypass_system_proxy=args.bypass_system_proxy,
            pause_seconds=args.pause_seconds,
        )

    observation_pool, raw_weights, summary = build_latest_picks_table(
        data_path=args.slice_path,
        research_config_path=args.research_config,
        data_adjust=args.adjust,
        prediction_date=args.prediction_date or None,
    )
    strategy_snapshot, _, _strategy_summary = build_latest_strategy_table(
        data_path=args.slice_path,
        research_config_path=args.research_config,
        data_adjust=args.adjust,
        prediction_date=args.prediction_date or None,
    )
    factor_weights = extract_display_weights(raw_weights)

    metrics = None
    result = None
    strategy_assessment = None
    if not args.skip_backtest:
        result, metrics, score_details, _targets = run_research_pipeline(
            data_path=args.slice_path,
            research_config_path=args.research_config,
            backtest_config_path=args.backtest_config,
            data_adjust=args.adjust,
        )
        factor_weights = extract_display_weights(score_details)
        strategy_assessment = assessment_payload(metrics=metrics, picks=strategy_snapshot)

    prefix = f"{args.output_prefix}_{summary.selection_date.replace('-', '')}"
    picks_csv_path = output_dir / f"{prefix}_picks.csv"
    picks_json_path = output_dir / f"{prefix}_picks.json"
    metrics_path = output_dir / f"{prefix}_metrics.json"
    metric_pack_json_path = output_dir / f"{prefix}_metric_pack.json"
    metric_pack_csv_path = output_dir / f"{prefix}_metric_pack.csv"
    metric_pack_md_path = output_dir / f"{prefix}_metric_pack.md"
    equity_curve_csv_path = output_dir / f"{prefix}_equity_curve.csv"
    calendar_year_csv_path = output_dir / f"{prefix}_calendar_year_metrics.csv"
    calendar_year_json_path = output_dir / f"{prefix}_calendar_year_metrics.json"
    calendar_month_csv_path = output_dir / f"{prefix}_calendar_month_metrics.csv"
    calendar_month_json_path = output_dir / f"{prefix}_calendar_month_metrics.json"
    performance_md_path = output_dir / f"{prefix}_performance_eval.md"
    equity_curve_png_path = output_dir / f"{prefix}_equity_curve.png"
    drawdown_curve_png_path = output_dir / f"{prefix}_drawdown_curve.png"
    calendar_year_png_path = output_dir / f"{prefix}_calendar_year_return.png"
    calendar_month_png_path = output_dir / f"{prefix}_calendar_month_return.png"
    report_path = output_dir / f"{prefix}_report.md"
    performance_artifacts: dict[str, str] = {}

    observation_pool.to_csv(picks_csv_path, index=False)
    picks_json_path.write_text(
        json.dumps(
            {
                "profile": profile.__dict__,
                "summary": summary.__dict__,
                "factor_weights": factor_weights,
                "strategy_assessment": strategy_assessment or {},
                "candidate_pool_quality": (metrics or {}).get("candidate_pool_quality", {}),
                "research_summary": (metrics or {}).get("research_summary", {}),
                "observation_pool": json.loads(observation_pool.to_json(orient="records", date_format="iso", force_ascii=False)),
                "strategy_snapshot": json.loads(strategy_snapshot.to_json(orient="records", date_format="iso", force_ascii=False)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if metrics is not None:
        from ashare_quant.analysis.performance_eval import (
            build_calendar_metrics,
            build_equity_curve_frame,
            build_performance_eval_markdown,
        )
        from ashare_quant.analysis.performance_plots import (
            save_calendar_bar_plot,
            save_drawdown_curve_plot,
            save_equity_curve_plot,
        )

        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        metric_pack = build_standard_metric_pack(
            metrics=metrics,
            selection_date=summary.selection_date,
            source_metrics_json=str(metrics_path),
            strategy_classification=(strategy_assessment or {}).get("classification", ""),
        )
        metric_pack_json_path.write_text(json.dumps(metric_pack, ensure_ascii=False, indent=2), encoding="utf-8")
        metric_pack_dataframe(metric_pack).to_csv(metric_pack_csv_path, index=False)
        metric_pack_md_path.write_text(metric_pack_markdown(metric_pack), encoding="utf-8")
        equity_curve = build_equity_curve_frame(result)
        yearly_metrics = build_calendar_metrics(result, freq="Y", annual_days=int(metrics.get("annual_trading_days", 252)))
        monthly_metrics = build_calendar_metrics(result, freq="M", annual_days=int(metrics.get("annual_trading_days", 252)))
        equity_curve.to_csv(equity_curve_csv_path, index=False)
        yearly_metrics.to_csv(calendar_year_csv_path, index=False)
        monthly_metrics.to_csv(calendar_month_csv_path, index=False)
        calendar_year_json_path.write_text(yearly_metrics.to_json(orient="records", force_ascii=False, date_format="iso", indent=2), encoding="utf-8")
        calendar_month_json_path.write_text(monthly_metrics.to_json(orient="records", force_ascii=False, date_format="iso", indent=2), encoding="utf-8")
        save_equity_curve_plot(equity_curve, equity_curve_png_path)
        save_drawdown_curve_plot(equity_curve, drawdown_curve_png_path)
        save_calendar_bar_plot(yearly_metrics, calendar_year_png_path, value_col="annual_return", title="Calendar Year Annual Return")
        save_calendar_bar_plot(monthly_metrics, calendar_month_png_path, value_col="annual_return", title="Calendar Month Annualized Return")
        performance_md_path.write_text(
            build_performance_eval_markdown(
                yearly_metrics,
                monthly_metrics,
                equity_curve_csv=equity_curve_csv_path,
                equity_curve_png=equity_curve_png_path,
                drawdown_curve_png=drawdown_curve_png_path,
            ),
            encoding="utf-8",
        )
        performance_artifacts = {
            "equity_curve_csv": str(equity_curve_csv_path),
            "calendar_year_csv": str(calendar_year_csv_path),
            "calendar_month_csv": str(calendar_month_csv_path),
            "equity_curve_png": str(equity_curve_png_path),
            "drawdown_curve_png": str(drawdown_curve_png_path),
            "calendar_year_png": str(calendar_year_png_path),
            "calendar_month_png": str(calendar_month_png_path),
            "performance_eval_md": str(performance_md_path),
        }
    report_path.write_text(
        build_daily_monitor_markdown(
            summary=summary,
            picks_json_path=picks_json_path,
            observation_table=observation_pool,
            strategy_table=strategy_snapshot,
            factor_weights=factor_weights,
            metrics=metrics,
            update_info=update_info,
            strategy_assessment=strategy_assessment,
            research_summary=(metrics or {}).get("research_summary", {}),
            performance_artifacts=performance_artifacts,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "profile": profile.__dict__,
                "summary": summary.__dict__,
                "factor_weights": factor_weights,
                "strategy_assessment": strategy_assessment,
                "update": update_info,
                "report_path": str(report_path),
                "metric_pack_path": str(metric_pack_json_path) if metrics is not None else "",
                "performance_artifacts": performance_artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[OK] daily monitor outputs saved to {output_dir}")
