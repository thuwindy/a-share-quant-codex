from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.csv_adapter import CSVDataSource
from ashare_quant.execution.paper import PaperExecutionConfig
from ashare_quant.execution.paper_monitor import (
    PaperRiskConfig,
    apply_paper_risk_overlay,
    assess_paper_risk,
    execute_due_orders,
    load_paper_state,
    mark_to_market,
    save_paper_state,
    stage_orders_for_next_execution,
    write_paper_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Advance a daily-K paper portfolio with real monitor outputs, staged next-day orders, and risk-first guardrails."
    )
    parser.add_argument("--monitor-json", default="")
    parser.add_argument("--monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_auto"))
    parser.add_argument("--slice-path", default=str(ROOT / "data" / "daily_monitor_slice.csv"))
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--paper-config", default=str(ROOT / "configs" / "paper_trade_guardrails.json"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_liquidity.json"))
    parser.add_argument("--state-path", default=str(ROOT / "outputs" / "paper_monitor_auto" / "paper_state.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "paper_monitor_auto"))
    return parser


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _latest_monitor_json(monitor_dir: str | Path) -> Path:
    files = sorted(Path(monitor_dir).glob("*_picks.json"))
    if not files:
        raise FileNotFoundError(f"No monitor pick payload found under {monitor_dir}.")
    def _score(path: Path) -> tuple[str, int, str]:
        match = re.search(r"(20\d{6})", path.name)
        date_key = match.group(1) if match else "00000000"
        priority = 0 if "_fast_" in path.name else 1
        return (date_key, priority, str(path))
    return sorted(files, key=_score)[-1]


def _related_metrics_payload(monitor_json_path: Path) -> dict:
    metrics_path = monitor_json_path.with_name(monitor_json_path.name.replace("_picks.json", "_metrics.json"))
    if metrics_path.exists():
        return _load_json(metrics_path)
    return {}


def _pick_series(df: pd.DataFrame, *candidates: str, default: str = "") -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index, dtype="object")


def main() -> None:
    args = build_parser().parse_args()
    monitor_json_path = Path(args.monitor_json) if args.monitor_json else _latest_monitor_json(args.monitor_dir)
    payload = _load_json(monitor_json_path)
    metrics_payload = _related_metrics_payload(monitor_json_path)
    monitor_summary = payload.get("summary", {})
    strategy_assessment = payload.get("strategy_assessment", {})
    selection_date = pd.Timestamp(monitor_summary["selection_date"])

    strategy_snapshot = pd.DataFrame(payload.get("strategy_snapshot", []) or payload.get("observation_pool", []))
    if strategy_snapshot.empty:
        raise ValueError(f"No strategy snapshot found in {monitor_json_path}.")
    strategy_snapshot["code"] = strategy_snapshot["code"].astype(str)
    strategy_snapshot["name"] = _pick_series(strategy_snapshot, "name").astype(str)
    strategy_snapshot["industry"] = _pick_series(strategy_snapshot, "industry_y", "industry_x", "industry", default="Unknown").astype(str)
    strategy_snapshot["target_weight"] = pd.to_numeric(strategy_snapshot["target_weight"], errors="coerce").fillna(0.0)
    strategy_snapshot["score"] = pd.to_numeric(strategy_snapshot["score"], errors="coerce").fillna(0.0)
    strategy_snapshot["trade_reason"] = _pick_series(strategy_snapshot, "trade_reason").astype(str)
    strategy_snapshot["risk_state"] = _pick_series(strategy_snapshot, "risk_state", default="full_risk").astype(str)

    market_df = CSVDataSource(args.slice_path, adjust=args.adjust).load()
    latest_rows = market_df.loc[market_df["date"] == selection_date].copy()
    if latest_rows.empty:
        raise ValueError(f"No market slice rows found for selection date {selection_date.date()} in {args.slice_path}.")

    paper_cfg_payload = _load_json(args.paper_config)
    backtest_cfg_payload = _load_json(args.backtest_config)
    paper_cfg = PaperRiskConfig(**paper_cfg_payload)
    execution_cfg = PaperExecutionConfig(
        commission=float(backtest_cfg_payload.get("commission", 0.0003)),
        slippage=float(backtest_cfg_payload.get("slippage", 0.0005)),
        sell_tax=float(backtest_cfg_payload.get("sell_tax", 0.001)),
    )

    state = load_paper_state(args.state_path, initial_cash=float(paper_cfg.initial_cash))
    state, executed_orders, blocked_orders = execute_due_orders(
        state=state,
        as_of_date=selection_date,
        latest_rows=latest_rows,
        config=execution_cfg,
    )
    state, nav_snapshot, holdings = mark_to_market(
        state=state,
        as_of_date=selection_date,
        latest_rows=latest_rows,
    )
    strategy_risk_state = str(strategy_snapshot["risk_state"].iloc[0]) if "risk_state" in strategy_snapshot.columns else "full_risk"
    risk_snapshot = assess_paper_risk(
        state=state,
        strategy_risk_state=strategy_risk_state,
        config=paper_cfg,
    )
    adjusted_targets, overlay_summary = apply_paper_risk_overlay(
        strategy_snapshot=strategy_snapshot,
        current_codes=set(state.get("positions", {}).keys()),
        risk_snapshot=risk_snapshot,
        config=paper_cfg,
    )

    execution_series = (
        pd.to_datetime(strategy_snapshot["execution_date"], errors="coerce")
        if "execution_date" in strategy_snapshot.columns
        else pd.Series(pd.NaT, index=strategy_snapshot.index)
    )
    execution_date = execution_series.min()
    if pd.isna(execution_date) or execution_date <= selection_date:
        execution_date = pd.Timestamp(selection_date) + pd.offsets.BDay(1)
    pending_orders, staged_orders = stage_orders_for_next_execution(
        state=state,
        targets=adjusted_targets,
        signal_date=selection_date,
        execution_date=execution_date,
        latest_rows=latest_rows,
        config=paper_cfg,
    )
    state["pending_orders"] = pending_orders
    save_paper_state(args.state_path, state)

    prefix = f"paper_monitor_{selection_date.strftime('%Y%m%d')}"
    outputs = write_paper_outputs(
        output_dir=args.output_dir,
        prefix=prefix,
        state=state,
        nav_snapshot=nav_snapshot,
        risk_snapshot=risk_snapshot,
        holdings=holdings,
        staged_orders=staged_orders,
        executed_orders=executed_orders,
        blocked_orders=blocked_orders,
        monitor_summary={
            "selection_date": selection_date.strftime("%Y-%m-%d"),
            "classification": strategy_assessment.get("classification", ""),
            "gross_annual_return": float(metrics_payload.get("gross_annual_return", 0.0) or 0.0),
            "annual_return": float(metrics_payload.get("annual_return", 0.0) or 0.0),
            "sharpe": float(metrics_payload.get("sharpe", 0.0) or 0.0),
            "max_drawdown": float(metrics_payload.get("max_drawdown", 0.0) or 0.0),
        },
        overlay_summary=overlay_summary,
    )

    summary_payload = {
        "selection_date": selection_date.strftime("%Y-%m-%d"),
        "paper_nav": float(nav_snapshot["nav"]),
        "paper_drawdown": float(risk_snapshot.drawdown),
        "paper_daily_return": float(risk_snapshot.daily_return),
        "portfolio_state": risk_snapshot.portfolio_state,
        "allow_new_entries": risk_snapshot.allow_new_entries,
        "pending_order_count": len(state.get("pending_orders", [])),
        "executed_order_count": int(len(executed_orders)),
        "blocked_order_count": int(len(blocked_orders)),
        "report_path": str(outputs["report_path"]),
        "state_path": str(Path(args.state_path)),
    }
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"[OK] paper monitor outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
