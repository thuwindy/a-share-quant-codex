from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.execution.base import Order
from ashare_quant.execution.paper import PaperExecutionAdapter, PaperExecutionConfig


@dataclass(frozen=True)
class PaperRiskConfig:
    initial_cash: float = 1_000_000.0
    min_cash_buffer: float = 0.05
    min_trade_notional: float = 5_000.0
    max_new_positions_per_day: int = 3
    drawdown_soft_limit: float = 0.08
    drawdown_hard_limit: float = 0.12
    daily_loss_soft_limit: float = 0.015
    daily_loss_hard_limit: float = 0.03
    soft_risk_multiplier: float = 0.50
    hard_risk_multiplier: float = 0.0
    block_new_entries_on_soft_risk: bool = True
    block_new_entries_on_low_risk: bool = True


@dataclass(frozen=True)
class PaperRiskSnapshot:
    portfolio_state: str
    strategy_risk_state: str
    target_risk_multiplier: float
    allow_new_entries: bool
    current_nav: float
    peak_nav: float
    drawdown: float
    daily_return: float
    cash_ratio: float
    alerts: list[str]


def _ts(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value)


def _state_template(initial_cash: float) -> dict[str, Any]:
    return {
        "cash": float(initial_cash),
        "peak_nav": float(initial_cash),
        "last_nav": float(initial_cash),
        "last_mark_date": None,
        "positions": {},
        "pending_orders": [],
        "fills": [],
        "nav_history": [],
    }


def load_paper_state(path: str | Path, initial_cash: float) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return _state_template(initial_cash)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    state = _state_template(initial_cash)
    state.update(payload)
    state["cash"] = float(state.get("cash", initial_cash))
    state["peak_nav"] = float(state.get("peak_nav", max(state["cash"], initial_cash)))
    state["last_nav"] = float(state.get("last_nav", state["cash"]))
    state["positions"] = {
        str(code): {
            "quantity": float(info.get("quantity", 0.0)),
            "avg_price": float(info.get("avg_price", 0.0)),
            "name": info.get("name", ""),
            "industry": info.get("industry", ""),
        }
        for code, info in dict(state.get("positions", {})).items()
        if float(info.get("quantity", 0.0)) > 0.0
    }
    state["pending_orders"] = list(state.get("pending_orders", []))
    state["fills"] = list(state.get("fills", []))
    state["nav_history"] = list(state.get("nav_history", []))
    return state


def save_paper_state(path: str | Path, state: dict[str, Any]) -> Path:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state_path


def _adapter_from_state(state: dict[str, Any], config: PaperExecutionConfig) -> PaperExecutionAdapter:
    adapter = PaperExecutionAdapter(initial_cash=float(state.get("cash", 0.0)), config=config)
    adapter.cash_ = float(state.get("cash", 0.0))
    adapter.positions_ = {
        code: float(info.get("quantity", 0.0))
        for code, info in dict(state.get("positions", {})).items()
        if float(info.get("quantity", 0.0)) > 0.0
    }
    adapter.avg_prices_ = {
        code: float(info.get("avg_price", 0.0))
        for code, info in dict(state.get("positions", {})).items()
        if float(info.get("quantity", 0.0)) > 0.0
    }
    return adapter


def _sync_state_from_adapter(
    state: dict[str, Any],
    adapter: PaperExecutionAdapter,
    latest_meta: dict[str, dict[str, Any]],
    fills: list[dict[str, Any]],
) -> dict[str, Any]:
    positions: dict[str, dict[str, Any]] = {}
    for code, qty in adapter.positions_.items():
        meta = latest_meta.get(code, {})
        positions[code] = {
            "quantity": float(qty),
            "avg_price": float(adapter.avg_prices_.get(code, 0.0)),
            "name": str(meta.get("name", "")),
            "industry": str(meta.get("industry", "")),
        }
    state["cash"] = float(adapter.cash_)
    state["positions"] = positions
    state["fills"] = fills
    return state


def build_market_snapshot(latest_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    latest_rows = latest_rows.copy()
    industry_col = "industry"
    if industry_col not in latest_rows.columns:
        if "industry_y" in latest_rows.columns:
            industry_col = "industry_y"
        elif "industry_x" in latest_rows.columns:
            industry_col = "industry_x"
    rows = {}
    for row in latest_rows.itertuples(index=False):
        code = str(getattr(row, "code"))
        raw_price = pd.to_numeric(getattr(row, "trade_close", getattr(row, "close", 0.0)), errors="coerce")
        trade_close = float(raw_price) if pd.notna(raw_price) else 0.0
        rows[code] = {
            "trade_close": trade_close,
            "can_buy": bool(getattr(row, "can_buy", True)),
            "can_sell": bool(getattr(row, "can_sell", True)),
            "name": str(getattr(row, "name", "") or ""),
            "industry": str(getattr(row, industry_col, "") or "Unknown"),
        }
    return rows


def execute_due_orders(
    state: dict[str, Any],
    as_of_date: str | pd.Timestamp,
    latest_rows: pd.DataFrame,
    config: PaperExecutionConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    execution_cfg = config or PaperExecutionConfig()
    as_of_ts = _ts(as_of_date)
    market_snapshot = build_market_snapshot(latest_rows)
    latest_meta = {code: {"name": info["name"], "industry": info["industry"]} for code, info in market_snapshot.items()}
    adapter = _adapter_from_state(state, execution_cfg)
    stored_fills = list(state.get("fills", []))

    due_orders = []
    future_orders = []
    for record in list(state.get("pending_orders", [])):
        execution_date = _ts(record["execution_date"])
        if execution_date <= as_of_ts:
            due_orders.append(record)
        else:
            future_orders.append(record)

    due_orders = sorted(due_orders, key=lambda item: 0 if str(item.get("side", "")).upper() == "SELL" else 1)
    executed_records: list[dict[str, Any]] = []
    blocked_records: list[dict[str, Any]] = []
    for record in due_orders:
        code = str(record["code"])
        side = str(record["side"]).upper()
        market = market_snapshot.get(code)
        if market is None or market["trade_close"] <= 0.0:
            blocked_records.append({**record, "blocked_reason": "missing_price"})
            continue
        if side == "BUY" and not market["can_buy"]:
            blocked_records.append({**record, "blocked_reason": "cannot_buy"})
            continue
        if side == "SELL" and not market["can_sell"]:
            blocked_records.append({**record, "blocked_reason": "cannot_sell"})
            continue
        try:
            fill = adapter.submit_orders(
                [
                    Order(
                        trading_day=as_of_ts,
                        code=code,
                        side=side,
                        quantity=float(record["quantity"]),
                        reference_price=float(market["trade_close"]),
                    )
                ]
            )[0]
            fill_dict = {
                "trading_day": fill.trading_day.strftime("%Y-%m-%d"),
                "code": fill.code,
                "side": fill.side,
                "quantity": float(fill.quantity),
                "fill_price": float(fill.fill_price),
                "notional": float(fill.notional),
                "cost": float(fill.cost),
                "signal_date": record.get("signal_date", ""),
                "scheduled_execution_date": record.get("execution_date", ""),
                "trade_reason": record.get("trade_reason", ""),
                "risk_state": record.get("risk_state", ""),
            }
            executed_records.append(fill_dict)
            stored_fills.append(fill_dict)
        except ValueError as exc:
            blocked_records.append({**record, "blocked_reason": str(exc)})

    state["pending_orders"] = future_orders
    state = _sync_state_from_adapter(state, adapter, latest_meta=latest_meta, fills=stored_fills)
    return state, pd.DataFrame(executed_records), pd.DataFrame(blocked_records)


def mark_to_market(
    state: dict[str, Any],
    as_of_date: str | pd.Timestamp,
    latest_rows: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    as_of_ts = _ts(as_of_date)
    market_snapshot = build_market_snapshot(latest_rows)
    cash = float(state.get("cash", 0.0))
    positions = dict(state.get("positions", {}))
    records: list[dict[str, Any]] = []
    market_value = 0.0
    for code, info in positions.items():
        market = market_snapshot.get(code, {})
        last_price = float(market.get("trade_close", info.get("avg_price", 0.0)))
        quantity = float(info.get("quantity", 0.0))
        position_value = quantity * last_price
        market_value += position_value
        records.append(
            {
                "code": code,
                "name": market.get("name", info.get("name", "")),
                "industry": market.get("industry", info.get("industry", "Unknown")),
                "quantity": quantity,
                "avg_price": float(info.get("avg_price", 0.0)),
                "last_price": last_price,
                "market_value": position_value,
                "pnl": position_value - quantity * float(info.get("avg_price", 0.0)),
            }
        )

    nav = cash + market_value
    previous_nav = float(state.get("last_nav", nav)) if state.get("last_mark_date") else nav
    daily_return = nav / previous_nav - 1.0 if previous_nav > 0 else 0.0
    peak_nav = max(float(state.get("peak_nav", nav)), nav)
    drawdown = nav / peak_nav - 1.0 if peak_nav > 0 else 0.0
    cash_ratio = cash / nav if nav > 0 else 1.0

    nav_entry = {
        "date": as_of_ts.strftime("%Y-%m-%d"),
        "nav": float(nav),
        "cash": float(cash),
        "market_value": float(market_value),
        "daily_return": float(daily_return),
        "drawdown": float(drawdown),
        "cash_ratio": float(cash_ratio),
    }
    history = [row for row in list(state.get("nav_history", [])) if row.get("date") != nav_entry["date"]]
    history.append(nav_entry)
    state["nav_history"] = history
    state["peak_nav"] = float(peak_nav)
    state["last_nav"] = float(nav)
    state["last_mark_date"] = nav_entry["date"]

    holdings = pd.DataFrame(records).sort_values("market_value", ascending=False).reset_index(drop=True) if records else pd.DataFrame(
        columns=["code", "name", "industry", "quantity", "avg_price", "last_price", "market_value", "pnl"]
    )
    return state, nav_entry, holdings


def assess_paper_risk(
    state: dict[str, Any],
    strategy_risk_state: str,
    config: PaperRiskConfig | None = None,
) -> PaperRiskSnapshot:
    cfg = config or PaperRiskConfig()
    nav_history = list(state.get("nav_history", []))
    current = nav_history[-1] if nav_history else {
        "nav": cfg.initial_cash,
        "drawdown": 0.0,
        "daily_return": 0.0,
        "cash_ratio": 1.0,
    }
    current_nav = float(current.get("nav", cfg.initial_cash))
    peak_nav = float(state.get("peak_nav", max(current_nav, cfg.initial_cash)))
    drawdown = float(current.get("drawdown", 0.0))
    daily_return = float(current.get("daily_return", 0.0))
    cash_ratio = float(current.get("cash_ratio", 1.0))

    strategy_risk_state = str(strategy_risk_state or "full_risk")
    multiplier_map = {"full_risk": 1.0, "half_risk": cfg.soft_risk_multiplier, "low_risk": 0.0}
    target_multiplier = float(multiplier_map.get(strategy_risk_state, 1.0))
    allow_new_entries = not (strategy_risk_state == "low_risk" and cfg.block_new_entries_on_low_risk)
    alerts: list[str] = []
    portfolio_state = strategy_risk_state

    if drawdown <= -abs(cfg.drawdown_hard_limit):
        portfolio_state = "capital_preservation"
        target_multiplier = min(target_multiplier, float(cfg.hard_risk_multiplier))
        allow_new_entries = False
        alerts.append("paper_drawdown_hard_limit")
    elif drawdown <= -abs(cfg.drawdown_soft_limit):
        portfolio_state = "drawdown_watch"
        target_multiplier = min(target_multiplier, float(cfg.soft_risk_multiplier))
        if cfg.block_new_entries_on_soft_risk:
            allow_new_entries = False
        alerts.append("paper_drawdown_soft_limit")

    if daily_return <= -abs(cfg.daily_loss_hard_limit):
        portfolio_state = "capital_preservation"
        target_multiplier = min(target_multiplier, float(cfg.hard_risk_multiplier))
        allow_new_entries = False
        alerts.append("paper_daily_loss_hard_limit")
    elif daily_return <= -abs(cfg.daily_loss_soft_limit):
        if portfolio_state == "full_risk":
            portfolio_state = "loss_watch"
        target_multiplier = min(target_multiplier, float(cfg.soft_risk_multiplier))
        if cfg.block_new_entries_on_soft_risk:
            allow_new_entries = False
        alerts.append("paper_daily_loss_soft_limit")

    return PaperRiskSnapshot(
        portfolio_state=portfolio_state,
        strategy_risk_state=strategy_risk_state,
        target_risk_multiplier=float(max(target_multiplier, 0.0)),
        allow_new_entries=bool(allow_new_entries),
        current_nav=current_nav,
        peak_nav=peak_nav,
        drawdown=drawdown,
        daily_return=daily_return,
        cash_ratio=cash_ratio,
        alerts=alerts,
    )


def apply_paper_risk_overlay(
    strategy_snapshot: pd.DataFrame,
    current_codes: set[str],
    risk_snapshot: PaperRiskSnapshot,
    config: PaperRiskConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config or PaperRiskConfig()
    if strategy_snapshot is None or strategy_snapshot.empty:
        return pd.DataFrame(), {"dropped_new_entries": [], "scaled_gross_exposure": 0.0}

    out = strategy_snapshot.copy()
    out["paper_target_weight"] = pd.to_numeric(out["target_weight"], errors="coerce").fillna(0.0)
    out["paper_target_weight"] *= float(risk_snapshot.target_risk_multiplier)

    dropped_new_entries: list[str] = []
    if not risk_snapshot.allow_new_entries:
        keep_mask = out["code"].astype(str).isin(current_codes)
        dropped_new_entries = out.loc[~keep_mask, "code"].astype(str).tolist()
        out = out.loc[keep_mask].copy()
    else:
        new_mask = ~out["code"].astype(str).isin(current_codes)
        new_entries = out.loc[new_mask].sort_values("score", ascending=False)
        allowed_new = max(int(cfg.max_new_positions_per_day), 0)
        if allowed_new >= 0 and len(new_entries) > allowed_new:
            keep_codes = set(new_entries.head(allowed_new)["code"].astype(str))
            drop_mask = new_mask & ~out["code"].astype(str).isin(keep_codes)
            dropped_new_entries = out.loc[drop_mask, "code"].astype(str).tolist()
            out = out.loc[~drop_mask].copy()

    investable_cap = max(0.0, 1.0 - float(cfg.min_cash_buffer))
    gross_exposure = float(out["paper_target_weight"].sum())
    if gross_exposure > investable_cap and gross_exposure > 0:
        out["paper_target_weight"] *= investable_cap / gross_exposure
        gross_exposure = investable_cap

    out = out.loc[out["paper_target_weight"] > 1e-8].copy()
    return out.reset_index(drop=True), {
        "dropped_new_entries": dropped_new_entries,
        "scaled_gross_exposure": gross_exposure,
    }


def stage_orders_for_next_execution(
    state: dict[str, Any],
    targets: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    execution_date: str | pd.Timestamp,
    latest_rows: pd.DataFrame,
    config: PaperRiskConfig | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    cfg = config or PaperRiskConfig()
    signal_ts = _ts(signal_date)
    execution_ts = _ts(execution_date)
    market_snapshot = build_market_snapshot(latest_rows)
    nav = float(state.get("last_nav", state.get("cash", cfg.initial_cash)))
    positions = dict(state.get("positions", {}))
    target_map = dict(zip(targets["code"].astype(str), pd.to_numeric(targets["paper_target_weight"], errors="coerce").fillna(0.0)))
    score_map = dict(zip(targets["code"].astype(str), pd.to_numeric(targets["score"], errors="coerce").fillna(0.0)))
    reason_map = dict(zip(targets["code"].astype(str), targets.get("trade_reason", pd.Series("", index=targets.index)).astype(str)))
    risk_state_map = dict(zip(targets["code"].astype(str), targets.get("risk_state", pd.Series("full_risk", index=targets.index)).astype(str)))
    name_map = dict(zip(targets["code"].astype(str), targets.get("name", pd.Series("", index=targets.index)).astype(str)))

    orders: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    all_codes = set(positions) | set(target_map)
    for code in sorted(all_codes):
        market = market_snapshot.get(code, {})
        reference_price = float(market.get("trade_close", positions.get(code, {}).get("avg_price", 0.0)))
        if reference_price <= 0.0:
            continue
        current_qty = float(positions.get(code, {}).get("quantity", 0.0))
        current_value = current_qty * reference_price
        desired_value = nav * float(target_map.get(code, 0.0))
        delta_notional = desired_value - current_value
        if abs(delta_notional) < float(cfg.min_trade_notional):
            continue
        side = "BUY" if delta_notional > 0 else "SELL"
        quantity = abs(delta_notional) / reference_price
        if quantity <= 1e-12:
            continue
        trade_reason = reason_map.get(code, "forced_exit" if side == "SELL" else "new_entry_stronger")
        record = {
            "signal_date": signal_ts.strftime("%Y-%m-%d"),
            "execution_date": execution_ts.strftime("%Y-%m-%d"),
            "code": code,
            "name": name_map.get(code, market.get("name", positions.get(code, {}).get("name", ""))),
            "industry": market.get("industry", positions.get(code, {}).get("industry", "Unknown")),
            "side": side,
            "quantity": float(quantity),
            "reference_price_signal": float(reference_price),
            "trade_reason": trade_reason,
            "risk_state": risk_state_map.get(code, "full_risk"),
            "target_weight": float(target_map.get(code, 0.0)),
            "score": float(score_map.get(code, 0.0)),
            "delta_notional": float(delta_notional),
        }
        orders.append(record)
        order_rows.append(record)
    orders = sorted(orders, key=lambda item: 0 if item["side"] == "SELL" else 1)
    return orders, pd.DataFrame(order_rows)


def write_paper_outputs(
    output_dir: str | Path,
    prefix: str,
    state: dict[str, Any],
    nav_snapshot: dict[str, Any],
    risk_snapshot: PaperRiskSnapshot,
    holdings: pd.DataFrame,
    staged_orders: pd.DataFrame,
    executed_orders: pd.DataFrame,
    blocked_orders: pd.DataFrame,
    monitor_summary: dict[str, Any],
    overlay_summary: dict[str, Any],
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    holdings_path = out_dir / f"{prefix}_holdings.csv"
    staged_path = out_dir / f"{prefix}_staged_orders.csv"
    executed_path = out_dir / f"{prefix}_executed_orders.csv"
    blocked_path = out_dir / f"{prefix}_blocked_orders.csv"
    metrics_path = out_dir / f"{prefix}_paper_metrics.json"
    report_path = out_dir / f"{prefix}_paper_report.md"

    holdings.to_csv(holdings_path, index=False)
    staged_orders.to_csv(staged_path, index=False)
    executed_orders.to_csv(executed_path, index=False)
    blocked_orders.to_csv(blocked_path, index=False)
    metrics_payload = {
        "nav_snapshot": nav_snapshot,
        "risk_snapshot": asdict(risk_snapshot),
        "monitor_summary": monitor_summary,
        "overlay_summary": overlay_summary,
        "pending_order_count": len(state.get("pending_orders", [])),
        "position_count": len(state.get("positions", {})),
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Dynamic Paper Monitor",
        "",
        "## Risk-First Summary",
        "",
        f"- selection date: `{monitor_summary.get('selection_date', '')}`",
        f"- paper NAV: `{nav_snapshot.get('nav', 0.0):,.2f}`",
        f"- peak NAV: `{risk_snapshot.peak_nav:,.2f}`",
        f"- drawdown: `{risk_snapshot.drawdown:.2%}`",
        f"- daily return: `{risk_snapshot.daily_return:.2%}`",
        f"- cash ratio: `{risk_snapshot.cash_ratio:.2%}`",
        f"- strategy risk state: `{risk_snapshot.strategy_risk_state}`",
        f"- paper portfolio state: `{risk_snapshot.portfolio_state}`",
        f"- allow new entries: `{risk_snapshot.allow_new_entries}`",
        f"- target risk multiplier: `{risk_snapshot.target_risk_multiplier:.2f}`",
        f"- alerts: `{', '.join(risk_snapshot.alerts) if risk_snapshot.alerts else 'none'}`",
        "",
        "## Monitor Context",
        "",
        f"- monitor classification: `{monitor_summary.get('classification', '')}`",
        f"- gross annual return: `{monitor_summary.get('gross_annual_return', 0.0):.2%}`",
        f"- net annual return: `{monitor_summary.get('annual_return', 0.0):.2%}`",
        f"- Sharpe: `{monitor_summary.get('sharpe', 0.0):.4f}`",
        f"- max drawdown: `{monitor_summary.get('max_drawdown', 0.0):.2%}`",
        "",
        "## Current Holdings",
        "",
    ]
    if holdings.empty:
        lines.append("_No active positions._")
    else:
        lines.extend(
            [
                "| code | name | industry | quantity | avg_price | last_price | market_value | pnl |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in holdings.itertuples(index=False):
            lines.append(
                "| {code} | {name} | {industry} | {quantity:.2f} | {avg_price:.2f} | {last_price:.2f} | {market_value:,.2f} | {pnl:,.2f} |".format(
                    code=getattr(row, "code"),
                    name=getattr(row, "name"),
                    industry=getattr(row, "industry"),
                    quantity=float(getattr(row, "quantity")),
                    avg_price=float(getattr(row, "avg_price")),
                    last_price=float(getattr(row, "last_price")),
                    market_value=float(getattr(row, "market_value")),
                    pnl=float(getattr(row, "pnl")),
                )
            )
    lines.extend(
        [
            "",
            "## Orders Executed Today",
            "",
        ]
    )
    if executed_orders.empty:
        lines.append("_No paper fills were executed today._")
    else:
        lines.extend(
            [
                "| code | side | quantity | fill_price | cost | trade_reason |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in executed_orders.itertuples(index=False):
            lines.append(
                "| {code} | {side} | {quantity:.2f} | {fill_price:.2f} | {cost:,.2f} | {trade_reason} |".format(
                    code=getattr(row, "code"),
                    side=getattr(row, "side"),
                    quantity=float(getattr(row, "quantity")),
                    fill_price=float(getattr(row, "fill_price")),
                    cost=float(getattr(row, "cost")),
                    trade_reason=getattr(row, "trade_reason", ""),
                )
            )
    lines.extend(
        [
            "",
            "## Staged Orders For Next Session",
            "",
            f"- dropped new entries by risk overlay: `{', '.join(overlay_summary.get('dropped_new_entries', [])) or 'none'}`",
            f"- staged gross exposure: `{overlay_summary.get('scaled_gross_exposure', 0.0):.2%}`",
            "",
        ]
    )
    if staged_orders.empty:
        lines.append("_No staged orders for the next execution window._")
    else:
        lines.extend(
            [
                "| code | side | quantity | target_weight | score | trade_reason | risk_state |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in staged_orders.itertuples(index=False):
            lines.append(
                "| {code} | {side} | {quantity:.2f} | {target_weight:.3f} | {score:.4f} | {trade_reason} | {risk_state} |".format(
                    code=getattr(row, "code"),
                    side=getattr(row, "side"),
                    quantity=float(getattr(row, "quantity")),
                    target_weight=float(getattr(row, "target_weight")),
                    score=float(getattr(row, "score")),
                    trade_reason=getattr(row, "trade_reason", ""),
                    risk_state=getattr(row, "risk_state", ""),
                )
            )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "holdings_path": holdings_path,
        "staged_orders_path": staged_path,
        "executed_orders_path": executed_path,
        "blocked_orders_path": blocked_path,
        "metrics_path": metrics_path,
        "report_path": report_path,
    }
