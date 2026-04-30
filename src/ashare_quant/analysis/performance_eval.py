from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_quant.backtest.metrics import summarize_returns


def build_equity_curve_frame(result: pd.DataFrame) -> pd.DataFrame:
    if result is None or result.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "gross_return",
                "net_return",
                "cost",
                "turnover",
                "gross_equity",
                "equity",
                "drawdown",
                "gross_drawdown",
                "max_position_weight",
            ]
        )
    frame = result.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["gross_equity"] = pd.to_numeric(frame.get("gross_equity"), errors="coerce").ffill().fillna(1.0)
    frame["equity"] = pd.to_numeric(frame.get("equity"), errors="coerce").ffill().fillna(1.0)
    gross_peak = frame["gross_equity"].cummax()
    net_peak = frame["equity"].cummax()
    frame["gross_drawdown"] = frame["gross_equity"] / gross_peak - 1.0
    frame["drawdown"] = frame["equity"] / net_peak - 1.0
    ordered_cols = [
        "date",
        "gross_return",
        "net_return",
        "cost",
        "turnover",
        "gross_equity",
        "equity",
        "gross_drawdown",
        "drawdown",
        "max_position_weight",
    ]
    for col in ordered_cols:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame[ordered_cols]


def build_calendar_metrics(
    result: pd.DataFrame,
    *,
    freq: str = "Y",
    annual_days: int = 252,
) -> pd.DataFrame:
    curve = build_equity_curve_frame(result)
    if curve.empty:
        return pd.DataFrame()
    period = curve["date"].dt.to_period(freq)
    rows: list[dict[str, object]] = []
    for key, sl in curve.groupby(period):
        metrics = summarize_returns(
            sl["net_return"],
            turnover=sl["turnover"],
            annual_days=annual_days,
            gross_returns=sl["gross_return"],
            concentration=sl["max_position_weight"],
            cost=sl["cost"],
        )
        rows.append(
            {
                "period": str(key),
                "start_date": pd.Timestamp(sl["date"].min()).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(sl["date"].max()).strftime("%Y-%m-%d"),
                "days": int(len(sl)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_performance_eval_markdown(
    yearly_metrics: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    *,
    equity_curve_csv: str | Path = "",
    equity_curve_png: str | Path = "",
    drawdown_curve_png: str | Path = "",
) -> str:
    lines = [
        "# Performance Evaluation",
        "",
        f"- equity_curve_csv: `{equity_curve_csv}`",
        f"- equity_curve_png: `{equity_curve_png}`",
        f"- drawdown_curve_png: `{drawdown_curve_png}`",
        "",
        "## Calendar Year Metrics",
        "",
    ]
    if yearly_metrics is not None and not yearly_metrics.empty:
        lines.extend(
            [
                "| period | annual_return | sharpe | max_drawdown | hit_rate | payoff_ratio | gross_annual_return | after_cost_return_drag |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in yearly_metrics.itertuples(index=False):
            lines.append(
                "| {period} | {annual_return:.4f} | {sharpe:.4f} | {max_drawdown:.4f} | {hit_rate:.4f} | {payoff_ratio:.4f} | {gross_annual_return:.4f} | {after_cost_return_drag:.4f} |".format(
                    period=getattr(row, "period"),
                    annual_return=float(getattr(row, "annual_return", 0.0)),
                    sharpe=float(getattr(row, "sharpe", 0.0)),
                    max_drawdown=float(getattr(row, "max_drawdown", 0.0)),
                    hit_rate=float(getattr(row, "hit_rate", 0.0)),
                    payoff_ratio=float(getattr(row, "payoff_ratio", 0.0)),
                    gross_annual_return=float(getattr(row, "gross_annual_return", 0.0)),
                    after_cost_return_drag=float(getattr(row, "after_cost_return_drag", 0.0)),
                )
            )
    else:
        lines.append("- no yearly metrics")
    lines.extend(["", "## Calendar Month Metrics", ""])
    if monthly_metrics is not None and not monthly_metrics.empty:
        lines.extend(
            [
                "| period | annual_return | sharpe | max_drawdown | hit_rate | payoff_ratio |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in monthly_metrics.itertuples(index=False):
            lines.append(
                "| {period} | {annual_return:.4f} | {sharpe:.4f} | {max_drawdown:.4f} | {hit_rate:.4f} | {payoff_ratio:.4f} |".format(
                    period=getattr(row, "period"),
                    annual_return=float(getattr(row, "annual_return", 0.0)),
                    sharpe=float(getattr(row, "sharpe", 0.0)),
                    max_drawdown=float(getattr(row, "max_drawdown", 0.0)),
                    hit_rate=float(getattr(row, "hit_rate", 0.0)),
                    payoff_ratio=float(getattr(row, "payoff_ratio", 0.0)),
                )
            )
    else:
        lines.append("- no monthly metrics")
    lines.append("")
    return "\n".join(lines)
