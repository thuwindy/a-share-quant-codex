from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def save_equity_curve_plot(curve: pd.DataFrame, output_path: str | Path) -> str:
    if curve is None or curve.empty:
        return ""
    out = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(curve["date"], curve["gross_equity"], label="gross_equity", linewidth=1.6, color="#999999")
    ax.plot(curve["date"], curve["equity"], label="equity", linewidth=2.0, color="#1f77b4")
    ax.set_title("Equity Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def save_drawdown_curve_plot(curve: pd.DataFrame, output_path: str | Path) -> str:
    if curve is None or curve.empty:
        return ""
    out = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.fill_between(curve["date"], curve["drawdown"], 0.0, color="#d62728", alpha=0.35, label="drawdown")
    ax.plot(curve["date"], curve["drawdown"], color="#d62728", linewidth=1.5)
    ax.set_title("Drawdown Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def save_calendar_bar_plot(
    metrics_table: pd.DataFrame,
    output_path: str | Path,
    *,
    value_col: str = "annual_return",
    title: str = "Calendar Return",
) -> str:
    if metrics_table is None or metrics_table.empty or value_col not in metrics_table.columns:
        return ""
    out = _ensure_parent(output_path)
    values = pd.to_numeric(metrics_table[value_col], errors="coerce").fillna(0.0)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(metrics_table["period"].astype(str), values, color=colors, alpha=0.9)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)
