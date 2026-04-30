from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


STANDARD_FIELDS = (
    "annual_return",
    "sharpe",
    "max_drawdown",
    "turnover",
    "cost_drag",
    "payoff_ratio",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(out):
        return float(default)
    return out


def _derive_cost_drag(metrics: dict[str, Any]) -> float:
    if "cost_drag" in metrics:
        return _as_float(metrics.get("cost_drag"), 0.0)
    if "after_cost_return_drag" in metrics:
        return _as_float(metrics.get("after_cost_return_drag"), 0.0)
    if "gross_annual_return" in metrics and "annual_return" in metrics:
        return _as_float(metrics.get("gross_annual_return"), 0.0) - _as_float(metrics.get("annual_return"), 0.0)
    return 0.0


def build_standard_metric_pack(
    metrics: dict[str, Any],
    *,
    selection_date: str | None = None,
    source_metrics_json: str | Path | None = None,
    strategy_classification: str | None = None,
) -> dict[str, Any]:
    pack = {
        "selection_date": str(selection_date or ""),
        "annual_return": _as_float(metrics.get("annual_return"), 0.0),
        "sharpe": _as_float(metrics.get("sharpe"), 0.0),
        "max_drawdown": _as_float(metrics.get("max_drawdown"), 0.0),
        "turnover": _as_float(metrics.get("turnover", metrics.get("avg_turnover")), 0.0),
        "cost_drag": _derive_cost_drag(metrics),
        "gross_annual_return": _as_float(metrics.get("gross_annual_return"), 0.0),
        "annual_volatility": _as_float(metrics.get("annual_volatility"), 0.0),
        "hit_rate": _as_float(metrics.get("hit_rate"), 0.0),
        "payoff_ratio": _as_float(metrics.get("payoff_ratio"), 0.0),
        "signal_time": str(metrics.get("signal_time", "")),
        "execution_time": str(metrics.get("execution_time", "")),
        "holding_window": int(_as_float(metrics.get("holding_window"), 0)),
        "sleeve_count": int(_as_float(metrics.get("sleeve_count"), 0)),
        "strategy_classification": str(
            strategy_classification
            or metrics.get("strategy_assessment", {}).get("classification", "")
        ),
        "source_metrics_json": str(source_metrics_json or ""),
    }
    return pack


def metric_pack_dataframe(pack: dict[str, Any]) -> pd.DataFrame:
    ordered = {
        "selection_date": pack.get("selection_date", ""),
        "annual_return": _as_float(pack.get("annual_return"), 0.0),
        "sharpe": _as_float(pack.get("sharpe"), 0.0),
        "max_drawdown": _as_float(pack.get("max_drawdown"), 0.0),
        "turnover": _as_float(pack.get("turnover"), 0.0),
        "cost_drag": _as_float(pack.get("cost_drag"), 0.0),
        "gross_annual_return": _as_float(pack.get("gross_annual_return"), 0.0),
        "annual_volatility": _as_float(pack.get("annual_volatility"), 0.0),
        "hit_rate": _as_float(pack.get("hit_rate"), 0.0),
        "payoff_ratio": _as_float(pack.get("payoff_ratio"), 0.0),
        "signal_time": str(pack.get("signal_time", "")),
        "execution_time": str(pack.get("execution_time", "")),
        "holding_window": int(_as_float(pack.get("holding_window"), 0)),
        "sleeve_count": int(_as_float(pack.get("sleeve_count"), 0)),
        "strategy_classification": str(pack.get("strategy_classification", "")),
        "source_metrics_json": str(pack.get("source_metrics_json", "")),
    }
    return pd.DataFrame([ordered])


def metric_pack_markdown(pack: dict[str, Any]) -> str:
    lines = [
        "# Standard Metric Pack",
        "",
        f"- selection_date: `{pack.get('selection_date', '')}`",
        f"- annual_return: `{_as_float(pack.get('annual_return'), 0.0):.4f}`",
        f"- sharpe: `{_as_float(pack.get('sharpe'), 0.0):.4f}`",
        f"- max_drawdown: `{_as_float(pack.get('max_drawdown'), 0.0):.4f}`",
        f"- turnover: `{_as_float(pack.get('turnover'), 0.0):.4f}`",
        f"- cost_drag: `{_as_float(pack.get('cost_drag'), 0.0):.4f}`",
        "",
        "## Extended Fields",
        "",
        f"- gross_annual_return: `{_as_float(pack.get('gross_annual_return'), 0.0):.4f}`",
        f"- annual_volatility: `{_as_float(pack.get('annual_volatility'), 0.0):.4f}`",
        f"- hit_rate: `{_as_float(pack.get('hit_rate'), 0.0):.4f}`",
        f"- payoff_ratio: `{_as_float(pack.get('payoff_ratio'), 0.0):.4f}`",
        f"- signal_time: `{pack.get('signal_time', '')}`",
        f"- execution_time: `{pack.get('execution_time', '')}`",
        f"- holding_window: `{int(_as_float(pack.get('holding_window'), 0))}`",
        f"- sleeve_count: `{int(_as_float(pack.get('sleeve_count'), 0))}`",
        f"- strategy_classification: `{pack.get('strategy_classification', '')}`",
        f"- source_metrics_json: `{pack.get('source_metrics_json', '')}`",
        "",
    ]
    return "\n".join(lines)
