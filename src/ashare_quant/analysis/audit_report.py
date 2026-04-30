from __future__ import annotations

from typing import Any

import pandas as pd


def _execution_lag_from_row(row: dict[str, Any]) -> int | None:
    if "execution_lag" in row and row["execution_lag"] is not None:
        try:
            return int(row["execution_lag"])
        except Exception:
            return None
    execution_time = str(row.get("execution_time", ""))
    if execution_time.startswith("t+"):
        token = execution_time.split(" ", 1)[0]
        try:
            return int(token.replace("t+", ""))
        except Exception:
            return None
    return None


def build_strategy_change_log(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    ordered = summary_df.copy().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for item in ordered.to_dict(orient="records"):
        current = {
            "scenario": str(item.get("scenario", "")),
            "description": str(item.get("description", "")),
            "annual_return": float(item.get("annual_return", 0.0) or 0.0),
            "sharpe": float(item.get("sharpe", 0.0) or 0.0),
            "max_drawdown": float(item.get("max_drawdown", 0.0) or 0.0),
            "avg_turnover": float(item.get("avg_turnover", 0.0) or 0.0),
            "after_cost_return_drag": float(item.get("after_cost_return_drag", 0.0) or 0.0),
        }
        if previous is None:
            rows.append(
                {
                    "scenario": current["scenario"],
                    "previous_scenario": "",
                    "change_summary": current["description"],
                    "impact_summary": "baseline",
                    "delta_annual_return": 0.0,
                    "delta_sharpe": 0.0,
                    "delta_max_drawdown": 0.0,
                    "delta_avg_turnover": 0.0,
                    "delta_cost_drag": 0.0,
                }
            )
        else:
            delta_annual = current["annual_return"] - previous["annual_return"]
            delta_sharpe = current["sharpe"] - previous["sharpe"]
            delta_drawdown = current["max_drawdown"] - previous["max_drawdown"]
            delta_turnover = current["avg_turnover"] - previous["avg_turnover"]
            delta_cost_drag = current["after_cost_return_drag"] - previous["after_cost_return_drag"]
            impact_parts = [
                f"sharpe {'up' if delta_sharpe >= 0 else 'down'} {abs(delta_sharpe):.4f}",
                f"return {'up' if delta_annual >= 0 else 'down'} {abs(delta_annual):.4f}",
                f"turnover {'down' if delta_turnover <= 0 else 'up'} {abs(delta_turnover):.4f}",
                f"cost_drag {'down' if delta_cost_drag <= 0 else 'up'} {abs(delta_cost_drag):.4f}",
            ]
            rows.append(
                {
                    "scenario": current["scenario"],
                    "previous_scenario": previous["scenario"],
                    "change_summary": current["description"],
                    "impact_summary": "; ".join(impact_parts),
                    "delta_annual_return": delta_annual,
                    "delta_sharpe": delta_sharpe,
                    "delta_max_drawdown": delta_drawdown,
                    "delta_avg_turnover": delta_turnover,
                    "delta_cost_drag": delta_cost_drag,
                }
            )
        previous = current
    return pd.DataFrame(rows)


def build_leakage_bias_checklist(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for item in summary_df.to_dict(orient="records"):
        execution_lag = _execution_lag_from_row(item)
        event_visibility = "PASS"
        execution_alignment = "PASS" if execution_lag is not None and execution_lag >= 1 else "WARN"
        survivor_guard = "PASS" if "tradeable_ratio" in item else "WARN"
        cost_declared = (
            "PASS"
            if ("avg_turnover" in item and "after_cost_return_drag" in item)
            else "WARN"
        )
        label_documented = "PASS" if str(item.get("label_type", "")).strip() else "WARN"
        warnings: list[str] = []
        if execution_alignment != "PASS":
            warnings.append("execution_lag_lt_1_or_missing")
        if survivor_guard != "PASS":
            warnings.append("tradeable_ratio_missing")
        if cost_declared != "PASS":
            warnings.append("cost_metrics_missing")
        if label_documented != "PASS":
            warnings.append("label_type_missing")
        rows.append(
            {
                "scenario": str(item.get("scenario", "")),
                "event_point_in_time": event_visibility,
                "execution_alignment": execution_alignment,
                "survivor_guard": survivor_guard,
                "cost_declared": cost_declared,
                "label_documented": label_documented,
                "overall_status": "WARN" if warnings else "PASS",
                "warnings": ",".join(warnings),
            }
        )
    return pd.DataFrame(rows)
