from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from ashare_quant.factors.factor_preprocess import build_cross_section_design, stable_cross_section_prediction


@dataclass(frozen=True)
class ExecutionSpec:
    signal_time: str = "close"
    execution_price: str = "close"
    execution_lag: int = 1
    holding_window: int = 5


def label_column_name(horizon: int, label_type: str) -> str:
    if label_type == "raw":
        return f"forward_return_{horizon}d"
    if label_type == "industry_excess":
        return f"forward_return_{horizon}d_industry_excess"
    if label_type == "neutralized_residual":
        return f"forward_return_{horizon}d_neutralized_residual"
    if re.fullmatch(r"(high|close)_\d+d_up", str(label_type)):
        return f"label_{label_type}"
    raise ValueError(f"Unsupported label_type: {label_type}")


def _entry_exit_columns(spec: ExecutionSpec) -> tuple[str, str]:
    if spec.execution_price == "close":
        return "research_close", "research_close"
    if spec.execution_price == "open":
        return "research_open", "research_close"
    if spec.execution_price == "vwap":
        return "research_vwap", "research_close"
    raise ValueError(f"Unsupported execution_price: {spec.execution_price}")


def _build_raw_label_for_horizon(df: pd.DataFrame, horizon: int, spec: ExecutionSpec) -> pd.DataFrame:
    out = df.sort_values(["code", "date"]).copy()
    entry_col, exit_col = _entry_exit_columns(spec)
    g = out.groupby("code", group_keys=False)

    if spec.execution_price == "close":
        entry_price = g[entry_col].shift(-spec.execution_lag)
        exit_price = g[exit_col].shift(-(spec.execution_lag + horizon))
        available_date = g["date"].shift(-(spec.execution_lag + horizon))
        execution_date = g["date"].shift(-spec.execution_lag)
    else:
        entry_price = g[entry_col].shift(-spec.execution_lag)
        exit_price = g[exit_col].shift(-horizon)
        available_date = g["date"].shift(-horizon)
        execution_date = g["date"].shift(-spec.execution_lag)

    raw_col = label_column_name(horizon, "raw")
    out[raw_col] = entry_price.where(entry_price != 0).rdiv(exit_price) - 1.0
    out[f"label_available_date_{horizon}d"] = pd.to_datetime(available_date)
    out[f"execution_date_{horizon}d"] = pd.to_datetime(execution_date)
    return out


def _add_industry_excess_label(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    raw_col = label_column_name(horizon, "raw")
    target_col = label_column_name(horizon, "industry_excess")
    out[target_col] = out[raw_col] - out.groupby(["date", "industry"])[raw_col].transform("mean")
    return out


def _add_neutralized_residual_label(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    raw_col = label_column_name(horizon, "raw")
    target_col = label_column_name(horizon, "neutralized_residual")
    out[target_col] = np.nan
    for _, idx in out.groupby("date").groups.items():
        sl = out.loc[idx]
        x = build_cross_section_design(sl)
        x_arr = x.to_numpy(dtype=float)
        y = pd.to_numeric(sl[raw_col], errors="coerce").to_numpy(dtype=float)
        residual = np.full(len(sl), np.nan, dtype=float)
        mask, pred = stable_cross_section_prediction(
            x_arr=x_arr,
            y=y,
            min_regression_rows=8,
        )
        if pred is not None:
            residual[mask] = y[mask] - pred[mask]
        elif mask.sum() > 0:
            residual[mask] = y[mask] - np.nanmean(y[mask])
        out.loc[idx, target_col] = residual
    return out


def _binary_up_label_spec(label_type: str) -> tuple[str, int]:
    match = re.fullmatch(r"(high|close)_(\d+)d_up", str(label_type))
    if not match:
        raise ValueError(f"Unsupported binary up label type: {label_type}")
    price_field = str(match.group(1))
    horizon = int(match.group(2))
    return price_field, horizon


def add_binary_up_label(
    df: pd.DataFrame,
    *,
    label_type: str,
    threshold_up: float = 0.05,
) -> pd.DataFrame:
    """Build a strict binary label from today's close to the future max move.

    Default financial meaning:
    - high_5d_up: if any future high in t+1..t+5 reaches +5%, label=1
    - close_5d_up: conservative version, use future close instead of future high
    """

    price_field, horizon = _binary_up_label_spec(label_type)
    out = df.sort_values(["code", "date"]).copy()
    base_close_col = "research_close" if "research_close" in out.columns else "close"
    future_price_col = f"research_{price_field}" if f"research_{price_field}" in out.columns else price_field
    g = out.groupby("code", group_keys=False)
    future_moves = []
    for step in range(1, horizon + 1):
        future_price = g[future_price_col].shift(-step)
        future_moves.append(pd.to_numeric(future_price, errors="coerce"))
    future_max_price = pd.concat(future_moves, axis=1).max(axis=1)
    max_up = future_max_price / pd.to_numeric(out[base_close_col], errors="coerce") - 1.0
    target_col = label_column_name(horizon, label_type)
    out[target_col] = np.where(max_up >= float(threshold_up), 1.0, 0.0)
    out.loc[max_up.isna(), target_col] = np.nan
    out[f"{target_col}_max_up"] = max_up
    out[f"{target_col}_available_date"] = pd.to_datetime(g["date"].shift(-horizon))
    return out


def add_label_columns(
    df: pd.DataFrame,
    horizons: list[int],
    spec: ExecutionSpec,
    label_types: list[str] | None = None,
    binary_up_threshold: float = 0.05,
) -> pd.DataFrame:
    out = df.copy()
    requested = set(label_types or ["raw", "industry_excess", "neutralized_residual"])
    requested.add("raw")
    for horizon in sorted(set(int(h) for h in horizons if int(h) > 0)):
        out = _build_raw_label_for_horizon(out, horizon=horizon, spec=spec)
        if "industry_excess" in requested or "neutralized_residual" in requested:
            out = _add_industry_excess_label(out, horizon=horizon)
        if "neutralized_residual" in requested:
            out = _add_neutralized_residual_label(out, horizon=horizon)
    binary_label_types = [label for label in requested if re.fullmatch(r"(high|close)_\d+d_up", str(label))]
    for binary_label_type in sorted(binary_label_types):
        out = add_binary_up_label(out, label_type=str(binary_label_type), threshold_up=float(binary_up_threshold))
    return out
