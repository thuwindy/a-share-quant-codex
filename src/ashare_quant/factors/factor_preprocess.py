from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorPreprocessConfig:
    clip_method: str = "mad"
    mad_scale: float = 5.0
    quantile_lower: float = 0.01
    quantile_upper: float = 0.99
    min_regression_rows: int = 8
    use_rank_output: bool = True


def winsorize_series(
    series: pd.Series,
    method: str = "mad",
    mad_scale: float = 5.0,
    quantile_lower: float = 0.01,
    quantile_upper: float = 0.99,
) -> pd.Series:
    if series.dropna().empty:
        return series.copy()

    out = series.astype(float).copy()
    if method == "mad":
        median = float(out.median())
        mad = float((out - median).abs().median())
        if mad <= 0 or np.isnan(mad):
            return out
        scale = 1.4826 * mad
        lower = median - mad_scale * scale
        upper = median + mad_scale * scale
        return out.clip(lower=lower, upper=upper)
    if method == "quantile":
        lower = float(out.quantile(quantile_lower))
        upper = float(out.quantile(quantile_upper))
        return out.clip(lower=lower, upper=upper)
    raise ValueError(f"Unsupported clip method: {method}")


def zscore_series(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index, dtype=float)
    return (series - float(series.mean())) / std


def centered_rank_pct(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if valid.empty:
        return out
    if len(valid) == 1:
        out.loc[valid.index] = 0.0
        return out
    ranked = valid.rank(method="average")
    out.loc[valid.index] = (ranked - 1.0) / (len(valid) - 1.0) - 0.5
    return out


def build_cross_section_design(sl: pd.DataFrame) -> pd.DataFrame:
    x_num = pd.DataFrame({"log_mcap": np.log(pd.to_numeric(sl["market_cap"], errors="coerce").clip(lower=1.0))}, index=sl.index)
    x_ind = pd.get_dummies(sl["industry"].fillna("Unknown").astype(str), prefix="ind", dtype=float)
    return pd.concat([pd.Series(1.0, index=sl.index, name="intercept"), x_num, x_ind], axis=1)


def stable_cross_section_prediction(
    x_arr: np.ndarray,
    y: np.ndarray,
    *,
    min_regression_rows: int = 8,
) -> tuple[np.ndarray, np.ndarray | None]:
    mask = np.isfinite(y) & np.all(np.isfinite(x_arr), axis=1)
    required = min(min_regression_rows, x_arr.shape[1])
    if mask.sum() < required:
        return mask, None

    x_fit = np.asarray(x_arr[mask], dtype=float)
    y_fit = np.asarray(y[mask], dtype=float)
    x_all = np.asarray(x_arr, dtype=float)

    # Keep a deterministic intercept and drop only truly degenerate body columns.
    x_fit_body = x_fit[:, 1:] if x_fit.shape[1] > 1 else np.empty((len(x_fit), 0), dtype=float)
    x_all_body = x_all[:, 1:] if x_all.shape[1] > 1 else np.empty((len(x_all), 0), dtype=float)

    if x_fit_body.shape[1] > 0:
        body_std = np.nanstd(x_fit_body, axis=0)
        keep_body = np.isfinite(body_std) & (body_std > 1e-12)
        x_fit_body = x_fit_body[:, keep_body]
        x_all_body = x_all_body[:, keep_body]
    if x_fit_body.shape[1] > 0:
        body_mean = np.nanmean(x_fit_body, axis=0)
        body_std = np.nanstd(x_fit_body, axis=0)
        body_std = np.where(body_std > 1e-12, body_std, 1.0)
        x_fit_body = (x_fit_body - body_mean) / body_std
        x_all_body = (x_all_body - body_mean) / body_std

    x_fit_stable = np.concatenate([np.ones((len(x_fit), 1), dtype=float), x_fit_body], axis=1)
    x_all_stable = np.concatenate([np.ones((len(x_all), 1), dtype=float), x_all_body], axis=1)

    try:
        beta = np.linalg.pinv(x_fit_stable, rcond=1e-8) @ y_fit
        pred = x_all_stable @ beta
    except Exception:
        return mask, None
    if not np.all(np.isfinite(pred[mask])):
        return mask, None
    return mask, pred


def residualize_series(
    values: pd.Series,
    sl: pd.DataFrame,
    min_regression_rows: int = 8,
) -> pd.Series:
    x = build_cross_section_design(sl)
    x_arr = x.to_numpy(dtype=float)
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    residual = np.full(len(sl), np.nan, dtype=float)
    mask, pred = stable_cross_section_prediction(
        x_arr=x_arr,
        y=y,
        min_regression_rows=min_regression_rows,
    )
    if pred is not None:
        residual[mask] = y[mask] - pred[mask]
    elif mask.sum() > 0:
        residual[mask] = y[mask] - np.nanmean(y[mask])
    return pd.Series(residual, index=sl.index, dtype=float)


def preprocess_factor_cross_section(
    values: pd.Series,
    sl: pd.DataFrame,
    config: FactorPreprocessConfig | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    cfg = config or FactorPreprocessConfig()
    clipped = winsorize_series(
        values,
        method=cfg.clip_method,
        mad_scale=cfg.mad_scale,
        quantile_lower=cfg.quantile_lower,
        quantile_upper=cfg.quantile_upper,
    )
    residual = residualize_series(clipped, sl=sl, min_regression_rows=cfg.min_regression_rows)
    zscore = zscore_series(residual.fillna(0.0))
    ranked = centered_rank_pct(zscore)
    return residual, zscore, ranked if cfg.use_rank_output else zscore
