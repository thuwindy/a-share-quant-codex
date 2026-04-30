from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.csv_adapter import CSVDataSource, ensure_price_views
from ashare_quant.data.universe import apply_basic_universe_filters
from ashare_quant.factors.neutralize import neutralize_by_size_and_industry
from ashare_quant.labels.label_builder import add_label_columns
from ashare_quant.models.offline_signal_model import save_offline_signal_artifact
from ashare_quant.pipeline import (
    DEFAULT_RESEARCH,
    build_execution_spec,
    build_factor_preprocess_config,
    build_universe_filter_config,
    load_json,
    resolve_factor_columns,
)


@dataclass
class SplitEval:
    start: str
    end: str
    rows: int
    positive_ratio: float
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


@dataclass
class BucketHitRate:
    top_pct: float
    selected_rows: int
    hit_rate: float
    lift_vs_base: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train XGBoost/LightGBM binary model for 5-day up-prob scoring.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--research-config", required=True)
    parser.add_argument("--output-model-path", required=True)
    parser.add_argument("--model-type", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--train-start", default="2024-01-02")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--valid-start", default="2025-01-02")
    parser.add_argument("--valid-end", default="2025-09-30")
    parser.add_argument("--test-start", default="2025-10-01")
    parser.add_argument("--test-end", default="")
    parser.add_argument("--ml-label-type", choices=["high_5d_up", "close_5d_up"], default="high_5d_up")
    parser.add_argument("--threshold-up", type=float, default=0.05)
    parser.add_argument("--max-depth-grid", default="3,4,5")
    parser.add_argument("--learning-rate-grid", default="0.03,0.05,0.08")
    parser.add_argument("--n-estimators-grid", default="100,200,300")
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--time-cv-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _parse_grid(text: str, cast) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def _build_classifier(model_type: str, *, learning_rate: float, n_estimators: int, max_depth: int, scale_pos_weight: float, subsample: float, colsample_bytree: float, random_state: int):
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            learning_rate=float(learning_rate),
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_alpha=0.0,
            reg_lambda=1.0,
            scale_pos_weight=float(scale_pos_weight),
            random_state=int(random_state),
            n_jobs=1,
            verbosity=0,
        )
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            learning_rate=float(learning_rate),
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_alpha=0.0,
            reg_lambda=1.0,
            scale_pos_weight=float(scale_pos_weight),
            random_state=int(random_state),
            verbosity=-1,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _fit_with_early_stopping(model, x_train, y_train, x_valid, y_valid):
    if model.__class__.__name__.startswith("XGB"):
        model.set_params(early_stopping_rounds=30)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            verbose=False,
        )
        return model
    if model.__class__.__name__.startswith("LGBM"):
        from lightgbm import early_stopping, log_evaluation

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
        )
        return model
    model.fit(x_train, y_train)
    return model


def _evaluate(model, x: pd.DataFrame, y: pd.Series, dates: pd.Series) -> SplitEval:
    prob = model.predict_proba(x)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y, prob)) if y.nunique() > 1 else 0.0
    return SplitEval(
        start=pd.Timestamp(dates.min()).strftime("%Y-%m-%d"),
        end=pd.Timestamp(dates.max()).strftime("%Y-%m-%d"),
        rows=int(len(y)),
        positive_ratio=float(y.mean()) if len(y) else 0.0,
        accuracy=float(accuracy_score(y, pred)),
        auc=auc,
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        confusion_matrix=confusion_matrix(y, pred).astype(int).tolist(),
    )


def _evaluate_bucket_hit_rates(model, x: pd.DataFrame, y: pd.Series, *, buckets: tuple[float, ...] = (0.05, 0.10, 0.20)) -> list[BucketHitRate]:
    prob = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    target = pd.Series(y).astype(int).reset_index(drop=True)
    scored = pd.DataFrame({"prob": prob, "label": target})
    scored = scored.sort_values("prob", ascending=False, kind="mergesort").reset_index(drop=True)
    base_rate = float(scored["label"].mean()) if len(scored) else 0.0
    rows: list[BucketHitRate] = []
    for bucket in buckets:
        top_n = max(1, int(len(scored) * float(bucket)))
        subset = scored.iloc[:top_n]
        hit_rate = float(subset["label"].mean()) if len(subset) else 0.0
        lift = float(hit_rate / base_rate) if base_rate > 0 else 0.0
        rows.append(
            BucketHitRate(
                top_pct=float(bucket),
                selected_rows=int(len(subset)),
                hit_rate=hit_rate,
                lift_vs_base=lift,
            )
        )
    return rows


def _feature_importance(model, feature_cols: list[str]) -> list[dict[str, float]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    rows = []
    for feature, importance in zip(feature_cols, importances):
        rows.append({"feature": str(feature), "importance": float(importance)})
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows


def _time_mask(series: pd.Series, start: str, end: str | None) -> pd.Series:
    ts = pd.to_datetime(series)
    mask = ts >= pd.Timestamp(start)
    if end:
        mask &= ts <= pd.Timestamp(end)
    return mask


def _clip_raw_frame_by_dates(df: pd.DataFrame, *, start: str, end: str | None, lookback_days: int = 160, lookahead_days: int = 7) -> pd.DataFrame:
    date_col = "date" if "date" in df.columns else "trade_date" if "trade_date" in df.columns else None
    if date_col is None:
        return df
    ts = pd.to_datetime(df[date_col], errors="coerce")
    start_ts = pd.Timestamp(start) - pd.Timedelta(days=int(lookback_days))
    end_ts = (pd.Timestamp(end) if end else ts.max()) + pd.Timedelta(days=int(lookahead_days))
    mask = (ts >= start_ts) & (ts <= end_ts)
    return df.loc[mask].copy()


def _load_csv_range(
    path: str | Path,
    *,
    start: str,
    end: str | None,
    lookback_days: int = 160,
    lookahead_days: int = 7,
    chunksize: int = 300_000,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    csv_path = Path(path)
    effective_usecols = None
    if usecols:
        header_cols = pd.read_csv(csv_path, nrows=0).columns.astype(str).tolist()
        header_set = set(header_cols)
        effective_usecols = [col for col in usecols if col in header_set]
    start_ts = pd.Timestamp(start) - pd.Timedelta(days=int(lookback_days))
    end_ts = (pd.Timestamp(end) if end else None)
    if end_ts is not None:
        end_ts = end_ts + pd.Timedelta(days=int(lookahead_days))
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, chunksize=int(chunksize), low_memory=False, usecols=effective_usecols):
        date_col = "date" if "date" in chunk.columns else "trade_date" if "trade_date" in chunk.columns else None
        if date_col is None:
            raise ValueError("CSV has no date/trade_date column for range loading.")
        dates = pd.to_datetime(chunk[date_col], errors="coerce")
        mask = dates >= start_ts
        if end_ts is not None:
            mask &= dates <= end_ts
        clipped = chunk.loc[mask].copy()
        if not clipped.empty:
            frames.append(clipped)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _required_training_columns(research_cfg: dict[str, Any]) -> list[str]:
    raw_factor_cols, _ = resolve_factor_columns(research_cfg)
    cols = {
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "amount",
        "market_cap",
        "industry",
        "is_st",
        "is_suspended",
        "can_buy",
        "can_sell",
        "list_date",
        "up_limit",
        "down_limit",
    }
    if "smart_money_inflow_20" in raw_factor_cols:
        cols.add("smart_money_inflow_20")
    return sorted(cols)


def _lean_ml_feature_frame(df: pd.DataFrame, research_cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    raw_factor_cols, factor_cols = resolve_factor_columns(research_cfg)
    out = ensure_price_views(df)
    for col in ("open", "high", "low", "close", "research_open", "research_high", "research_low", "research_close", "amount", "market_cap"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    if "code" in out.columns:
        out["code"] = out["code"].astype("category")
    if "industry" in out.columns:
        out["industry"] = out["industry"].fillna("Unknown").astype("category")
    out, _ = apply_basic_universe_filters(
        out,
        config=build_universe_filter_config(research_cfg),
        return_summary=True,
    )
    print(f"[lean] universe filtered rows: {len(out)}", flush=True)
    out = out.sort_values(["code", "date"]).copy()
    g = out.groupby("code", group_keys=False)

    close_col = "research_close" if "research_close" in out.columns else "close"
    out["ret_1"] = g[close_col].pct_change(fill_method=None)

    if "momentum_20" in raw_factor_cols:
        out["momentum_20"] = g[close_col].pct_change(20, fill_method=None)
    if "momentum_60" in raw_factor_cols:
        out["momentum_60"] = g[close_col].pct_change(60, fill_method=None)
    if "volatility_20" in raw_factor_cols:
        out["volatility_20"] = g["ret_1"].transform(lambda s: s.rolling(20, min_periods=10).std())
    if "turnover_20" in raw_factor_cols:
        out["turnover_1"] = pd.to_numeric(out["amount"], errors="coerce") / pd.to_numeric(
            out["market_cap"], errors="coerce"
        ).replace(0.0, np.nan)
        out["turnover_20"] = g["turnover_1"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    if "smart_money_inflow_20" in raw_factor_cols and "smart_money_inflow_20" not in out.columns:
        out["smart_money_inflow_20"] = np.nan

    missing_raw = [col for col in raw_factor_cols if col not in out.columns]
    for col in missing_raw:
        out[col] = np.nan
    print("[lean] factors built", flush=True)

    out = neutralize_by_size_and_industry(
        out,
        raw_factor_cols,
        preprocess_config=build_factor_preprocess_config(research_cfg),
    )
    print("[lean] neutralized", flush=True)
    out = add_label_columns(
        out,
        horizons=[],
        spec=build_execution_spec(research_cfg),
        label_types=[str(research_cfg.get("ml_label_type", "high_5d_up"))],
        binary_up_threshold=float(research_cfg.get("ml_threshold_up", 0.05)),
    )
    print("[lean] labels built", flush=True)
    drop_cols = [col for col in ("ret_1", "turnover_1") if col in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out, factor_cols


def _walk_forward_select(train_df: pd.DataFrame, feature_cols: list[str], label_col: str, *, model_type: str, learning_rate_grid: list[float], n_estimators_grid: list[int], max_depth_grid: list[int], subsample: float, colsample_bytree: float, random_state: int, time_cv_splits: int) -> tuple[Any, dict[str, Any], float]:
    x = train_df[feature_cols].astype(float).fillna(0.0)
    y = train_df[label_col].astype(int)
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    scale_pos_weight = float(negative / max(positive, 1))
    tscv = TimeSeriesSplit(n_splits=max(int(time_cv_splits), 2))
    best_auc = float("-inf")
    best_params: dict[str, Any] = {}
    for learning_rate in learning_rate_grid:
        for n_estimators in n_estimators_grid:
            for max_depth in max_depth_grid:
                fold_aucs: list[float] = []
                for fit_idx, valid_idx in tscv.split(x):
                    x_fit = x.iloc[fit_idx]
                    y_fit = y.iloc[fit_idx]
                    x_valid = x.iloc[valid_idx]
                    y_valid = y.iloc[valid_idx]
                    if y_fit.nunique() < 2 or y_valid.nunique() < 2:
                        continue
                    model = _build_classifier(
                        model_type,
                        learning_rate=float(learning_rate),
                        n_estimators=int(n_estimators),
                        max_depth=int(max_depth),
                        scale_pos_weight=scale_pos_weight,
                        subsample=float(subsample),
                        colsample_bytree=float(colsample_bytree),
                        random_state=int(random_state),
                    )
                    model = _fit_with_early_stopping(model, x_fit, y_fit, x_valid, y_valid)
                    prob = model.predict_proba(x_valid)[:, 1]
                    fold_aucs.append(float(roc_auc_score(y_valid, prob)))
                avg_auc = float(np.mean(fold_aucs)) if fold_aucs else float("-inf")
                if avg_auc > best_auc:
                    best_auc = avg_auc
                    best_params = {
                        "learning_rate": float(learning_rate),
                        "n_estimators": int(n_estimators),
                        "max_depth": int(max_depth),
                        "scale_pos_weight": scale_pos_weight,
                    }
    if not best_params:
        raise ValueError("No valid time-series CV fold was produced for binary classifier training.")
    return _build_classifier(
        model_type,
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        scale_pos_weight=best_params["scale_pos_weight"],
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        random_state=int(random_state),
    ), best_params, best_auc


def main() -> int:
    args = build_parser()
    research_cfg = DEFAULT_RESEARCH | load_json(args.research_config)
    research_cfg["use_ml_score"] = True
    research_cfg["ml_label_type"] = str(args.ml_label_type)
    research_cfg["ml_threshold_up"] = float(args.threshold_up)

    final_end = str(args.test_end).strip() or None
    print(f"[train] load raw data: {args.data_path}", flush=True)
    if str(args.data_path).lower().endswith(".csv"):
        df = _load_csv_range(
            args.data_path,
            start=str(args.train_start),
            end=final_end,
            usecols=_required_training_columns(research_cfg),
        )
    else:
        data_source = CSVDataSource(args.data_path, adjust=research_cfg.get("adjust", "qfq"))
        df = data_source.load()
        df = _clip_raw_frame_by_dates(df, start=str(args.train_start), end=final_end)
    print(f"[train] clipped raw rows: {len(df)}", flush=True)
    print("[train] build lean ml frame", flush=True)
    frame, factor_cols = _lean_ml_feature_frame(df, research_cfg)
    label_col = f"label_{args.ml_label_type}"

    dataset = frame.loc[frame[label_col].notna()].copy()
    dataset["date"] = pd.to_datetime(dataset["date"])
    print(f"[train] labeled rows: {len(dataset)}", flush=True)
    train_df = dataset.loc[_time_mask(dataset["date"], args.train_start, args.train_end)].copy()
    valid_df = dataset.loc[_time_mask(dataset["date"], args.valid_start, args.valid_end)].copy()
    test_df = dataset.loc[_time_mask(dataset["date"], args.test_start, final_end)].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Train/valid/test split has empty data. Check date ranges.")
    print(f"[train] split rows train={len(train_df)} valid={len(valid_df)} test={len(test_df)}", flush=True)

    learning_rate_grid = _parse_grid(args.learning_rate_grid, float)
    n_estimators_grid = _parse_grid(args.n_estimators_grid, int)
    max_depth_grid = _parse_grid(args.max_depth_grid, int)

    print("[train] walk-forward select params", flush=True)
    model, best_params, cv_auc = _walk_forward_select(
        train_df=train_df,
        feature_cols=factor_cols,
        label_col=label_col,
        model_type=str(args.model_type),
        learning_rate_grid=learning_rate_grid,
        n_estimators_grid=n_estimators_grid,
        max_depth_grid=max_depth_grid,
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        random_state=int(args.random_state),
        time_cv_splits=int(args.time_cv_splits),
    )
    x_train = train_df[factor_cols].astype(float).fillna(0.0)
    y_train = train_df[label_col].astype(int)
    x_valid = valid_df[factor_cols].astype(float).fillna(0.0)
    y_valid = valid_df[label_col].astype(int)
    x_test = test_df[factor_cols].astype(float).fillna(0.0)
    y_test = test_df[label_col].astype(int)

    print("[train] fit final model", flush=True)
    model = _fit_with_early_stopping(model, x_train, y_train, x_valid, y_valid)
    print("[train] evaluate", flush=True)
    train_eval = _evaluate(model, x_train, y_train, train_df["date"])
    valid_eval = _evaluate(model, x_valid, y_valid, valid_df["date"])
    test_eval = _evaluate(model, x_test, y_test, test_df["date"])
    train_buckets = _evaluate_bucket_hit_rates(model, x_train, y_train)
    valid_buckets = _evaluate_bucket_hit_rates(model, x_valid, y_valid)
    test_buckets = _evaluate_bucket_hit_rates(model, x_test, y_test)
    feature_importance = _feature_importance(model, factor_cols)

    print(f"[train] save artifact: {args.output_model_path}", flush=True)
    save_offline_signal_artifact(
        args.output_model_path,
        ranker=model,
        factor_cols=factor_cols,
        label_col=label_col,
        model_type=str(args.model_type),
        prediction_type="classification",
        trained_until=train_eval.end,
        metadata={
            "train_eval": asdict(train_eval),
            "valid_eval": asdict(valid_eval),
            "test_eval": asdict(test_eval),
            "train_bucket_hit_rates": [asdict(item) for item in train_buckets],
            "valid_bucket_hit_rates": [asdict(item) for item in valid_buckets],
            "test_bucket_hit_rates": [asdict(item) for item in test_buckets],
            "feature_importance": feature_importance,
            "cv_auc": cv_auc,
            "best_params": best_params,
            "ml_label_type": str(args.ml_label_type),
            "threshold_up": float(args.threshold_up),
            "note": "high_5d_up uses future high within 5 days; close_5d_up is the more conservative close-only version.",
        },
    )

    summary = {
        "model_type": str(args.model_type),
        "label_type": str(args.ml_label_type),
        "threshold_up": float(args.threshold_up),
        "feature_count": len(factor_cols),
        "train": asdict(train_eval),
        "valid": asdict(valid_eval),
        "test": asdict(test_eval),
        "train_bucket_hit_rates": [asdict(item) for item in train_buckets],
        "valid_bucket_hit_rates": [asdict(item) for item in valid_buckets],
        "test_bucket_hit_rates": [asdict(item) for item in test_buckets],
        "feature_importance": feature_importance,
        "cv_auc": cv_auc,
        "best_params": best_params,
        "output_model_path": str(Path(args.output_model_path)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
