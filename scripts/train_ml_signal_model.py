from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.csv_adapter import CSVDataSource
from ashare_quant.models.offline_signal_model import save_offline_signal_artifact
from ashare_quant.pipeline import (
    DEFAULT_RESEARCH,
    _available_date_col,
    _fit_ranker,
    _label_col_for_horizon,
    load_json,
    prepare_research_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a past-only ML signal model and save the artifact.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--research-config", required=True)
    parser.add_argument("--output-model-path", required=True)
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--ranker-type", default=None, choices=["ml_ridge", "ml_logistic", "xgboost", "lightgbm"])
    parser.add_argument("--prediction-type", default=None, choices=["classification", "regression"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    research_cfg = DEFAULT_RESEARCH | load_json(args.research_config)
    if args.ranker_type:
        research_cfg["ranker_type"] = args.ranker_type
    if args.prediction_type:
        research_cfg["prediction_type"] = args.prediction_type

    data_source = CSVDataSource(args.data_path, adjust=research_cfg.get("adjust", "qfq"))
    df = data_source.load()
    frame, metadata = prepare_research_frame(df, research_cfg=research_cfg)

    horizon = int(research_cfg.get("label_horizon", metadata["label_horizons"][0]))
    if horizon not in metadata["horizon_feature_map"]:
        horizon = int(metadata["label_horizons"][0])
    feature_cols = metadata["horizon_feature_map"][horizon]
    label_col = _label_col_for_horizon(research_cfg, horizon)
    available_date_col = _available_date_col(horizon)

    train_df = frame.loc[frame[label_col].notna()].copy()
    if args.train_end_date:
        train_end = pd.Timestamp(args.train_end_date)
        train_df = train_df.loc[pd.to_datetime(train_df["date"]) <= train_end].copy()
    else:
        train_end = pd.Timestamp(train_df["date"].max())
    if train_df.empty:
        raise ValueError("No train rows available for offline model training.")

    # Important: offline training only. The online strategy path must load this artifact and predict only.
    ranker = _fit_ranker(
        df=frame,
        train_df=train_df,
        feature_cols=feature_cols,
        label_col=label_col,
        available_date_col=available_date_col,
        research_cfg=research_cfg,
    )

    output_path = save_offline_signal_artifact(
        args.output_model_path,
        ranker=ranker,
        factor_cols=feature_cols,
        label_col=label_col,
        model_type=str(research_cfg["ranker_type"]),
        prediction_type=str(
            args.prediction_type
            or research_cfg.get("prediction_type")
            or ("classification" if str(research_cfg["ranker_type"]) == "ml_logistic" else "regression")
        ),
        trained_until=train_end.strftime("%Y-%m-%d"),
        metadata={
            "research_config_path": str(Path(args.research_config)),
            "data_path": str(Path(args.data_path)),
            "horizon": horizon,
            "label_type": str(research_cfg["label_type"]),
            "signal_time": str(research_cfg["signal_time"]),
            "execution_price": str(research_cfg["execution_price"]),
            "execution_lag": int(research_cfg["execution_lag"]),
        },
    )
    print(
        {
            "status": "ok",
            "output_model_path": str(output_path),
            "model_type": research_cfg["ranker_type"],
            "prediction_type": args.prediction_type
            or research_cfg.get("prediction_type")
            or ("classification" if str(research_cfg["ranker_type"]) == "ml_logistic" else "regression"),
            "feature_count": len(feature_cols),
            "train_rows": int(len(train_df)),
            "trained_until": train_end.strftime("%Y-%m-%d"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
