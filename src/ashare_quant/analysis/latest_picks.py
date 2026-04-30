from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.analysis.research_report import extract_display_weights
from ashare_quant.data.base import MarketDataSource
from ashare_quant.data.csv_adapter import CSVDataSource
from ashare_quant.models.offline_signal_model import OfflineSignalPredictor
from ashare_quant.portfolio.portfolio_optimizer import PortfolioConfig, optimize_portfolio_weights
from ashare_quant.pipeline import DEFAULT_RESEARCH, prepare_research_frame, restrict_train_dates, score_research_frame

ROOT = Path(__file__).resolve().parents[3]
PREMIUM_DIR = ROOT / "data" / "premium_v22"


@dataclass(frozen=True)
class LatestPickSummary:
    layer: str
    selection_date: str
    train_start_date: str
    train_end_date: str
    train_rows: int
    train_dates: int
    candidate_count: int
    selected_count: int
    top_n: int
    label_horizon: int
    label_type: str
    adjust_mode: str
    data_path: str
    signal_time: str
    execution_time: str
    system_mode: str
    main_score_frozen: bool
    execution_layer_frozen: bool
    observation_layer_frozen: bool
    new_research_gate_passed: bool
    new_research_gate_reason: str


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_research_config(path: str | Path) -> dict[str, Any]:
    return {**DEFAULT_RESEARCH, **load_json(path)}


def _prepare_latest_context(
    data_path: str | Path,
    research_config_path: str | Path,
    data_source: MarketDataSource | None = None,
    data_adjust: str = "none",
    prediction_date: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], pd.Timestamp, list[pd.Timestamp], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    research_cfg = _load_research_config(research_config_path)
    if data_source is not None:
        df = data_source.load_daily_bars()
    else:
        df = CSVDataSource(data_path, adjust=data_adjust).load()
    df, metadata = prepare_research_frame(df, research_cfg)
    candidate_dates = sorted(df.loc[df["tradeable"], "date"].drop_duplicates().tolist())
    if not candidate_dates:
        raise ValueError("No eligible prediction dates were found after factor construction.")

    if prediction_date is None:
        selection_ts = pd.Timestamp(candidate_dates[-1])
    else:
        selection_ts = pd.Timestamp(prediction_date)
        if selection_ts not in candidate_dates:
            raise ValueError(
                f"Prediction date {selection_ts.date()} is not available in the eligible universe. "
                f"Latest eligible date is {pd.Timestamp(candidate_dates[-1]).date()}."
            )

    train_dates = set(date for date in candidate_dates if pd.Timestamp(date) < selection_ts)
    train_dates = restrict_train_dates(
        train_dates=train_dates,
        train_window_years=int(research_cfg.get("train_window_years", 0)),
    )
    scored_df, score_details = score_research_frame(
        df=df,
        research_cfg=research_cfg,
        metadata=metadata,
        train_dates=train_dates,
    )
    train_df = scored_df.loc[scored_df["date"] < selection_ts].copy()
    return research_cfg, scored_df, metadata, selection_ts, candidate_dates, train_df, score_details, df


def _label_col(research_cfg: dict[str, Any], horizon: int) -> str:
    if research_cfg["label_type"] == "raw":
        return f"forward_return_{horizon}d"
    return {
        "industry_excess": f"forward_return_{horizon}d_industry_excess",
        "neutralized_residual": f"forward_return_{horizon}d_neutralized_residual",
    }[research_cfg["label_type"]]


def _next_execution_date(selection_ts: pd.Timestamp, candidate_dates: list[pd.Timestamp], lag: int) -> pd.Timestamp:
    if selection_ts in candidate_dates:
        idx = candidate_dates.index(selection_ts)
        target_idx = idx + lag
        if target_idx < len(candidate_dates):
            return pd.Timestamp(candidate_dates[target_idx])
    return pd.Timestamp(selection_ts) + pd.offsets.BDay(lag)


def _annotate_ml_observation_tags(selected: pd.DataFrame, research_cfg: dict[str, Any]) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out
    if not bool(research_cfg.get("observe_ml_score", False)):
        return out

    model_path = str(
        research_cfg.get("ml_observer_model_path")
        or research_cfg.get("ml_model_path")
        or ""
    ).strip()
    if not model_path:
        return out

    predictor = OfflineSignalPredictor.load(path=model_path)
    out["ml_score"] = predictor.predict(out)
    out["ml_rank"] = out["ml_score"].rank(method="first", ascending=False).astype(int)
    out["ml_observation_tag"] = "中性观察"

    observe_top_n = max(1, int(research_cfg.get("ml_observer_top_n", 20)))
    high_rank_cutoff = max(1, int(research_cfg.get("ml_observer_high_rank_cutoff", 10)))
    high_ml_rank_cutoff = max(1, int(research_cfg.get("ml_observer_high_ml_rank_cutoff", 10)))
    spike_ml_rank_cutoff = max(1, int(research_cfg.get("ml_observer_spike_ml_rank_cutoff", 5)))

    within_window = out["rank"] <= observe_top_n
    original_high = within_window & (out["rank"] <= high_rank_cutoff)
    ml_high = within_window & (out["ml_rank"] <= high_ml_rank_cutoff)
    ml_spike = within_window & (out["ml_rank"] <= spike_ml_rank_cutoff)

    out.loc[original_high & ml_high, "ml_observation_tag"] = "原排序高 + ml_score高"
    out.loc[original_high & ~ml_high, "ml_observation_tag"] = "原排序高 + ml_score低"
    out.loc[(within_window & ~original_high & ml_spike), "ml_observation_tag"] = "原排序一般 + ml_score异常高"
    out.loc[~within_window, "ml_observation_tag"] = ""
    return out


def _annotate_industry_breadth_observation(selected: pd.DataFrame, research_cfg: dict[str, Any]) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out
    if not bool(research_cfg.get("observe_industry_breadth", True)):
        return out
    if "industry_breadth_5" not in out.columns:
        return out

    out["industry_breadth_score"] = pd.to_numeric(out["industry_breadth_5"], errors="coerce")
    out["industry_breadth_rank"] = out["industry_breadth_score"].rank(method="first", ascending=False)
    out["industry_breadth_rank"] = out["industry_breadth_rank"].fillna(0).astype(int)

    strong_threshold = float(research_cfg.get("industry_breadth_strong_threshold", 0.60))
    medium_threshold = float(research_cfg.get("industry_breadth_medium_threshold", 0.50))

    out["industry_breadth_tag"] = "板块确认弱"
    out.loc[out["industry_breadth_score"] >= medium_threshold, "industry_breadth_tag"] = "板块确认一般"
    out.loc[out["industry_breadth_score"] >= strong_threshold, "industry_breadth_tag"] = "板块共振强"

    score_pct = (out["industry_breadth_score"].fillna(0.0) * 100.0).round(1)
    out["industry_breadth_detail"] = (
        "所属行业近5日强势占比 "
        + score_pct.astype(str)
        + "%，"
        + out["industry_breadth_tag"].astype(str)
    )
    return out


def _annotate_industry_leader_follow_observation(selected: pd.DataFrame, research_cfg: dict[str, Any]) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out
    if not bool(research_cfg.get("observe_industry_leader_follow", True)):
        return out
    if "industry_leader_follow_5" not in out.columns:
        return out

    out["industry_leader_follow_score"] = pd.to_numeric(out["industry_leader_follow_5"], errors="coerce")
    out["industry_leader_follow_rank"] = out["industry_leader_follow_score"].rank(method="first", ascending=False)
    out["industry_leader_follow_rank"] = out["industry_leader_follow_rank"].fillna(0).astype(int)

    strong_threshold = float(research_cfg.get("industry_leader_follow_strong_threshold", 0.30))
    medium_threshold = float(research_cfg.get("industry_leader_follow_medium_threshold", 0.10))

    out["industry_leader_follow_tag"] = "龙头扩散弱"
    out.loc[out["industry_leader_follow_score"] >= medium_threshold, "industry_leader_follow_tag"] = "龙头扩散一般"
    out.loc[out["industry_leader_follow_score"] >= strong_threshold, "industry_leader_follow_tag"] = "龙头扩散强"

    score_text = out["industry_leader_follow_score"].fillna(0.0).map(lambda x: f"{x:.4f}")
    out["industry_leader_follow_detail"] = (
        "行业龙头跟随扩散分="
        + score_text.astype(str)
        + "，主池扩散排序第 "
        + out["industry_leader_follow_rank"].astype(str)
        + " 名，"
        + out["industry_leader_follow_tag"].astype(str)
    )
    return out


def _annotate_overhead_density_observation(
    selected: pd.DataFrame,
    research_cfg: dict[str, Any],
    history_df: pd.DataFrame,
    selection_ts: pd.Timestamp,
    overhead_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out
    if not bool(research_cfg.get("observe_overhead_density", True)):
        return out

    if overhead_history is not None and not overhead_history.empty:
        keyed = overhead_history.copy()
        keyed["date"] = pd.to_datetime(keyed["date"])
        keyed["code"] = keyed["code"].astype(str)
        lookup = keyed.loc[keyed["date"] == selection_ts, ["code", "overhead_density_20"]].drop_duplicates("code")
        score_series = out["code"].map(lookup.set_index("code")["overhead_density_20"]) if not lookup.empty else pd.Series(np.nan, index=out.index)
    elif "overhead_density_20" in out.columns and pd.to_numeric(out["overhead_density_20"], errors="coerce").notna().any():
        score_series = pd.to_numeric(out["overhead_density_20"], errors="coerce")
    else:
        codes = set(out["code"].astype(str))
        hist = history_df.loc[
            history_df["code"].astype(str).isin(codes)
            & (pd.to_datetime(history_df["date"]) >= selection_ts - pd.Timedelta(days=45))
            & (pd.to_datetime(history_df["date"]) <= selection_ts),
            ["date", "code", "close", "amount", "market_cap"],
        ].copy()
        path = PREMIUM_DIR / "tushare_chip_daily.csv"
        chip = _load_recent_premium_rows(
            path,
            codes=codes,
            usecols=["date", "code", "cost_95pct", "weight_avg", "winner_rate"],
            date_col="date",
            start_date=selection_ts - pd.Timedelta(days=45),
            end_date=selection_ts,
        )
        if hist.empty or chip.empty:
            return out
        hist["date"] = pd.to_datetime(hist["date"])
        chip["date"] = pd.to_datetime(chip["date"])
        merged = hist.merge(chip, on=["date", "code"], how="left")
        if merged.empty:
            return out
        for col in ("close", "amount", "market_cap", "cost_95pct", "weight_avg", "winner_rate"):
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged["winner_rate_frac"] = (merged["winner_rate"] / 100.0).clip(lower=0.0, upper=1.0)
        merged["chip_upper_gap"] = ((merged["cost_95pct"] - merged["close"]) / merged["close"]).clip(lower=0.0)
        merged["chip_avg_gap"] = ((merged["weight_avg"] - merged["close"]) / merged["close"]).clip(lower=0.0)
        merged["winner_deficit"] = (1.0 - merged["winner_rate_frac"]).clip(lower=0.0, upper=1.0)
        merged["scaled_upper_gap"] = (merged["chip_upper_gap"] / 0.15).clip(lower=0.0, upper=1.0)
        merged["scaled_avg_gap"] = (merged["chip_avg_gap"] / 0.10).clip(lower=0.0, upper=1.0)
        merged["pressure_snapshot"] = (
            0.50 * merged["winner_deficit"].fillna(0.0)
            + 0.30 * merged["scaled_upper_gap"].fillna(0.0)
            + 0.20 * merged["scaled_avg_gap"].fillna(0.0)
        ).clip(lower=0.0, upper=1.0)
        merged["turnover_1"] = merged["amount"] / merged["market_cap"].replace(0.0, np.nan)
        merged["weighted_pressure"] = merged["pressure_snapshot"] * merged["turnover_1"].clip(lower=0.0).fillna(0.0)
        merged = merged.sort_values(["code", "date"]).copy()
        weighted_sum = merged.groupby("code")["weighted_pressure"].transform(lambda s: s.rolling(20, min_periods=10).sum())
        turnover_sum = merged.groupby("code")["turnover_1"].transform(lambda s: s.rolling(20, min_periods=10).sum())
        merged["overhead_density_20"] = (weighted_sum / (turnover_sum + 1e-12)).clip(lower=0.0, upper=1.0)
        latest = merged.sort_values(["code", "date"]).groupby("code", as_index=False).tail(1)
        score_series = out["code"].map(latest.set_index("code")["overhead_density_20"])

    out["overhead_density_score"] = pd.to_numeric(score_series, errors="coerce")
    out["overhead_density_rank"] = out["overhead_density_score"].rank(method="first", ascending=False)
    out["overhead_density_rank"] = out["overhead_density_rank"].fillna(0).astype(int)

    strong_threshold = float(research_cfg.get("overhead_density_strong_threshold", 0.55))
    medium_threshold = float(research_cfg.get("overhead_density_medium_threshold", 0.35))

    out["overhead_density_tag"] = "兑现压力轻"
    out.loc[out["overhead_density_score"] >= medium_threshold, "overhead_density_tag"] = "兑现压力一般"
    out.loc[out["overhead_density_score"] >= strong_threshold, "overhead_density_tag"] = "兑现压力大"

    score_text = out["overhead_density_score"].fillna(0.0).map(lambda x: f"{x:.4f}")
    out["overhead_density_detail"] = (
        "20日兑现压力密度="
        + score_text.astype(str)
        + "，主池压力排序第 "
        + out["overhead_density_rank"].astype(str)
        + " 名，"
        + out["overhead_density_tag"].astype(str)
    )
    return out


def _build_overhead_density_history(
    scored_df: pd.DataFrame,
    *,
    codes: set[str],
    selection_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    if not codes or not selection_dates:
        return pd.DataFrame(columns=["date", "code", "overhead_density_20"])
    scoped = scored_df.loc[
        scored_df["code"].astype(str).isin(codes)
        & scored_df["date"].isin({pd.Timestamp(v) for v in selection_dates}),
        ["date", "code", "overhead_density_20"] if "overhead_density_20" in scored_df.columns else ["date", "code"],
    ].copy()
    if "overhead_density_20" in scoped.columns and pd.to_numeric(scoped["overhead_density_20"], errors="coerce").notna().any():
        scoped["overhead_density_20"] = pd.to_numeric(scoped["overhead_density_20"], errors="coerce")
        return scoped.loc[:, ["date", "code", "overhead_density_20"]]

    min_date = min(pd.Timestamp(v) for v in selection_dates) - pd.Timedelta(days=45)
    max_date = max(pd.Timestamp(v) for v in selection_dates)
    hist = scored_df.loc[
        scored_df["code"].astype(str).isin(codes)
        & (pd.to_datetime(scored_df["date"]) >= min_date)
        & (pd.to_datetime(scored_df["date"]) <= max_date),
        ["date", "code", "close", "amount", "market_cap"],
    ].copy()
    path = PREMIUM_DIR / "tushare_chip_daily.csv"
    chip = _load_recent_premium_rows(
        path,
        codes=codes,
        usecols=["date", "code", "cost_95pct", "weight_avg", "winner_rate"],
        date_col="date",
        start_date=min_date,
        end_date=max_date,
    )
    if hist.empty or chip.empty:
        return pd.DataFrame(columns=["date", "code", "overhead_density_20"])
    hist["date"] = pd.to_datetime(hist["date"])
    chip["date"] = pd.to_datetime(chip["date"])
    merged = hist.merge(chip, on=["date", "code"], how="left")
    if merged.empty:
        return pd.DataFrame(columns=["date", "code", "overhead_density_20"])
    for col in ("close", "amount", "market_cap", "cost_95pct", "weight_avg", "winner_rate"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["winner_rate_frac"] = (merged["winner_rate"] / 100.0).clip(lower=0.0, upper=1.0)
    merged["chip_upper_gap"] = ((merged["cost_95pct"] - merged["close"]) / merged["close"]).clip(lower=0.0)
    merged["chip_avg_gap"] = ((merged["weight_avg"] - merged["close"]) / merged["close"]).clip(lower=0.0)
    merged["winner_deficit"] = (1.0 - merged["winner_rate_frac"]).clip(lower=0.0, upper=1.0)
    merged["scaled_upper_gap"] = (merged["chip_upper_gap"] / 0.15).clip(lower=0.0, upper=1.0)
    merged["scaled_avg_gap"] = (merged["chip_avg_gap"] / 0.10).clip(lower=0.0, upper=1.0)
    merged["pressure_snapshot"] = (
        0.50 * merged["winner_deficit"].fillna(0.0)
        + 0.30 * merged["scaled_upper_gap"].fillna(0.0)
        + 0.20 * merged["scaled_avg_gap"].fillna(0.0)
    ).clip(lower=0.0, upper=1.0)
    merged["turnover_1"] = merged["amount"] / merged["market_cap"].replace(0.0, np.nan)
    merged["weighted_pressure"] = merged["pressure_snapshot"] * merged["turnover_1"].clip(lower=0.0).fillna(0.0)
    merged = merged.sort_values(["code", "date"]).copy()
    weighted_sum = merged.groupby("code")["weighted_pressure"].transform(lambda s: s.rolling(20, min_periods=10).sum())
    turnover_sum = merged.groupby("code")["turnover_1"].transform(lambda s: s.rolling(20, min_periods=10).sum())
    merged["overhead_density_20"] = (weighted_sum / (turnover_sum + 1e-12)).clip(lower=0.0, upper=1.0)
    return merged.loc[merged["date"].isin({pd.Timestamp(v) for v in selection_dates}), ["date", "code", "overhead_density_20"]].copy()


def _annotate_moneyflow_observation(selected: pd.DataFrame, research_cfg: dict[str, Any]) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out
    if not bool(research_cfg.get("observe_moneyflow_confirmation", True)):
        return out

    source_col = ""
    for candidate in ("smart_money_inflow_20", "smart_money_inflow_20_neu"):
        if candidate in out.columns:
            source_col = candidate
            break
    if not source_col:
        return out

    out["moneyflow_confirmation_score"] = pd.to_numeric(out[source_col], errors="coerce")
    out["moneyflow_confirmation_rank"] = out["moneyflow_confirmation_score"].rank(method="first", ascending=False)
    out["moneyflow_confirmation_rank"] = out["moneyflow_confirmation_rank"].fillna(0).astype(int)

    strong_rank_cutoff = max(1, int(research_cfg.get("moneyflow_strong_rank_cutoff", 5)))
    medium_rank_cutoff = max(strong_rank_cutoff, int(research_cfg.get("moneyflow_medium_rank_cutoff", 10)))

    out["moneyflow_confirmation_tag"] = "资金确认弱"
    positive_score = out["moneyflow_confirmation_score"].fillna(0.0) > 0
    out.loc[positive_score, "moneyflow_confirmation_tag"] = "资金确认一般"
    out.loc[
        positive_score & (out["moneyflow_confirmation_rank"] <= strong_rank_cutoff),
        "moneyflow_confirmation_tag",
    ] = "资金确认强"
    out.loc[
        positive_score
        & (out["moneyflow_confirmation_rank"] > strong_rank_cutoff)
        & (out["moneyflow_confirmation_rank"] <= medium_rank_cutoff),
        "moneyflow_confirmation_tag",
    ] = "资金确认一般"

    score_text = out["moneyflow_confirmation_score"].fillna(0.0).map(lambda x: f"{x:.4f}")
    out["moneyflow_confirmation_detail"] = (
        "20日聪明钱流入="
        + score_text.astype(str)
        + "，主池资金排序第 "
        + out["moneyflow_confirmation_rank"].astype(str)
        + " 名，"
        + out["moneyflow_confirmation_tag"].astype(str)
    )
    return out


def _load_recent_premium_rows(
    path: Path,
    *,
    codes: set[str],
    usecols: list[str],
    date_col: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    if not path.exists() or not codes:
        return pd.DataFrame(columns=usecols)
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        frame = chunk.copy()
        frame["code"] = frame["code"].astype(str)
        frame = frame.loc[frame["code"].isin(codes)]
        if frame.empty:
            continue
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        if start_date is not None:
            frame = frame.loc[frame[date_col] >= start_date]
        if end_date is not None:
            frame = frame.loc[frame[date_col] <= end_date]
        if not frame.empty:
            pieces.append(frame)
    if not pieces:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values([date_col, "code"]).reset_index(drop=True)
    return out


def _latest_by_code(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.sort_values([date_col, "code"]).groupby("code", as_index=False).tail(1)
    return ranked.reset_index(drop=True)


def _fmt_wan(value: Any) -> str:
    return f"{float(value):.1f}万"


def _fmt_pct(value: Any) -> str:
    return f"{float(value):.1f}%"


def _annotate_premium_flow_confirmation(selected: pd.DataFrame, selection_ts: pd.Timestamp) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out
    path = PREMIUM_DIR / "tushare_flow_daily.csv"
    usecols = [
        "date",
        "code",
        "net_amount",
        "net_d5_amount",
        "buy_lg_amount_rate",
        "buy_lg_amount",
        "pct_change",
    ]
    codes = set(out["code"].astype(str))
    frame = _load_recent_premium_rows(
        path,
        codes=codes,
        usecols=usecols,
        date_col="date",
        start_date=selection_ts - pd.Timedelta(days=30),
        end_date=selection_ts,
    )
    if frame.empty:
        return out
    latest = _latest_by_code(frame, "date")
    for col in ("net_amount", "net_d5_amount", "buy_lg_amount_rate", "buy_lg_amount", "pct_change"):
        latest[col] = pd.to_numeric(latest[col], errors="coerce")
    latest["premium_flow_score"] = (
        0.45 * (latest["buy_lg_amount_rate"].fillna(0.0) / 100.0).clip(-1.0, 1.0)
        + 0.35 * (latest["net_amount"].fillna(0.0) / 50000.0).clip(-1.0, 1.0)
        + 0.20 * (latest["net_d5_amount"].fillna(0.0) / 100000.0).clip(-1.0, 1.0)
    )
    latest["premium_flow_rank"] = latest["premium_flow_score"].rank(method="first", ascending=False).astype(int)
    latest["moneyflow_confirmation_score"] = latest["premium_flow_score"]
    latest["moneyflow_confirmation_rank"] = latest["premium_flow_rank"]
    latest["moneyflow_confirmation_tag"] = "资金确认弱"
    medium_mask = (
        (latest["net_amount"].fillna(0.0) > 0)
        | (latest["net_d5_amount"].fillna(0.0) > 0)
        | (latest["buy_lg_amount_rate"].fillna(0.0) >= 10.0)
    )
    strong_mask = (
        (latest["net_amount"].fillna(0.0) > 0)
        & (latest["net_d5_amount"].fillna(0.0) > 0)
        & (latest["buy_lg_amount_rate"].fillna(0.0) >= 20.0)
    )
    latest.loc[medium_mask, "moneyflow_confirmation_tag"] = "资金确认一般"
    latest.loc[strong_mask, "moneyflow_confirmation_tag"] = "资金确认强"
    latest["moneyflow_confirmation_detail"] = (
        "Tushare资金流("
        + latest["date"].dt.strftime("%Y-%m-%d")
        + ")：单日净流入 "
        + latest["net_amount"].fillna(0.0).map(_fmt_wan)
        + "，5日净流入 "
        + latest["net_d5_amount"].fillna(0.0).map(_fmt_wan)
        + "，大单买入占比 "
        + latest["buy_lg_amount_rate"].fillna(0.0).map(_fmt_pct)
        + "，"
        + latest["moneyflow_confirmation_tag"].astype(str)
    )
    latest = latest.loc[
        :,
        [
            "code",
            "moneyflow_confirmation_score",
            "moneyflow_confirmation_rank",
            "moneyflow_confirmation_tag",
            "moneyflow_confirmation_detail",
        ],
    ]
    out = out.drop(
        columns=[
            col
            for col in (
                "moneyflow_confirmation_score",
                "moneyflow_confirmation_rank",
                "moneyflow_confirmation_tag",
                "moneyflow_confirmation_detail",
            )
            if col in out.columns
        ],
        errors="ignore",
    ).merge(latest, on="code", how="left")
    return out


def _annotate_premium_chip_context(selected: pd.DataFrame, selection_ts: pd.Timestamp) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out
    path = PREMIUM_DIR / "tushare_chip_daily.csv"
    usecols = [
        "date",
        "code",
        "cost_50pct",
        "cost_85pct",
        "cost_95pct",
        "weight_avg",
        "winner_rate",
    ]
    codes = set(out["code"].astype(str))
    frame = _load_recent_premium_rows(
        path,
        codes=codes,
        usecols=usecols,
        date_col="date",
        start_date=selection_ts - pd.Timedelta(days=45),
        end_date=selection_ts,
    )
    if frame.empty:
        return out
    latest = _latest_by_code(frame, "date")
    for col in ("cost_50pct", "cost_85pct", "cost_95pct", "weight_avg", "winner_rate"):
        latest[col] = pd.to_numeric(latest[col], errors="coerce")
    latest = latest.merge(
        out.loc[:, ["code", "close"]].drop_duplicates("code"),
        on="code",
        how="left",
    )
    latest["close"] = pd.to_numeric(latest["close"], errors="coerce")
    latest["winner_rate_frac"] = latest["winner_rate"].fillna(0.0) / 100.0
    latest["chip_upper_gap"] = ((latest["cost_95pct"] - latest["close"]) / latest["close"]).replace([pd.NA, pd.NaT], 0.0)
    latest["chip_avg_gap"] = ((latest["weight_avg"] - latest["close"]) / latest["close"]).replace([pd.NA, pd.NaT], 0.0)
    latest["chip_overhead_score"] = (
        latest["winner_rate_frac"].fillna(0.0)
        - 0.5 * latest["chip_upper_gap"].fillna(0.0).clip(lower=0.0, upper=1.0)
        - 0.5 * latest["chip_avg_gap"].fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    latest["chip_overhead_tag"] = "筹码压力偏大"
    latest.loc[
        (latest["winner_rate_frac"] >= 0.35) & (latest["chip_upper_gap"] <= 0.15),
        "chip_overhead_tag",
    ] = "筹码结构中性"
    latest.loc[
        (latest["winner_rate_frac"] >= 0.55) & (latest["chip_upper_gap"] <= 0.08),
        "chip_overhead_tag",
    ] = "筹码压力轻"
    latest["chip_overhead_detail"] = (
        "筹码分布("
        + latest["date"].dt.strftime("%Y-%m-%d")
        + ")：胜率 "
        + latest["winner_rate"].fillna(0.0).map(_fmt_pct)
        + "，平均成本 "
        + latest["weight_avg"].fillna(0.0).map(lambda x: f"{x:.2f}")
        + "，95%成本 "
        + latest["cost_95pct"].fillna(0.0).map(lambda x: f"{x:.2f}")
        + "，"
        + latest["chip_overhead_tag"].astype(str)
    )
    latest = latest.loc[
        :,
        [
            "code",
            "chip_overhead_score",
            "chip_overhead_tag",
            "chip_overhead_detail",
        ],
    ]
    out = out.merge(latest, on="code", how="left")
    return out


def _annotate_premium_fundamental_context(selected: pd.DataFrame, selection_ts: pd.Timestamp) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out
    path = PREMIUM_DIR / "tushare_fundamental_events.csv"
    usecols = [
        "code",
        "ann_date",
        "end_date",
        "profit_quality_factor",
        "roe_factor",
        "earnings_growth_factor",
        "cashflow_quality_factor",
        "debt_to_asset",
        "distress_risk_flag",
        "profit_warning_flag",
        "forecast_type",
        "p_change_min",
        "p_change_max",
    ]
    codes = set(out["code"].astype(str))
    frame = _load_recent_premium_rows(
        path,
        codes=codes,
        usecols=usecols,
        date_col="ann_date",
        start_date=selection_ts - pd.Timedelta(days=365),
        end_date=selection_ts,
    )
    if frame.empty:
        return out
    latest = _latest_by_code(frame, "ann_date")
    for col in (
        "profit_quality_factor",
        "roe_factor",
        "earnings_growth_factor",
        "cashflow_quality_factor",
        "debt_to_asset",
        "p_change_min",
        "p_change_max",
    ):
        latest[col] = pd.to_numeric(latest[col], errors="coerce")
    for col in ("distress_risk_flag", "profit_warning_flag"):
        latest[col] = latest[col].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    latest["fundamental_context_tag"] = "公告/预警中性"
    positive_mask = (
        latest["forecast_type"].fillna("").astype(str).str.strip().ne("")
        | (latest["earnings_growth_factor"].fillna(0.0) > 0)
        | (latest["profit_quality_factor"].fillna(0.0) > 0)
    )
    warn_mask = latest["distress_risk_flag"] | latest["profit_warning_flag"]
    latest.loc[positive_mask, "fundamental_context_tag"] = "公告/预警偏正面"
    latest.loc[warn_mask, "fundamental_context_tag"] = "公告/预警偏弱"
    latest["fundamental_context_detail"] = (
        "最新公告("
        + latest["ann_date"].dt.strftime("%Y-%m-%d")
        + ")：预测类型 "
        + latest["forecast_type"].fillna("无")
        + "，利润预警="
        + latest["profit_warning_flag"].map(lambda x: "是" if bool(x) else "否")
        + "，困境风险="
        + latest["distress_risk_flag"].map(lambda x: "是" if bool(x) else "否")
        + "，"
        + latest["fundamental_context_tag"].astype(str)
    )
    latest = latest.loc[
        :,
        [
            "code",
            "fundamental_context_tag",
            "fundamental_context_detail",
        ],
    ]
    out = out.merge(latest, on="code", how="left")
    return out


def _annotate_premium_observation_layers(selected: pd.DataFrame, selection_ts: pd.Timestamp) -> pd.DataFrame:
    out = selected.copy()
    out = _annotate_premium_flow_confirmation(out, selection_ts)
    out = _annotate_premium_chip_context(out, selection_ts)
    out = _annotate_premium_fundamental_context(out, selection_ts)
    return out


def annotate_observation_layers(
    selected: pd.DataFrame,
    *,
    research_cfg: dict[str, Any],
    scored_df: pd.DataFrame,
    selection_ts: pd.Timestamp,
    include_premium: bool = True,
    observation_cache: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    out = selected.copy()
    out = _annotate_ml_observation_tags(out, research_cfg)
    out = _annotate_industry_breadth_observation(out, research_cfg)
    out = _annotate_industry_leader_follow_observation(out, research_cfg)
    overhead_history = None if observation_cache is None else observation_cache.get("overhead_history")
    out = _annotate_overhead_density_observation(out, research_cfg, scored_df, selection_ts, overhead_history=overhead_history)
    out = _annotate_moneyflow_observation(out, research_cfg)
    if include_premium:
        out = _annotate_premium_observation_layers(out, selection_ts)
    return out


def build_observation_pool_history(
    scored_df: pd.DataFrame,
    *,
    research_cfg: dict[str, Any],
    observation_top_n: int | None = None,
    selection_dates: list[pd.Timestamp] | None = None,
    include_premium: bool = False,
) -> pd.DataFrame:
    top_n = int(observation_top_n or research_cfg.get("observation_top_n", research_cfg.get("top_n", 20)))
    eligible_col = "monitor_eligible" if "monitor_eligible" in scored_df.columns else "tradeable"
    eligible = scored_df.loc[
        scored_df[eligible_col].fillna(False).astype(bool) & scored_df["score"].notna()
    ].copy()
    if eligible.empty:
        return eligible.head(0).copy()
    if selection_dates:
        allowed_dates = {pd.Timestamp(value) for value in selection_dates}
        eligible = eligible.loc[eligible["date"].isin(allowed_dates)].copy()
    preselected: list[pd.DataFrame] = []
    for _, daily in eligible.groupby("date", sort=True):
        selected = daily.sort_values("score", ascending=False).head(top_n).copy()
        if selected.empty:
            continue
        selected["rank"] = range(1, len(selected) + 1)
        selected["target_weight"] = 1.0 / len(selected)
        preselected.append(selected)
    if not preselected:
        return eligible.head(0).copy()
    preselected_df = pd.concat(preselected, ignore_index=True)
    overhead_history = _build_overhead_density_history(
        scored_df,
        codes=set(preselected_df["code"].astype(str)),
        selection_dates=sorted(pd.to_datetime(preselected_df["date"]).drop_duplicates().tolist()),
    )
    grouped: list[pd.DataFrame] = []
    for selection_ts, selected in preselected_df.groupby("date", sort=True):
        selected = annotate_observation_layers(
            selected.copy(),
            research_cfg=research_cfg,
            scored_df=scored_df,
            selection_ts=pd.Timestamp(selection_ts),
            include_premium=include_premium,
            observation_cache={"overhead_history": overhead_history},
        )
        grouped.append(selected)
    return pd.concat(grouped, ignore_index=True)


def build_review_question_pack(item: dict[str, Any]) -> dict[str, str]:
    leader_tag = str(item.get("industry_leader_follow_tag") or "").strip() or "暂不可判定"
    pressure_tag = str(item.get("overhead_density_tag") or "").strip() or "暂不可判定"
    if leader_tag == "龙头扩散强" or pressure_tag == "兑现压力轻":
        selected_because = "主分高被选中，同时观察层提示它更值得重点关注。"
    else:
        selected_because = "主分高被选中；观察层当前主要负责解释，不改变排序。"
    label_combo = f"{leader_tag} + {pressure_tag}"
    failure_attribution = "暂不可判定（当前仍处在样本外路径运行中，需等后续兑现结果）。"
    return {
        "selected_because": selected_because,
        "label_combo": label_combo,
        "failure_attribution": failure_attribution,
    }


def build_latest_picks_table(
    data_path: str | Path,
    research_config_path: str | Path,
    data_source: MarketDataSource | None = None,
    data_adjust: str = "none",
    prediction_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float], LatestPickSummary]:
    research_cfg, scored_df, metadata, selection_ts, candidate_dates, train_df, score_details, _ = _prepare_latest_context(
        data_path=data_path,
        research_config_path=research_config_path,
        data_source=data_source,
        data_adjust=data_adjust,
        prediction_date=prediction_date,
    )
    factor_cols = metadata["horizon_feature_map"][metadata["label_horizons"][0]]
    horizon = metadata["label_horizons"][0]
    label_col = _label_col(research_cfg, horizon)
    train_df = train_df.loc[train_df[label_col].notna()].copy()
    if train_df.empty:
        raise ValueError("Not enough labeled history before the prediction date to fit the ranker.")

    latest_df = scored_df.loc[scored_df["date"] == selection_ts].copy()
    eligible_col = "monitor_eligible" if "monitor_eligible" in latest_df.columns else "tradeable"
    eligible = latest_df.loc[latest_df[eligible_col].fillna(False).astype(bool) & latest_df["score"].notna()].copy()
    if eligible.empty:
        raise ValueError(f"No tradeable securities with valid scores on {selection_ts.date()}.")

    observation_top_n = int(research_cfg.get("observation_top_n", research_cfg["top_n"]))
    selected = eligible.sort_values("score", ascending=False).head(observation_top_n).copy()
    selected["rank"] = range(1, len(selected) + 1)
    selected["target_weight"] = 1.0 / len(selected)
    selected = annotate_observation_layers(
        selected,
        research_cfg=research_cfg,
        scored_df=scored_df,
        selection_ts=selection_ts,
        include_premium=True,
        observation_cache=None,
    )

    keep_cols = [
        "date",
        "rank",
        "code",
        "name",
        "industry",
        "close",
        "market_cap",
        "amount",
        "score",
        "target_weight",
        "ml_score",
        "ml_rank",
        "ml_observation_tag",
        "industry_breadth_score",
        "industry_breadth_rank",
        "industry_breadth_tag",
        "industry_breadth_detail",
        "industry_leader_follow_score",
        "industry_leader_follow_rank",
        "industry_leader_follow_tag",
        "industry_leader_follow_detail",
        "overhead_density_score",
        "overhead_density_rank",
        "overhead_density_tag",
        "overhead_density_detail",
        "moneyflow_confirmation_score",
        "moneyflow_confirmation_rank",
        "moneyflow_confirmation_tag",
        "moneyflow_confirmation_detail",
        "chip_overhead_score",
        "chip_overhead_tag",
        "chip_overhead_detail",
        "fundamental_context_tag",
        "fundamental_context_detail",
        *factor_cols,
    ]
    available_cols = [col for col in keep_cols if col in selected.columns]
    picks = selected.loc[:, available_cols].reset_index(drop=True)

    summary = LatestPickSummary(
        layer="monitor / observation pool",
        selection_date=selection_ts.strftime("%Y-%m-%d"),
        train_start_date=train_df["date"].min().strftime("%Y-%m-%d"),
        train_end_date=train_df["date"].max().strftime("%Y-%m-%d"),
        train_rows=len(train_df),
        train_dates=train_df["date"].nunique(),
        candidate_count=len(eligible),
        selected_count=len(picks),
        top_n=observation_top_n,
        label_horizon=horizon,
        label_type=str(research_cfg["label_type"]),
        adjust_mode=data_adjust,
        data_path=str(data_path),
        signal_time=str(research_cfg["signal_time"]),
        execution_time=f"t+{int(research_cfg['execution_lag'])} {research_cfg['execution_price']}",
        system_mode=str(research_cfg.get("system_mode", "stable_observation_cycle")),
        main_score_frozen=bool(research_cfg.get("main_score_frozen", False)),
        execution_layer_frozen=bool(research_cfg.get("execution_layer_frozen", False)),
        observation_layer_frozen=bool(research_cfg.get("observation_layer_frozen", False)),
        new_research_gate_passed=bool(research_cfg.get("new_research_gate_passed", False)),
        new_research_gate_reason=str(research_cfg.get("new_research_gate_reason", "")),
    )
    weight_payload = extract_display_weights(score_details)
    return picks, weight_payload, summary


def build_latest_strategy_table(
    data_path: str | Path,
    research_config_path: str | Path,
    data_source: MarketDataSource | None = None,
    data_adjust: str = "none",
    prediction_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float], LatestPickSummary]:
    research_cfg, scored_df, metadata, selection_ts, candidate_dates, train_df, score_details, _ = _prepare_latest_context(
        data_path=data_path,
        research_config_path=research_config_path,
        data_source=data_source,
        data_adjust=data_adjust,
        prediction_date=prediction_date,
    )

    horizon = metadata["label_horizons"][0]
    label_col = _label_col(research_cfg, horizon)
    train_df = train_df.loc[train_df[label_col].notna()].copy()
    if train_df.empty:
        raise ValueError("Not enough labeled history before the prediction date to fit the ranker.")

    latest_df = scored_df.loc[scored_df["date"] == selection_ts].copy()
    eligible_col = "strategy_tradeable" if "strategy_tradeable" in latest_df.columns else "tradeable"
    eligible = latest_df.loc[latest_df[eligible_col].fillna(False).astype(bool) & latest_df["score"].notna()].copy()
    if eligible.empty:
        raise ValueError(f"No deployable strategy candidates with valid scores on {selection_ts.date()}.")

    risk_state = str(eligible["risk_state"].iloc[0]) if "risk_state" in eligible.columns else "full_risk"
    risk_multiplier = float(pd.to_numeric(eligible["risk_multiplier"], errors="coerce").iloc[0]) if "risk_multiplier" in eligible.columns else 1.0
    if risk_state == "low_risk" or risk_multiplier <= 1e-12:
        selected = eligible.head(0).copy()
    else:
        selected = optimize_portfolio_weights(
            eligible,
            cfg=PortfolioConfig(
                top_n=int(research_cfg["top_n"]),
                weighting_method=str(research_cfg["weighting_method"]),
                score_threshold=float(research_cfg["score_threshold"]),
                softmax_temperature=float(research_cfg["softmax_temperature"]),
                max_weight=float(research_cfg["max_weight"]),
                industry_cap=float(research_cfg["industry_cap"]),
                min_holdings=int(research_cfg["min_holdings"]),
                weight_change_threshold=float(research_cfg["weight_change_threshold"]),
                rank_change_threshold=int(research_cfg["rank_change_threshold"]),
                entry_score_advantage_threshold=float(research_cfg["entry_score_advantage_threshold"]),
            ),
            score_col="score",
        )
        selected["target_weight"] = pd.to_numeric(selected["target_weight"], errors="coerce").fillna(0.0) * risk_multiplier
    execution_date = _next_execution_date(
        selection_ts=selection_ts,
        candidate_dates=candidate_dates,
        lag=int(research_cfg["execution_lag"]),
    )
    selected["signal_date"] = selection_ts
    selected["execution_date"] = execution_date
    selected["sleeve"] = 0
    selected["trade_reason"] = selected.get("trade_reason", pd.Series("new_entry_stronger", index=selected.index))
    selected["risk_state"] = risk_state
    selected["risk_multiplier"] = risk_multiplier
    selected = annotate_observation_layers(
        selected,
        research_cfg=research_cfg,
        scored_df=scored_df,
        selection_ts=selection_ts,
        include_premium=True,
        observation_cache=None,
    )

    keep_cols = [
        "date",
        "signal_date",
        "execution_date",
        "sleeve",
        "rank",
        "code",
        "name",
        "industry",
        "close",
        "market_cap",
        "amount",
        "score",
        "target_weight",
        "ml_score",
        "ml_rank",
        "ml_observation_tag",
        "industry_breadth_score",
        "industry_breadth_rank",
        "industry_breadth_tag",
        "industry_breadth_detail",
        "industry_leader_follow_score",
        "industry_leader_follow_rank",
        "industry_leader_follow_tag",
        "industry_leader_follow_detail",
        "overhead_density_score",
        "overhead_density_rank",
        "overhead_density_tag",
        "overhead_density_detail",
        "moneyflow_confirmation_score",
        "moneyflow_confirmation_rank",
        "moneyflow_confirmation_tag",
        "moneyflow_confirmation_detail",
        "chip_overhead_score",
        "chip_overhead_tag",
        "chip_overhead_detail",
        "fundamental_context_tag",
        "fundamental_context_detail",
        "trade_reason",
        "risk_state",
        "risk_multiplier",
    ]
    strategy = selected.loc[:, [col for col in keep_cols if col in selected.columns]].reset_index(drop=True)
    summary = LatestPickSummary(
        layer="tradable strategy / deployable prototype",
        selection_date=selection_ts.strftime("%Y-%m-%d"),
        train_start_date=train_df["date"].min().strftime("%Y-%m-%d"),
        train_end_date=train_df["date"].max().strftime("%Y-%m-%d"),
        train_rows=len(train_df),
        train_dates=train_df["date"].nunique(),
        candidate_count=len(eligible),
        selected_count=len(strategy),
        top_n=int(research_cfg["top_n"]),
        label_horizon=horizon,
        label_type=str(research_cfg["label_type"]),
        adjust_mode=data_adjust,
        data_path=str(data_path),
        signal_time=str(research_cfg["signal_time"]),
        execution_time=f"t+{int(research_cfg['execution_lag'])} {research_cfg['execution_price']}",
        system_mode=str(research_cfg.get("system_mode", "stable_observation_cycle")),
        main_score_frozen=bool(research_cfg.get("main_score_frozen", False)),
        execution_layer_frozen=bool(research_cfg.get("execution_layer_frozen", False)),
        observation_layer_frozen=bool(research_cfg.get("observation_layer_frozen", False)),
        new_research_gate_passed=bool(research_cfg.get("new_research_gate_passed", False)),
        new_research_gate_reason=str(research_cfg.get("new_research_gate_reason", "")),
    )
    weight_payload = extract_display_weights(score_details)
    return strategy, weight_payload, summary


def build_latest_picks_markdown(
    picks: pd.DataFrame,
    factor_weights: dict[str, float],
    summary: LatestPickSummary,
) -> str:
    ranked_weights = sorted(factor_weights.items(), key=lambda item: abs(item[1]), reverse=True)
    weight_lines = "\n".join(
        f"- `{name}`: {value:.4f}" for name, value in ranked_weights[: min(5, len(ranked_weights))]
    )

    if picks.empty:
        table = "_No picks generated._"
        observation_block = "- observation layer not available"
    else:
        table_header = "| rank | code | name | weight | score | close | 龙头扩散 | 兑现压力 | industry |"
        table_sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        table_rows = []
        observation_lines = []
        review_lines = []
        for row in picks.itertuples(index=False):
            review_pack = build_review_question_pack(row._asdict())
            table_rows.append(
                "| {rank} | {code} | {name} | {weight:.3f} | {score:.4f} | {close:.2f} | {leader_tag} | {pressure_tag} | {industry} |".format(
                    rank=getattr(row, "rank", ""),
                    code=getattr(row, "code", ""),
                    name=getattr(row, "name", "") or "",
                    weight=float(getattr(row, "target_weight", 0.0)),
                    score=float(getattr(row, "score", 0.0)),
                    close=float(getattr(row, "close", 0.0)),
                    leader_tag=getattr(row, "industry_leader_follow_tag", "") or "无",
                    pressure_tag=getattr(row, "overhead_density_tag", "") or "无",
                    industry=getattr(row, "industry", "") or "Unknown",
                )
            )
            observation_lines.append(
                "- {code} {name}: 龙头扩散 {leader_score:.4f}/#{leader_rank} {leader_tag}；兑现压力 {pressure_score:.4f}/#{pressure_rank} {pressure_tag}。".format(
                    code=getattr(row, "code", ""),
                    name=getattr(row, "name", "") or "",
                    leader_score=float(getattr(row, "industry_leader_follow_score", 0.0) or 0.0),
                    leader_rank=int(getattr(row, "industry_leader_follow_rank", 0) or 0),
                    leader_tag=getattr(row, "industry_leader_follow_tag", "") or "无",
                    pressure_score=float(getattr(row, "overhead_density_score", 0.0) or 0.0),
                    pressure_rank=int(getattr(row, "overhead_density_rank", 0) or 0),
                    pressure_tag=getattr(row, "overhead_density_tag", "") or "无",
                )
            )
            review_lines.append(
                "- {code} {name}: 1) {selected_because} 2) 标签组合={label_combo} 3) 失败归因={failure_attribution}".format(
                    code=getattr(row, "code", ""),
                    name=getattr(row, "name", "") or "",
                    selected_because=review_pack["selected_because"],
                    label_combo=review_pack["label_combo"],
                    failure_attribution=review_pack["failure_attribution"],
                )
            )
        table = "\n".join([table_header, table_sep, *table_rows])
        observation_block = "\n".join(observation_lines[:10]) if observation_lines else "- observation layer not available"
        review_block = "\n".join(review_lines[:10]) if review_lines else "- 暂不可判定"

    lines = [
        "# Latest A-share Picks",
        "",
        "## Selection Summary",
        "",
        f"- layer: `{summary.layer}`",
        f"- selection date: `{summary.selection_date}`",
        f"- training window: `{summary.train_start_date}` to `{summary.train_end_date}`",
        f"- training rows: `{summary.train_rows}`",
        f"- train dates: `{summary.train_dates}`",
        f"- candidate count: `{summary.candidate_count}`",
        f"- selected count: `{summary.selected_count}`",
        f"- top_n target: `{summary.top_n}`",
        f"- label horizon: `{summary.label_horizon}` trading days",
        f"- label type: `{summary.label_type}`",
        f"- price adjustment: `{summary.adjust_mode}`",
        f"- signal time: `{summary.signal_time}`",
        f"- execution time: `{summary.execution_time}`",
        f"- data path: `{summary.data_path}`",
        "",
        "## Stable Observation Cycle",
        "",
        f"- system_mode: `{summary.system_mode}`",
        f"- main_score_frozen: `{summary.main_score_frozen}`",
        f"- execution_layer_frozen: `{summary.execution_layer_frozen}`",
        f"- observation_layer_frozen: `{summary.observation_layer_frozen}`",
        "- freeze_message: `当前只允许 bugfix、监控、质量看板和报告结构增强，不进行新因子升级，不进行执行规则升级。`",
        f"- new_research_gate_passed: `{summary.new_research_gate_passed}`",
        f"- why_not: `{summary.new_research_gate_reason}`",
        "",
        "## Factor Weights",
        "",
        weight_lines or "- no factor weights",
        "",
        "## Picks",
        "",
        table,
        "",
        "## Observation Layer",
        "",
        observation_block,
        "",
        "## 复盘三问",
        "",
        review_block if not picks.empty else "- 暂不可判定",
        "",
        "## Notes",
        "",
        "- This table is an observation pool generated from the latest available date in the dataset.",
        "- It is not a live-trading instruction set; use it together with the deployable strategy layer, risk checks, turnover limits, and data validation.",
        "- If `industry` is `Unknown`, the current neutralization step is weaker than it should be.",
    ]
    return "\n".join(lines) + "\n"


def write_latest_picks_markdown(
    output_path: str | Path,
    picks: pd.DataFrame,
    factor_weights: dict[str, float],
    summary: LatestPickSummary,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_latest_picks_markdown(
            picks=picks,
            factor_weights=factor_weights,
            summary=summary,
        ),
        encoding="utf-8",
    )
    return output_path


def write_latest_picks_json(
    output_path: str | Path,
    picks: pd.DataFrame,
    factor_weights: dict[str, float],
    summary: LatestPickSummary,
) -> Path:
    pick_records = json.loads(picks.to_json(orient="records", date_format="iso", force_ascii=False))
    payload = {
        "summary": asdict(summary),
        "factor_weights": factor_weights,
        "picks": pick_records,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
