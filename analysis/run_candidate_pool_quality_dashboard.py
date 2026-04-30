from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.candidate_pool_quality import (
    add_candidate_path_columns,
    build_candidate_pool_quality_timeseries,
    build_stable_observation_cycle_verdict,
    summarize_failure_attribution,
    summarize_candidate_pool_quality,
    summarize_observation_tag_paths,
)
from ashare_quant.pipeline import DEFAULT_RESEARCH, load_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a lightweight candidate-pool quality dashboard.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--research-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--rolling-window", type=int, default=60)
    parser.add_argument(
        "--max-selection-dates",
        type=int,
        default=180,
        help="Maximum number of historical selection dates to read from existing daily monitor outputs.",
    )
    return parser


def _normalize_pick_frame(rows: list[dict], *, fallback_date: pd.Timestamp) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return frame
    if "rank" not in frame.columns:
        return pd.DataFrame()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    elif "signal_date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    else:
        frame["date"] = fallback_date
    frame["date"] = frame["date"].fillna(fallback_date)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame = frame.loc[frame["rank"].notna()].copy()
    if frame.empty:
        return frame
    frame["rank"] = frame["rank"].astype(int)
    frame["score"] = pd.to_numeric(frame.get("score"), errors="coerce")
    frame["strategy_tradeable"] = True
    return frame


def _load_strategy_snapshot_history(
    *,
    output_root: Path,
    max_selection_dates: int,
    max_top_n: int,
    observation_top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(output_root.glob("daily_monitor_*_picks.json"))
    dated_payloads: list[tuple[pd.Timestamp, dict]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = payload.get("summary") or {}
        selection_date = summary.get("selection_date")
        if not selection_date:
            continue
        try:
            selection_ts = pd.Timestamp(selection_date)
        except Exception:
            continue
        dated_payloads.append((selection_ts, payload))
    if not dated_payloads:
        return pd.DataFrame(), pd.DataFrame()

    dated_payloads = sorted(dated_payloads, key=lambda item: item[0])[-max_selection_dates:]
    strategy_rows: list[pd.DataFrame] = []
    observation_rows: list[pd.DataFrame] = []
    for selection_ts, payload in dated_payloads:
        strategy_snapshot = payload.get("strategy_snapshot") or payload.get("picks") or []
        strategy_frame = _normalize_pick_frame(strategy_snapshot, fallback_date=selection_ts)
        if strategy_frame.empty:
            continue
        strategy_rows.append(strategy_frame.loc[strategy_frame["rank"] <= max_top_n].copy())
        observation_rows.append(strategy_frame.loc[strategy_frame["rank"] <= observation_top_n].copy())

    strategy_history = pd.concat(strategy_rows, ignore_index=True) if strategy_rows else pd.DataFrame()
    observation_history = pd.concat(observation_rows, ignore_index=True) if observation_rows else pd.DataFrame()
    return strategy_history, observation_history


def _load_price_window(
    *,
    data_path: str | Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    adjust: str,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    usecols = ["date", "code", "close", "high", "low"]
    if adjust != "none":
        usecols.append("adj_factor")
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(data_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        frame = chunk.loc[chunk["date"].astype(str).str.lower() != "date"].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.loc[frame["date"].notna()]
        if frame.empty:
            continue
        frame = frame.loc[(frame["date"] >= start_date) & (frame["date"] <= end_date)].copy()
        if frame.empty:
            continue
        for col in ("close", "high", "low", "adj_factor"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        pieces.append(frame)
    if not pieces:
        return pd.DataFrame(columns=usecols)
    price_df = pd.concat(pieces, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
    if adjust != "none" and "adj_factor" in price_df.columns:
        latest_factor = price_df.groupby("code")["adj_factor"].transform("last").replace(0.0, pd.NA)
        if adjust == "qfq":
            scale = (price_df["adj_factor"] / latest_factor).fillna(1.0)
        else:
            scale = price_df["adj_factor"].fillna(1.0)
        for col in ("close", "high", "low"):
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce") * scale
    return price_df


def _attach_realized_paths(
    strategy_history: pd.DataFrame,
    observation_history: pd.DataFrame,
    *,
    data_path: str | Path,
    adjust: str,
    holding_horizon: int,
    quality_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    if strategy_history.empty:
        return strategy_history, observation_history, []
    selection_dates = sorted(pd.to_datetime(strategy_history["date"]).dropna().unique().tolist())
    start_date = pd.Timestamp(selection_dates[0])
    max_horizon = max(holding_horizon, quality_horizon, 10)
    end_date = pd.Timestamp(selection_dates[-1]) + pd.offsets.BDay(max_horizon + 1)
    price_df = _load_price_window(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    if price_df.empty:
        return strategy_history.iloc[0:0].copy(), observation_history.iloc[0:0].copy(), []
    universe_paths = add_candidate_path_columns(price_df, horizons={holding_horizon, quality_horizon, 5, 10})
    future_cols = sorted(
        {
            f"path_return_{holding_horizon}d",
            f"path_max_up_{holding_horizon}d",
            f"path_max_drawdown_{holding_horizon}d",
            f"path_return_{quality_horizon}d",
            f"path_max_up_{quality_horizon}d",
            f"path_max_drawdown_{quality_horizon}d",
            "path_return_5d",
            "path_return_10d",
            "path_max_up_5d",
            "path_max_drawdown_5d",
        }
    )
    available_cols = ["date", "code", *[col for col in future_cols if col in universe_paths.columns]]
    median_col = f"path_return_{quality_horizon}d"
    daily_median = (
        universe_paths.groupby("date")[median_col].median().rename("daily_median_future_return")
        if median_col in universe_paths.columns
        else pd.Series(dtype=float)
    )

    def _merge_paths(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        merged = frame.merge(universe_paths.loc[:, available_cols], on=["date", "code"], how="left")
        merged["daily_median_future_return"] = pd.to_datetime(merged["date"]).map(daily_median)
        return merged

    strategy_with_paths = _merge_paths(strategy_history)
    observation_with_paths = _merge_paths(observation_history)
    realized_dates = sorted(pd.to_datetime(strategy_with_paths["date"]).dropna().unique().tolist())
    return strategy_with_paths, observation_with_paths, realized_dates


def build_markdown(
    *,
    summary_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    tag_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    payload: dict,
) -> str:
    def _frame_to_markdown(frame: pd.DataFrame, *, empty_text: str) -> str:
        if frame.empty:
            return empty_text
        try:
            return frame.to_markdown(index=False)
        except Exception:
            return frame.to_string(index=False)

    quality_horizon = int(payload.get("quality_horizon") or payload.get("holding_horizon") or 5)
    lines = [
        "# Candidate Pool Quality Dashboard",
        "",
        f"- research config: `{payload['research_config_path']}`",
        f"- data path: `{payload['data_path']}`",
        f"- evaluation dates: `{payload['evaluation_start_date']} ~ {payload['evaluation_end_date']}`",
        f"- holding horizon: `{payload['holding_horizon']}d`",
        f"- candidate quality horizon: `{quality_horizon}d`",
        f"- system_mode: `{payload['system_mode']}`",
        f"- keep_frozen: `{payload['keep_frozen']}`",
        f"- keep_frozen_reason: `{payload['keep_frozen_reason']}`",
        "",
        "## Candidate Pool Summary",
        "",
        payload.get("headline", "暂无候选池结论。"),
        "",
        _frame_to_markdown(summary_df, empty_text="暂无候选池质量数据。"),
        "",
        "## Failure Attribution",
        "",
        _frame_to_markdown(failure_df, empty_text="暂无失败归因数据。"),
        "",
        "## Observation Tag Paths",
        "",
        _frame_to_markdown(tag_df, empty_text="暂无观察层路径差异数据。"),
        "",
    ]
    trend_df = pd.DataFrame(payload.get("trend_rows") or [])
    if not trend_df.empty:
        lines.extend(
            [
                "## Rolling Quality Trend",
                "",
                _frame_to_markdown(trend_df, empty_text="暂无滚动质量趋势。"),
                "",
            ]
        )
    if not rolling_df.empty:
        latest = (
            rolling_df.sort_values(["pool_name", "date"])
            .groupby("pool_name", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        lines.extend(
            [
                "## Latest Rolling Snapshot",
                "",
                _frame_to_markdown(latest, empty_text="暂无滚动快照。"),
                "",
            ]
        )
    lines.extend(
        [
            "## 复盘三问",
            "",
            "1. 这批票是因为主分高被选中，还是因为观察层更值得关注？当前答案：主分负责选中，观察层负责解释与复盘。",
            "2. 它们更偏“扩散强 + 压力低”，还是“扩散弱 + 压力高”？请优先看 Observation Tag Paths 和 Latest Rolling Snapshot。",
            "3. 最近没做好，更像排序问题还是兑现路径问题？请优先看 Failure Attribution。",
            "",
            "## 新增研究门槛",
            "",
            f"- new_research_gate_passed: `{payload['new_research_gate_passed']}`",
            f"- why_not: `{payload['new_research_gate_reason']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    research_cfg = {**DEFAULT_RESEARCH, **load_json(args.research_config)}
    holding_horizon = int(research_cfg.get("holding_period", research_cfg.get("holding_window", 5)))
    quality_horizon = int(research_cfg.get("candidate_pool_quality_horizon", min(holding_horizon, 5)))
    top_ns = research_cfg.get("candidate_pool_top_ns", [20, 10, 5])
    observation_top_n = int(research_cfg.get("observation_top_n", 20))
    max_top_n = max(int(v) for v in top_ns if int(v) > 0)

    strategy_history, observation_history = _load_strategy_snapshot_history(
        output_root=ROOT / "outputs" / "daily_monitor_auto",
        max_selection_dates=int(args.max_selection_dates),
        max_top_n=max_top_n,
        observation_top_n=observation_top_n,
    )
    strategy_history, observation_history, realized_dates = _attach_realized_paths(
        strategy_history,
        observation_history,
        data_path=args.data_path,
        adjust=args.adjust,
        holding_horizon=holding_horizon,
        quality_horizon=quality_horizon,
    )

    summary_df = summarize_candidate_pool_quality(
        strategy_history,
        score_col="score",
        eligibility_col="strategy_tradeable",
        top_ns=top_ns,
        holding_horizon=quality_horizon,
    )
    rolling_df = build_candidate_pool_quality_timeseries(
        strategy_history,
        score_col="score",
        eligibility_col="strategy_tradeable",
        top_ns=top_ns,
        holding_horizon=quality_horizon,
        rolling_window=int(args.rolling_window),
    )
    tag_df = summarize_observation_tag_paths(observation_history)
    failure_df = summarize_failure_attribution(
        strategy_history,
        score_col="score",
        eligibility_col="strategy_tradeable",
        top_ns=top_ns,
        evaluation_horizon=quality_horizon,
    )
    verdict = build_stable_observation_cycle_verdict(summary_df, rolling_df, tag_df)

    headline = (
        "暂无候选池质量结论。"
        if summary_df.empty
        else "当前最值得盯的是 top{top_n}：平均收益 {avg_return:.4f}，命中率 {hit_rate:.2%}，胜率 {win_rate:.2%}，平均回撤 {avg_drawdown:.4f}。".format(
            **summary_df.sort_values(["avg_return", "win_rate"], ascending=False).iloc[0].to_dict()
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "candidate_pool_quality_summary.csv"
    rolling_path = output_dir / "candidate_pool_quality_rolling.csv"
    tag_path = output_dir / "candidate_pool_tag_paths.csv"
    failure_path = output_dir / "candidate_pool_failure_attribution.csv"
    md_path = output_dir / "candidate_pool_quality_dashboard.md"
    json_path = output_dir / "candidate_pool_quality_dashboard.json"

    summary_df.to_csv(summary_path, index=False)
    rolling_df.to_csv(rolling_path, index=False)
    tag_df.to_csv(tag_path, index=False)
    failure_df.to_csv(failure_path, index=False)
    payload = {
        "data_path": str(args.data_path),
        "research_config_path": str(args.research_config),
        "evaluation_start_date": pd.Timestamp(realized_dates[0]).strftime("%Y-%m-%d") if realized_dates else "",
        "evaluation_end_date": pd.Timestamp(realized_dates[-1]).strftime("%Y-%m-%d") if realized_dates else "",
        "holding_horizon": holding_horizon,
        "quality_horizon": quality_horizon,
        "system_mode": str(research_cfg.get("system_mode", "stable_observation_cycle")),
        "headline": headline,
        "keep_frozen": bool(verdict.get("keep_frozen", True)),
        "keep_frozen_reason": str(verdict.get("why", "")),
        "trend_rows": verdict.get("trend_rows", []),
        "new_research_gate_passed": bool(research_cfg.get("new_research_gate_passed", False)),
        "new_research_gate_reason": str(
            research_cfg.get("new_research_gate_reason")
            or "does not improve candidate pool quality or action clarity"
        ),
        "history_source": "existing_daily_monitor_picks_plus_lightweight_price_paths",
        "summary": summary_df.to_dict(orient="records"),
        "failure_attribution": failure_df.to_dict(orient="records"),
        "tag_paths": tag_df.to_dict(orient="records"),
    }
    md_path.write_text(
        build_markdown(
            summary_df=summary_df,
            rolling_df=rolling_df,
            tag_df=tag_df,
            failure_df=failure_df,
            payload=payload,
        ),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"summary_csv={summary_path}")
    print(f"rolling_csv={rolling_path}")
    print(f"tag_csv={tag_path}")
    print(f"failure_csv={failure_path}")
    print(f"markdown={md_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
