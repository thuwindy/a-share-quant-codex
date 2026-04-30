from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.analysis.latest_picks import build_observation_pool_history
from ashare_quant.pipeline import DEFAULT_RESEARCH, assign_role_assignment, prepare_research_frame, score_research_frame

HOLDING_HORIZONS = [3, 5, 10, 15, 20]
POLICY_DECISION_THRESHOLDS = {
    "candidate_min_samples": 100,
    "candidate_min_years": 2,
    "candidate_min_return_lift": 0.005,
    "candidate_min_win_rate_lift": 0.01,
    "candidate_max_drawdown_slack": 0.01,
}
BASELINE_POLICY = "fixed_5d"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_research_config(path: str | Path) -> dict[str, Any]:
    cfg = {**DEFAULT_RESEARCH, **load_json(path)}
    cfg["observe_ml_score"] = False
    cfg["observation_top_n"] = int(cfg.get("observation_top_n", 20))
    return cfg


def add_holding_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["code", "date"]).copy()
    close_col = "research_close" if "research_close" in out.columns else "close"
    low_col = "research_low" if "research_low" in out.columns else "low"
    base_close = pd.to_numeric(out[close_col], errors="coerce").replace(0.0, np.nan)
    g = out.groupby("code", group_keys=False)
    for horizon in HOLDING_HORIZONS:
        future_close = pd.to_numeric(g[close_col].shift(-horizon), errors="coerce")
        out[f"holding_return_{horizon}d"] = future_close / base_close - 1.0
        future_lows = [pd.to_numeric(g[low_col].shift(-step), errors="coerce") for step in range(1, horizon + 1)]
        out[f"holding_drawdown_{horizon}d"] = pd.concat(future_lows, axis=1).min(axis=1) / base_close - 1.0
    return out


def define_policy_horizon(frame: pd.DataFrame, policy_name: str) -> pd.Series:
    high_score = frame["rank"] <= 10
    spread_strong = frame["industry_leader_follow_tag"].eq("龙头扩散强")
    pressure_low = frame["overhead_density_tag"].eq("兑现压力轻")
    pressure_high = frame["overhead_density_tag"].eq("兑现压力大")

    if policy_name == "fixed_5d":
        return pd.Series(5, index=frame.index)
    if policy_name == "fixed_10d":
        return pd.Series(10, index=frame.index)
    if policy_name == "spread_strong_extend_10d":
        return pd.Series(np.where(high_score & spread_strong, 10, 5), index=frame.index)
    if policy_name == "pressure_high_cut_3d":
        return pd.Series(np.where(high_score & pressure_high, 3, 5), index=frame.index)
    if policy_name == "spread_strong_pressure_low_extend_10d":
        return pd.Series(np.where(high_score & spread_strong & pressure_low, 10, 5), index=frame.index)
    if policy_name == "mixed_tag_path":
        return pd.Series(
            np.where(high_score & spread_strong & pressure_low, 10, np.where(pressure_high, 3, 5)),
            index=frame.index,
        )
    raise ValueError(f"Unsupported policy_name: {policy_name}")


def _policy_specs() -> list[tuple[str, str]]:
    return [
        ("fixed_5d", "固定持有5天"),
        ("fixed_10d", "固定持有10天"),
        ("spread_strong_extend_10d", "扩散强延长到10天"),
        ("pressure_high_cut_3d", "压力大提前到3天"),
        ("spread_strong_pressure_low_extend_10d", "扩散强+压力低延长到10天"),
        ("mixed_tag_path", "扩散强+压力低延长，压力大提前"),
    ]


def summarize_policy(frame: pd.DataFrame, *, policy_name: str, policy_label: str) -> dict[str, Any]:
    subset = frame.copy()
    subset["policy_holding_days"] = define_policy_horizon(subset, policy_name)
    subset["policy_return"] = np.nan
    subset["policy_drawdown"] = np.nan
    for horizon in HOLDING_HORIZONS:
        mask = subset["policy_holding_days"] == horizon
        subset.loc[mask, "policy_return"] = pd.to_numeric(subset.loc[mask, f"holding_return_{horizon}d"], errors="coerce")
        subset.loc[mask, "policy_drawdown"] = pd.to_numeric(subset.loc[mask, f"holding_drawdown_{horizon}d"], errors="coerce")
    subset = subset.loc[subset["policy_return"].notna()].copy()
    metrics: dict[str, Any] = {
        "policy_name": policy_name,
        "policy_label": policy_label,
        "sample_count": int(len(subset)),
    }
    if subset.empty:
        metrics.update(
            {
                "avg_holding_days": np.nan,
                "avg_return": np.nan,
                "win_rate": np.nan,
                "avg_drawdown": np.nan,
            }
        )
        return metrics
    metrics["avg_holding_days"] = float(pd.to_numeric(subset["policy_holding_days"], errors="coerce").mean())
    metrics["avg_return"] = float(pd.to_numeric(subset["policy_return"], errors="coerce").mean())
    metrics["win_rate"] = float((pd.to_numeric(subset["policy_return"], errors="coerce") > 0).mean())
    metrics["avg_drawdown"] = float(pd.to_numeric(subset["policy_drawdown"], errors="coerce").mean())
    return metrics


def summarize_policy_yearly(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = frame.copy()
    base["year"] = pd.to_datetime(base["date"]).dt.year.astype(int)
    for policy_name, policy_label in _policy_specs():
        policy_frame = base.copy()
        policy_frame["policy_holding_days"] = define_policy_horizon(policy_frame, policy_name)
        policy_frame["policy_return"] = np.nan
        policy_frame["policy_drawdown"] = np.nan
        for horizon in HOLDING_HORIZONS:
            mask = policy_frame["policy_holding_days"] == horizon
            policy_frame.loc[mask, "policy_return"] = pd.to_numeric(policy_frame.loc[mask, f"holding_return_{horizon}d"], errors="coerce")
            policy_frame.loc[mask, "policy_drawdown"] = pd.to_numeric(policy_frame.loc[mask, f"holding_drawdown_{horizon}d"], errors="coerce")
        policy_frame = policy_frame.loc[policy_frame["policy_return"].notna()].copy()
        if policy_frame.empty:
            continue
        for year, yearly in policy_frame.groupby("year", sort=True):
            rows.append(
                {
                    "year": int(year),
                    "policy_name": policy_name,
                    "policy_label": policy_label,
                    "sample_count": int(len(yearly)),
                    "avg_holding_days": float(pd.to_numeric(yearly["policy_holding_days"], errors="coerce").mean()),
                    "avg_return": float(pd.to_numeric(yearly["policy_return"], errors="coerce").mean()),
                    "win_rate": float((pd.to_numeric(yearly["policy_return"], errors="coerce") > 0).mean()),
                    "avg_drawdown": float(pd.to_numeric(yearly["policy_drawdown"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def assign_policy_decisions(summary_df: pd.DataFrame, yearly_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    out["policy_status"] = "observe_only"
    out["policy_comment"] = ""
    out["role_assignment"] = "observation_label"
    baseline_row = out.loc[out["policy_name"] == BASELINE_POLICY]
    if baseline_row.empty:
        out["policy_comment"] = "缺少固定5天基线，无法判断持有管理实验。"
        return out
    baseline = baseline_row.iloc[0]
    baseline_yearly = yearly_df.loc[yearly_df["policy_name"] == BASELINE_POLICY].copy()
    for idx, row in out.iterrows():
        if row["policy_name"] == BASELINE_POLICY:
            out.at[idx, "policy_comment"] = "固定持有5天的基线，用来判断标签是否改变更优持有路径。"
            continue
        if int(row["sample_count"]) < int(POLICY_DECISION_THRESHOLDS["candidate_min_samples"]):
            out.at[idx, "policy_comment"] = "样本太少，先保留研究结论，不讨论执行。"
            continue
        overall_pass = (
            float(row["avg_return"]) >= float(baseline["avg_return"]) + float(POLICY_DECISION_THRESHOLDS["candidate_min_return_lift"])
            and float(row["win_rate"]) >= float(baseline["win_rate"]) + float(POLICY_DECISION_THRESHOLDS["candidate_min_win_rate_lift"])
            and float(row["avg_drawdown"]) >= float(baseline["avg_drawdown"]) - float(POLICY_DECISION_THRESHOLDS["candidate_max_drawdown_slack"])
        )
        policy_yearly = yearly_df.loc[yearly_df["policy_name"] == row["policy_name"]].copy()
        merged_yearly = policy_yearly.merge(baseline_yearly, on="year", suffixes=("_policy", "_baseline"))
        stable_mask = (
            (merged_yearly["avg_return_policy"] > merged_yearly["avg_return_baseline"])
            & (merged_yearly["win_rate_policy"] >= merged_yearly["win_rate_baseline"])
            & (
                merged_yearly["avg_drawdown_policy"]
                >= merged_yearly["avg_drawdown_baseline"] - float(POLICY_DECISION_THRESHOLDS["candidate_max_drawdown_slack"])
            )
        )
        stable_years = int(stable_mask.sum()) if not merged_yearly.empty else 0
        if overall_pass and stable_years >= int(POLICY_DECISION_THRESHOLDS["candidate_min_years"]):
            out.at[idx, "policy_status"] = "management_candidate"
            out.at[idx, "policy_comment"] = "跨时间稳定优于固定5天基线，才值得进入下一步执行管理研究。"
        else:
            out.at[idx, "policy_comment"] = "还没有形成足够稳定的持有管理优势，继续停在研究层。"
    out["role_assignment"] = [
        assign_role_assignment(
            actionability_clear=status == "management_candidate",
            cross_time_stable=status == "management_candidate",
            has_explanatory_value=True,
            config_mode="research",
            ranker_identity="observation_score",
        )
        for status in out["policy_status"]
    ]
    return out


def build_markdown(summary_df: pd.DataFrame, yearly_df: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Observation Tag Holding Management Experiment",
        "",
        f"- research config: `{payload['research_config_path']}`",
        f"- data path: `{payload['data_path']}`",
        f"- evaluation dates: `{payload['evaluation_start_date']} ~ {payload['evaluation_end_date']}`",
        "",
        "## Summary",
        "",
        "| 策略 | 样本数 | 平均持有天数 | 平均收益 | 胜率 | 平均回撤 | 结论 | 角色 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            "| {label} | {sample_count} | {holding:.2f} | {ret:.4f} | {win:.2%} | {dd:.4f} | {status} | {role} |".format(
                label=row.policy_label,
                sample_count=int(row.sample_count),
                holding=float(row.avg_holding_days) if pd.notna(row.avg_holding_days) else 0.0,
                ret=float(row.avg_return) if pd.notna(row.avg_return) else 0.0,
                win=float(row.win_rate) if pd.notna(row.win_rate) else 0.0,
                dd=float(row.avg_drawdown) if pd.notna(row.avg_drawdown) else 0.0,
                status=row.policy_status,
                role=row.role_assignment,
            )
        )
    lines.extend(["", "## Comments", ""])
    for row in summary_df.itertuples(index=False):
        lines.append(f"- `{row.policy_label}`: {row.policy_comment}")
    if not yearly_df.empty:
        lines.extend(
            [
                "",
                "## Yearly Check",
                "",
                "| 年份 | 策略 | 样本数 | 平均持有天数 | 平均收益 | 胜率 | 平均回撤 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in yearly_df.itertuples(index=False):
            lines.append(
                "| {year} | {label} | {sample_count} | {holding:.2f} | {ret:.4f} | {win:.2%} | {dd:.4f} |".format(
                    year=int(row.year),
                    label=row.policy_label,
                    sample_count=int(row.sample_count),
                    holding=float(row.avg_holding_days),
                    ret=float(row.avg_return),
                    win=float(row.win_rate),
                    dd=float(row.avg_drawdown),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Research tag-driven holding path experiments inside observation top20.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--research-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adjust", default="qfq")
    args = parser.parse_args()

    from ashare_quant.data.csv_adapter import CSVDataSource

    research_cfg = load_research_config(args.research_config)
    df = CSVDataSource(args.data_path, adjust=args.adjust).load()
    df, metadata = prepare_research_frame(df, research_cfg)
    dates = sorted(df["date"].drop_duplicates())
    split = int(len(dates) * 0.6)
    train_dates = set(dates[:split])
    test_dates = [pd.Timestamp(d) for d in dates[split:]]
    scored_df, _ = score_research_frame(df=df, research_cfg=research_cfg, metadata=metadata, train_dates=train_dates)
    scored_df = add_holding_columns(scored_df)
    observation_history = build_observation_pool_history(
        scored_df,
        research_cfg=research_cfg,
        observation_top_n=int(research_cfg.get("observation_top_n", 20)),
        selection_dates=test_dates,
        include_premium=False,
    )
    observation_history = add_holding_columns(observation_history)

    summary_rows = [summarize_policy(observation_history, policy_name=name, policy_label=label) for name, label in _policy_specs()]
    summary_df = pd.DataFrame(summary_rows)
    yearly_df = summarize_policy_yearly(observation_history)
    summary_df = assign_policy_decisions(summary_df, yearly_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "holding_policy_summary.csv"
    yearly_path = output_dir / "holding_policy_yearly.csv"
    json_path = output_dir / "holding_policy_experiment.json"
    md_path = output_dir / "holding_policy_experiment.md"
    summary_df.to_csv(summary_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)
    payload = {
        "data_path": str(args.data_path),
        "research_config_path": str(args.research_config),
        "evaluation_start_date": pd.Timestamp(test_dates[0]).strftime("%Y-%m-%d") if test_dates else "",
        "evaluation_end_date": pd.Timestamp(test_dates[-1]).strftime("%Y-%m-%d") if test_dates else "",
        "summary": summary_df.to_dict(orient="records"),
        "yearly": yearly_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(summary_df, yearly_df, payload), encoding="utf-8")

    candidates = summary_df.loc[summary_df["policy_status"] == "management_candidate"]
    print(f"summary_csv={summary_path}")
    print(f"yearly_csv={yearly_path}")
    print(f"markdown={md_path}")
    print(f"json={json_path}")
    print(f"candidate_count={len(candidates)}")
    if not candidates.empty:
        print(candidates.loc[:, ["policy_label", "policy_status", "policy_comment"]].to_string(index=False))
    else:
        print("no_management_candidate")


if __name__ == "__main__":
    main()
