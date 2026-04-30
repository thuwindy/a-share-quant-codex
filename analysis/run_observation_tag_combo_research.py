from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.analysis.latest_picks import build_observation_pool_history
from ashare_quant.pipeline import DEFAULT_RESEARCH, assign_role_assignment, prepare_research_frame, score_research_frame

HOLDING_HORIZONS = [1, 3, 5, 10]
COMBO_RESEARCH_THRESHOLDS = {
    "high_score_rank_cutoff": 10,
    "candidate_min_samples": 30,
    "candidate_min_years": 2,
    "candidate_min_yearly_samples": 10,
    "candidate_min_return_lift_5d": 0.005,
    "candidate_min_return_lift_10d": 0.005,
    "candidate_min_hit_rate_lift_5d": 0.03,
    "candidate_max_drawdown_slack_5d": 0.01,
}
BASELINE_COMBO = "high_score"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_research_config(path: str | Path) -> dict[str, Any]:
    cfg = {**DEFAULT_RESEARCH, **load_json(path)}
    cfg["observe_ml_score"] = False
    cfg["observation_top_n"] = int(cfg.get("observation_top_n", 20))
    return cfg


def add_holding_path_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["code", "date"]).copy()
    close_col = "research_close" if "research_close" in out.columns else "close"
    high_col = "research_high" if "research_high" in out.columns else "high"
    low_col = "research_low" if "research_low" in out.columns else "low"
    g = out.groupby("code", group_keys=False)
    base_close = pd.to_numeric(out[close_col], errors="coerce").replace(0.0, np.nan)

    for horizon in HOLDING_HORIZONS:
        future_close = pd.to_numeric(g[close_col].shift(-horizon), errors="coerce")
        out[f"holding_return_{horizon}d"] = future_close / base_close - 1.0

    future_highs = []
    future_lows = []
    for step in range(1, 6):
        future_highs.append(pd.to_numeric(g[high_col].shift(-step), errors="coerce"))
        future_lows.append(pd.to_numeric(g[low_col].shift(-step), errors="coerce"))
    out["holding_max_up_5d"] = pd.concat(future_highs, axis=1).max(axis=1) / base_close - 1.0
    out["holding_max_drawdown_5d"] = pd.concat(future_lows, axis=1).min(axis=1) / base_close - 1.0
    return out


def classify_path_shape(metrics: dict[str, float]) -> str:
    ret1 = float(metrics.get("avg_return_1d", 0.0))
    ret3 = float(metrics.get("avg_return_3d", 0.0))
    ret5 = float(metrics.get("avg_return_5d", 0.0))
    ret10 = float(metrics.get("avg_return_10d", 0.0))
    max_up_5d = float(metrics.get("avg_max_up_5d", 0.0))
    if ret1 > 0 and ret3 >= ret1 and ret5 >= ret3 and ret10 >= ret5:
        return "持有扩张"
    if max_up_5d - max(ret5, 0.0) >= 0.03 and ret5 <= ret1:
        return "会冲但留不住"
    if ret1 > 0 and ret10 < ret1:
        return "先冲后落"
    if ret5 > 0 or ret10 > 0:
        return "温和延续"
    return "中性偏弱"


def build_combo_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    high_score_cutoff = int(COMBO_RESEARCH_THRESHOLDS["high_score_rank_cutoff"])
    out["combo_top20_all"] = True
    out["combo_high_score"] = out["rank"] <= high_score_cutoff
    out["combo_high_score_spread_strong"] = out["combo_high_score"] & out["industry_leader_follow_tag"].eq("龙头扩散强")
    out["combo_high_score_pressure_low"] = out["combo_high_score"] & out["overhead_density_tag"].eq("兑现压力轻")
    out["combo_high_score_spread_strong_pressure_low"] = (
        out["combo_high_score_spread_strong"] & out["overhead_density_tag"].eq("兑现压力轻")
    )
    return out


def _combo_specs() -> list[tuple[str, str]]:
    return [
        ("top20_all", "主池前20"),
        ("high_score", "高分"),
        ("high_score_spread_strong", "高分 + 扩散强"),
        ("high_score_pressure_low", "高分 + 压力低"),
        ("high_score_spread_strong_pressure_low", "高分 + 扩散强 + 压力低"),
    ]


def summarize_combo(frame: pd.DataFrame, *, combo_name: str, combo_label: str) -> dict[str, Any]:
    subset = frame.loc[frame[f"combo_{combo_name}"]].copy()
    subset = subset.loc[subset["holding_return_5d"].notna()].copy()
    metrics: dict[str, Any] = {
        "combo_name": combo_name,
        "combo_label": combo_label,
        "sample_count": int(len(subset)),
    }
    if subset.empty:
        metrics.update(
            {
                "avg_return_1d": np.nan,
                "avg_return_3d": np.nan,
                "avg_return_5d": np.nan,
                "avg_return_10d": np.nan,
                "win_rate_1d": np.nan,
                "win_rate_3d": np.nan,
                "win_rate_5d": np.nan,
                "win_rate_10d": np.nan,
                "avg_max_up_5d": np.nan,
                "avg_max_drawdown_5d": np.nan,
                "path_shape": "无样本",
            }
        )
        return metrics
    for horizon in HOLDING_HORIZONS:
        col = f"holding_return_{horizon}d"
        values = pd.to_numeric(subset[col], errors="coerce")
        metrics[f"avg_return_{horizon}d"] = float(values.mean())
        metrics[f"win_rate_{horizon}d"] = float((values > 0).mean())
    metrics["avg_max_up_5d"] = float(pd.to_numeric(subset["holding_max_up_5d"], errors="coerce").mean())
    metrics["avg_max_drawdown_5d"] = float(pd.to_numeric(subset["holding_max_drawdown_5d"], errors="coerce").mean())
    metrics["path_shape"] = classify_path_shape(metrics)
    return metrics


def summarize_combo_yearly(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scoped = frame.loc[frame["holding_return_5d"].notna()].copy()
    if scoped.empty:
        return pd.DataFrame(columns=["year", "combo_name", "sample_count", "avg_return_5d"])
    scoped["year"] = pd.to_datetime(scoped["date"]).dt.year.astype(int)
    for combo_name, combo_label in _combo_specs():
        mask = scoped[f"combo_{combo_name}"]
        subset = scoped.loc[mask].copy()
        if subset.empty:
            continue
        for year, yearly in subset.groupby("year", sort=True):
            rows.append(
                {
                    "year": int(year),
                    "combo_name": combo_name,
                    "combo_label": combo_label,
                    "sample_count": int(len(yearly)),
                    "avg_return_5d": float(pd.to_numeric(yearly["holding_return_5d"], errors="coerce").mean()),
                    "avg_return_10d": float(pd.to_numeric(yearly["holding_return_10d"], errors="coerce").mean()),
                    "win_rate_5d": float((pd.to_numeric(yearly["holding_return_5d"], errors="coerce") > 0).mean()),
                    "avg_max_up_5d": float(pd.to_numeric(yearly["holding_max_up_5d"], errors="coerce").mean()),
                    "avg_max_drawdown_5d": float(pd.to_numeric(yearly["holding_max_drawdown_5d"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def assign_combo_decisions(summary_df: pd.DataFrame, yearly_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    out["combo_status"] = "observe_only"
    out["combo_comment"] = ""
    out["role_assignment"] = "observation_label"
    baseline_row = out.loc[out["combo_name"] == BASELINE_COMBO]
    if baseline_row.empty:
        out["combo_comment"] = "缺少高分基线，无法判断是否值得升级。"
        return out
    baseline = baseline_row.iloc[0]
    baseline_yearly = yearly_df.loc[yearly_df["combo_name"] == BASELINE_COMBO].copy()

    for idx, row in out.iterrows():
        if row["combo_name"] == "top20_all":
            out.at[idx, "combo_comment"] = "主池前20的整体底盘，用来对照标签组合。"
            continue
        if row["combo_name"] == BASELINE_COMBO:
            out.at[idx, "combo_comment"] = "主池前20里的高分基线，用来判断标签组合是否真的改善持有路径。"
            continue
        if int(row["sample_count"]) < int(COMBO_RESEARCH_THRESHOLDS["candidate_min_samples"]):
            out.at[idx, "combo_comment"] = "样本太少，先保留观察层，不讨论执行规则。"
            continue
        overall_pass = (
            float(row["avg_return_5d"]) >= float(baseline["avg_return_5d"]) + float(COMBO_RESEARCH_THRESHOLDS["candidate_min_return_lift_5d"])
            and float(row["avg_return_10d"]) >= float(baseline["avg_return_10d"]) + float(COMBO_RESEARCH_THRESHOLDS["candidate_min_return_lift_10d"])
            and float(row["win_rate_5d"]) >= float(baseline["win_rate_5d"]) + float(COMBO_RESEARCH_THRESHOLDS["candidate_min_hit_rate_lift_5d"])
            and float(row["avg_max_drawdown_5d"]) >= float(baseline["avg_max_drawdown_5d"]) - float(COMBO_RESEARCH_THRESHOLDS["candidate_max_drawdown_slack_5d"])
        )
        combo_yearly = yearly_df.loc[yearly_df["combo_name"] == row["combo_name"]].copy()
        merged_yearly = combo_yearly.merge(
            baseline_yearly,
            on="year",
            suffixes=("_combo", "_baseline"),
        )
        merged_yearly = merged_yearly.loc[
            (merged_yearly["sample_count_combo"] >= int(COMBO_RESEARCH_THRESHOLDS["candidate_min_yearly_samples"]))
            & (merged_yearly["sample_count_baseline"] >= int(COMBO_RESEARCH_THRESHOLDS["candidate_min_yearly_samples"]))
        ].copy()
        stable_years = 0
        if not merged_yearly.empty:
            stable_mask = (
                (merged_yearly["avg_return_5d_combo"] > merged_yearly["avg_return_5d_baseline"])
                & (merged_yearly["avg_return_10d_combo"] > merged_yearly["avg_return_10d_baseline"])
                & (
                    merged_yearly["avg_max_drawdown_5d_combo"]
                    >= merged_yearly["avg_max_drawdown_5d_baseline"] - float(COMBO_RESEARCH_THRESHOLDS["candidate_max_drawdown_slack_5d"])
                )
            )
            stable_years = int(stable_mask.sum())
        if overall_pass and stable_years >= int(COMBO_RESEARCH_THRESHOLDS["candidate_min_years"]):
            out.at[idx, "combo_status"] = "execution_rule_candidate"
            out.at[idx, "combo_comment"] = "整体与分年都优于高分基线，才值得进入下一步执行管理规则研究。"
        else:
            out.at[idx, "combo_comment"] = "没有形成稳定的系统级持有路径优势，继续留在观察层。"
    out["role_assignment"] = [
        assign_role_assignment(
            actionability_clear=status == "execution_rule_candidate",
            cross_time_stable=status == "execution_rule_candidate",
            has_explanatory_value=True,
            config_mode="research",
            ranker_identity="observation_score",
        )
        for status in out["combo_status"]
    ]
    return out


def build_markdown(
    *,
    summary_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    output_payload: dict[str, Any],
) -> str:
    lines = [
        "# Observation Tag Combo Research",
        "",
        f"- research config: `{output_payload['research_config_path']}`",
        f"- data path: `{output_payload['data_path']}`",
        f"- evaluation dates: `{output_payload['evaluation_start_date']} ~ {output_payload['evaluation_end_date']}`",
        f"- observation top n: `{output_payload['observation_top_n']}`",
        f"- high-score cutoff: `{output_payload['high_score_rank_cutoff']}`",
        "",
        "## Summary",
        "",
        "| 组合 | 样本数 | 1d | 3d | 5d | 10d | 5d胜率 | 5d最大上冲 | 5d最大回撤 | 持有路径 | 结论 | 角色 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            "| {label} | {sample_count} | {ret1:.4f} | {ret3:.4f} | {ret5:.4f} | {ret10:.4f} | {win5:.2%} | {maxup:.4f} | {drawdown:.4f} | {path_shape} | {status} | {role} |".format(
                label=row.combo_label,
                sample_count=int(row.sample_count),
                ret1=float(row.avg_return_1d) if pd.notna(row.avg_return_1d) else 0.0,
                ret3=float(row.avg_return_3d) if pd.notna(row.avg_return_3d) else 0.0,
                ret5=float(row.avg_return_5d) if pd.notna(row.avg_return_5d) else 0.0,
                ret10=float(row.avg_return_10d) if pd.notna(row.avg_return_10d) else 0.0,
                win5=float(row.win_rate_5d) if pd.notna(row.win_rate_5d) else 0.0,
                maxup=float(row.avg_max_up_5d) if pd.notna(row.avg_max_up_5d) else 0.0,
                drawdown=float(row.avg_max_drawdown_5d) if pd.notna(row.avg_max_drawdown_5d) else 0.0,
                path_shape=row.path_shape,
                status=row.combo_status,
                role=row.role_assignment,
            )
        )
    lines.extend(
        [
            "",
            "## Comments",
            "",
        ]
    )
    for row in summary_df.itertuples(index=False):
        lines.append(f"- `{row.combo_label}`: {row.combo_comment}")
    if not yearly_df.empty:
        lines.extend(
            [
                "",
                "## Yearly 5d/10d Check",
                "",
                "| 年份 | 组合 | 样本数 | 5d | 10d | 5d胜率 | 5d最大上冲 | 5d最大回撤 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in yearly_df.itertuples(index=False):
            lines.append(
                "| {year} | {label} | {sample_count} | {ret5:.4f} | {ret10:.4f} | {win5:.2%} | {maxup:.4f} | {drawdown:.4f} |".format(
                    year=int(row.year),
                    label=row.combo_label,
                    sample_count=int(row.sample_count),
                    ret5=float(row.avg_return_5d),
                    ret10=float(row.avg_return_10d),
                    win5=float(row.win_rate_5d),
                    maxup=float(row.avg_max_up_5d),
                    drawdown=float(row.avg_max_drawdown_5d),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Research observation-tag combinations inside main-strategy top20.")
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
    scored_df, _ = score_research_frame(
        df=df,
        research_cfg=research_cfg,
        metadata=metadata,
        train_dates=train_dates,
    )
    scored_df = add_holding_path_columns(scored_df)
    observation_history = build_observation_pool_history(
        scored_df,
        research_cfg=research_cfg,
        observation_top_n=int(research_cfg.get("observation_top_n", 20)),
        selection_dates=test_dates,
        include_premium=False,
    )
    observation_history = add_holding_path_columns(observation_history)
    observation_history = build_combo_columns(observation_history)

    summary_rows = [summarize_combo(observation_history, combo_name=name, combo_label=label) for name, label in _combo_specs()]
    summary_df = pd.DataFrame(summary_rows)
    yearly_df = summarize_combo_yearly(observation_history)
    summary_df = assign_combo_decisions(summary_df, yearly_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "combo_summary.csv"
    yearly_path = output_dir / "combo_yearly.csv"
    json_path = output_dir / "combo_research.json"
    md_path = output_dir / "combo_research.md"

    summary_df.to_csv(summary_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)
    payload = {
        "data_path": str(args.data_path),
        "research_config_path": str(args.research_config),
        "observation_top_n": int(research_cfg.get("observation_top_n", 20)),
        "high_score_rank_cutoff": int(COMBO_RESEARCH_THRESHOLDS["high_score_rank_cutoff"]),
        "evaluation_start_date": pd.Timestamp(test_dates[0]).strftime("%Y-%m-%d") if test_dates else "",
        "evaluation_end_date": pd.Timestamp(test_dates[-1]).strftime("%Y-%m-%d") if test_dates else "",
        "summary": summary_df.to_dict(orient="records"),
        "yearly": yearly_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(summary_df=summary_df, yearly_df=yearly_df, output_payload=payload), encoding="utf-8")

    best_candidate = summary_df.loc[summary_df["combo_status"] == "execution_rule_candidate"]
    print(f"summary_csv={summary_path}")
    print(f"yearly_csv={yearly_path}")
    print(f"markdown={md_path}")
    print(f"json={json_path}")
    print(f"candidate_count={len(best_candidate)}")
    if not best_candidate.empty:
        print(best_candidate.loc[:, ["combo_label", "combo_status", "combo_comment"]].to_string(index=False))
    else:
        print("no_execution_rule_candidate")


if __name__ == "__main__":
    main()
