from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.candidate_pool_quality import add_candidate_path_columns, summarize_candidate_pool_quality
from ashare_quant.data.csv_adapter import CSVDataSource
from ashare_quant.factors.factor_registry import FACTOR_REGISTRY, summarize_factor_families
from ashare_quant.models.dynamic_weighting import spearman_corr
from ashare_quant.pipeline import (
    DEFAULT_RESEARCH,
    assign_role_assignment,
    load_json,
    prepare_research_frame,
    resolve_factor_columns,
    score_research_frame,
)

RESEARCH_DECISION_THRESHOLDS = {
    "keep_min_ic_mean": 0.02,
    "keep_min_ir": 0.20,
    "observe_min_abs_ic_mean": 0.005,
    "observe_min_ir": 0.05,
    "warn_missing_rate": 0.35,
    "reject_missing_rate": 0.60,
    "warn_max_abs_corr_to_main": 0.75,
    "reject_max_abs_corr_to_main": 0.90,
}


def _resolve_label_alias(research_cfg: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    cfg = research_cfg.copy()
    requested = str(cfg.get("label_type", "raw")).strip()
    if requested == "future_return_5d":
        cfg["label_type"] = "raw"
        cfg["label_horizons"] = [5]
        return cfg, "forward_return_5d", requested
    if requested == "industry_excess":
        horizon = int((cfg.get("label_horizons") or [cfg.get("label_horizon", 5)])[0])
        return cfg, f"forward_return_{horizon}d_industry_excess", requested
    if requested == "neutralized_residual":
        horizon = int((cfg.get("label_horizons") or [cfg.get("label_horizon", 5)])[0])
        return cfg, f"forward_return_{horizon}d_neutralized_residual", requested
    if requested == "raw":
        horizon = int((cfg.get("label_horizons") or [cfg.get("label_horizon", 5)])[0])
        return cfg, f"forward_return_{horizon}d", requested
    if re.fullmatch(r"(high|close)_\d+d_up", requested):
        horizon = int(re.findall(r"(\d+)d", requested)[0])
        return cfg, f"label_{requested}", requested
    raise ValueError(f"unsupported research label_type: {requested}")


def _daily_factor_ic(df: pd.DataFrame, factor_col: str, label_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, sl in df.groupby("date", sort=True):
        part = sl[[factor_col, label_col]].dropna()
        ic = spearman_corr(part[factor_col], part[label_col]) if len(part) >= 5 else np.nan
        if pd.notna(ic):
            rows.append({"date": pd.Timestamp(date), "ic": float(ic)})
    return pd.DataFrame(rows)


def _quantile_summary(df: pd.DataFrame, factor_col: str, label_col: str, bins: int = 5) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date, sl in df.groupby("date", sort=True):
        part = sl[[factor_col, label_col]].dropna().copy()
        if len(part) < bins:
            continue
        rank_pct = part[factor_col].rank(method="first", pct=True)
        bucket = np.ceil(rank_pct * bins).clip(1, bins).astype(int)
        part["bucket"] = bucket
        grouped = part.groupby("bucket", as_index=False)[label_col].mean()
        grouped["date"] = pd.Timestamp(date)
        rows.append(grouped.rename(columns={label_col: "label_mean"}))
    if not rows:
        empty = pd.DataFrame(columns=["bucket", "label_mean"])
        return empty, {"monotonicity": "insufficient", "direction": "flat", "top_bottom_spread": 0.0}
    detail = pd.concat(rows, ignore_index=True)
    summary = detail.groupby("bucket", as_index=False)["label_mean"].mean().sort_values("bucket").reset_index(drop=True)
    diffs = summary["label_mean"].diff().dropna()
    top_bottom_spread = float(summary["label_mean"].iloc[-1] - summary["label_mean"].iloc[0])
    if abs(top_bottom_spread) <= 1e-12:
        direction = "flat"
        monotonic_ratio = 0.0
    else:
        sign = 1.0 if top_bottom_spread > 0 else -1.0
        monotonic_ratio = float(((np.sign(diffs) == sign) | (diffs.abs() <= 1e-12)).mean()) if not diffs.empty else 0.0
        direction = "ascending" if sign > 0 else "descending"
    return summary, {
        "monotonicity": "pass" if monotonic_ratio >= 0.75 else "warn",
        "direction": direction,
        "monotonic_ratio": monotonic_ratio,
        "top_bottom_spread": top_bottom_spread,
    }


def _yearly_ic_table(ic_df: pd.DataFrame) -> pd.DataFrame:
    if ic_df.empty:
        return pd.DataFrame(columns=["year", "ic_mean", "ic_std", "ir", "observations"])
    out = ic_df.copy()
    out["year"] = out["date"].dt.year
    yearly = out.groupby("year", as_index=False)["ic"].agg(["mean", "std", "count"]).reset_index()
    yearly = yearly.rename(columns={"mean": "ic_mean", "std": "ic_std", "count": "observations"})
    yearly["ir"] = yearly["ic_mean"] / yearly["ic_std"].replace(0.0, np.nan)
    yearly["ir"] = yearly["ir"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return yearly


def _factor_summary_rows(
    df: pd.DataFrame,
    *,
    factor_cols: list[str],
    label_col: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    yearly_tables: dict[str, pd.DataFrame] = {}
    quantile_tables: dict[str, pd.DataFrame] = {}
    for factor_col in factor_cols:
        factor_frame = df[[factor_col, label_col, "date"]].copy()
        ic_df = _daily_factor_ic(df, factor_col=factor_col, label_col=label_col)
        ic_mean = float(ic_df["ic"].mean()) if not ic_df.empty else 0.0
        ic_std = float(ic_df["ic"].std(ddof=0)) if len(ic_df) > 1 else 0.0
        ir = float(ic_mean / ic_std) if ic_std > 1e-12 else 0.0
        quantile_df, monotonic = _quantile_summary(df, factor_col=factor_col, label_col=label_col, bins=5)
        yearly_df = _yearly_ic_table(ic_df)
        raw_name = factor_col.removesuffix("_neu")
        rows.append(
            {
                "factor": raw_name,
                "factor_col": factor_col,
                "family": FACTOR_REGISTRY.get(raw_name).family if raw_name in FACTOR_REGISTRY else "unknown",
                "missing_rate": float(factor_frame[factor_col].isna().mean()),
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ir": ir,
                "quantile_direction": monotonic["direction"],
                "quantile_monotonicity": monotonic["monotonicity"],
                "quantile_monotonic_ratio": monotonic.get("monotonic_ratio", 0.0),
                "top_bottom_spread": monotonic["top_bottom_spread"],
            }
        )
        yearly_tables[raw_name] = yearly_df
        quantile_tables[raw_name] = quantile_df
    summary = pd.DataFrame(rows).sort_values(["ir", "ic_mean"], ascending=False).reset_index(drop=True)
    return summary, yearly_tables, quantile_tables


def _correlation_matrix(df: pd.DataFrame, selected_cols: list[str], base_cols: list[str]) -> pd.DataFrame:
    cols = [col for col in selected_cols + base_cols if col in df.columns]
    if not cols:
        return pd.DataFrame()
    corr = df[cols].corr(method="spearman")
    keep = [col for col in selected_cols if col in corr.index]
    base_keep = [col for col in base_cols if col in corr.columns]
    if not keep or not base_keep:
        return pd.DataFrame()
    return corr.loc[keep, base_keep]


def _research_decision(row: pd.Series, thresholds: dict[str, float]) -> tuple[str, str]:
    missing_rate = float(row.get("missing_rate", 1.0) or 1.0)
    ic_mean = float(row.get("ic_mean", 0.0) or 0.0)
    ir = float(row.get("ir", 0.0) or 0.0)
    monotonicity = str(row.get("quantile_monotonicity", "warn"))
    max_corr = float(row.get("max_abs_corr_to_main", 0.0) or 0.0)

    if (
        missing_rate >= thresholds["reject_missing_rate"]
        or abs(ic_mean) < thresholds["observe_min_abs_ic_mean"]
        or ir < thresholds["observe_min_ir"]
        or max_corr >= thresholds["reject_max_abs_corr_to_main"]
    ):
        return "reject", "IC/IR偏弱、缺失率过高或与主因子重复度过高，当前阶段不建议继续投入。"

    if (
        ic_mean >= thresholds["keep_min_ic_mean"]
        and ir >= thresholds["keep_min_ir"]
        and monotonicity == "pass"
        and missing_rate <= thresholds["warn_missing_rate"]
        and max_corr < thresholds["warn_max_abs_corr_to_main"]
    ):
        return "keep", "单因子表现较完整，值得进入下一阶段研究或组合实验。"

    return "observe", "有一定研究价值，但稳定性、缺失率或独立性仍需继续观察。"


def run_factor_family_research(
    *,
    data_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    adjust: str = "none",
    main_config_path: str | Path | None = None,
) -> dict[str, Any]:
    research_cfg = {**DEFAULT_RESEARCH, **load_json(config_path)}
    research_cfg, label_col, requested_label = _resolve_label_alias(research_cfg)
    data = CSVDataSource(data_path, adjust=adjust).load()
    frame, metadata = prepare_research_frame(data, research_cfg)
    factor_cols = [col for col in metadata["factor_cols"] if col in frame.columns]
    eligible = frame.loc[frame[label_col].notna()].copy()
    summary_df, yearly_tables, quantile_tables = _factor_summary_rows(
        eligible,
        factor_cols=factor_cols,
        label_col=label_col,
    )
    main_cfg_path = Path(main_config_path) if main_config_path else ROOT / "configs" / "research_production_default.json"
    main_cfg = {**DEFAULT_RESEARCH, **load_json(main_cfg_path)}
    _, main_factor_cols = resolve_factor_columns(main_cfg)
    corr_df = _correlation_matrix(eligible, selected_cols=factor_cols, base_cols=main_factor_cols)
    if not corr_df.empty:
        max_corr = corr_df.abs().max(axis=1).groupby(level=0).max().rename("max_abs_corr_to_main")
        summary_df["max_abs_corr_to_main"] = summary_df["factor_col"].map(max_corr).fillna(0.0)
    else:
        summary_df["max_abs_corr_to_main"] = 0.0
    thresholds = {
        **RESEARCH_DECISION_THRESHOLDS,
        **(research_cfg.get("research_decision_thresholds") or {}),
    }
    decisions = summary_df.apply(lambda row: _research_decision(row, thresholds), axis=1) if not summary_df.empty else []
    if len(summary_df):
        summary_df["research_status"] = [item[0] for item in decisions]
        summary_df["research_comment"] = [item[1] for item in decisions]
        summary_df["role_assignment"] = [
            assign_role_assignment(
                research_status=status,
                has_explanatory_value=(status == "observe"),
                has_system_alpha=False,
                config_mode="research",
                ranker_identity="research_ranker",
            )
            for status in summary_df["research_status"]
        ]
    family_summary = summarize_factor_families([col.removesuffix("_neu") for col in factor_cols])
    dates = sorted(frame["date"].drop_duplicates())
    split = int(len(dates) * 0.6)
    train_dates = set(dates[:split])
    scored_df, _ = score_research_frame(
        df=frame,
        research_cfg=research_cfg,
        metadata=metadata,
        train_dates=train_dates,
    )
    holding_horizon = int(research_cfg.get("holding_period", research_cfg.get("holding_window", 5)))
    scored_df = add_candidate_path_columns(scored_df, horizons={holding_horizon})
    candidate_quality_df = summarize_candidate_pool_quality(
        scored_df.loc[~scored_df["date"].isin(train_dates)].copy(),
        score_col="score",
        eligibility_col="strategy_tradeable",
        top_ns=research_cfg.get("candidate_pool_top_ns", [20, 10, 5]),
        holding_horizon=holding_horizon,
    )
    candidate_quality_rows = candidate_quality_df.to_dict(orient="records")
    candidate_pool_headline = (
        "候选池质量暂无有效样本。"
        if candidate_quality_df.empty
        else "候选池质量先看 top{top_n}：平均收益 {avg_return:.4f}，命中率 {hit_rate:.2%}，胜率 {win_rate:.2%}，平均回撤 {avg_drawdown:.4f}。".format(
            **candidate_quality_df.sort_values(["avg_return", "win_rate"], ascending=False).iloc[0].to_dict()
        )
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_root / "factor_summary.csv", index=False)
    candidate_quality_df.to_csv(output_root / "candidate_pool_quality.csv", index=False)
    corr_df.to_csv(output_root / "factor_correlation_vs_main.csv", index=True)
    yearly_concat = []
    for factor, yearly in yearly_tables.items():
        tmp = yearly.copy()
        tmp.insert(0, "factor", factor)
        yearly_concat.append(tmp)
        yearly.to_csv(output_root / f"{factor}_yearly_ic.csv", index=False)
    quantile_concat = []
    for factor, quantile in quantile_tables.items():
        tmp = quantile.copy()
        tmp.insert(0, "factor", factor)
        quantile_concat.append(tmp)
        quantile.to_csv(output_root / f"{factor}_quantile_summary.csv", index=False)
    yearly_df = pd.concat(yearly_concat, ignore_index=True) if yearly_concat else pd.DataFrame()
    quantile_df = pd.concat(quantile_concat, ignore_index=True) if quantile_concat else pd.DataFrame()
    if not yearly_df.empty:
        yearly_df.to_csv(output_root / "yearly_ic_summary.csv", index=False)
    if not quantile_df.empty:
        quantile_df.to_csv(output_root / "quantile_summary.csv", index=False)

    markdown_lines = [
        "# Factor Family Research",
        "",
        f"- label_type: `{requested_label}`",
        f"- resolved_label_col: `{label_col}`",
        f"- factor_families: `{', '.join(sorted(family_summary['families'].keys()))}`",
        f"- contains_low_freq_experimental: `{family_summary['contains_low_freq_experimental']}`",
        "",
        "## Candidate Pool Quality",
        "",
        candidate_pool_headline,
        "",
        candidate_quality_df.to_markdown(index=False) if not candidate_quality_df.empty else "暂无候选池质量结果。",
        "",
        "## Factor Summary",
        "",
        summary_df.to_markdown(index=False) if not summary_df.empty else "暂无结果。",
        "",
        "## Research Decision Guide",
        "",
        "- `keep` 代表值得进入下一阶段研究或组合实验，不代表直接进入生产。",
        "- `observe` 代表继续观察，不建议立即升级。",
        "- `reject` 代表当前阶段不建议继续投入。",
        "",
        "## Family Mix",
        "",
        json.dumps(family_summary, ensure_ascii=False, indent=2),
    ]
    if family_summary["contains_low_freq_experimental"]:
        markdown_lines.extend(
            [
                "",
                "## Low Frequency Risk",
                "",
                "低频基本面因子默认标记为 `low_freq_experimental`。它们依赖公告可见时点，不应直接进入生产主策略。",
            ]
        )
    (output_root / "factor_family_research.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    payload = {
        "label_type": requested_label,
        "label_col": label_col,
        "research_summary": metadata.get("research_summary", {}),
        "research_decision_thresholds": thresholds,
        "candidate_pool_quality": candidate_quality_rows,
        "candidate_pool_quality_headline": candidate_pool_headline,
        "factor_summary_rows": summary_df.to_dict(orient="records"),
        "family_summary": family_summary,
    }
    (output_root / "factor_family_research.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run factor family research without portfolio backtest.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="none")
    parser.add_argument("--main-config", default=str(ROOT / "configs" / "research_production_default.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_factor_family_research(
        data_path=args.data_path,
        config_path=args.config,
        output_dir=args.output_dir,
        adjust=args.adjust,
        main_config_path=args.main_config,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
