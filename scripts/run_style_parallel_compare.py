from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.backtest.metrics import summarize_returns
from ashare_quant.pipeline import run_research_pipeline


DEFAULT_STYLES = {
    "trend": ROOT / "configs" / "research_production_default.json",
    "rule": ROOT / "configs" / "research_production_rule.json",
    "ml_classification": ROOT / "configs" / "research_production_ml_classification.json",
    "hybrid": ROOT / "configs" / "research_production_hybrid.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run parallel style comparison plus monthly/quarterly style rotation.")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_production_managed_15bps.json"))
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--rotation-frequency", choices=("monthly", "quarterly"), default="monthly")
    parser.add_argument("--lookback-days", type=int, default=63)
    parser.add_argument("--styles", default=",".join(DEFAULT_STYLES.keys()))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "style_compare"))
    parser.add_argument("--output-prefix", default="style_compare")
    return parser


def _sha256(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _period_key(dates: pd.Series, freq: str) -> pd.Series:
    return pd.to_datetime(dates).dt.to_period("M" if freq == "monthly" else "Q")


def _rotation_backtest(returns_df: pd.DataFrame, *, frequency: str, lookback_days: int) -> tuple[pd.DataFrame, dict[str, float]]:
    work = returns_df.copy().sort_values("date").reset_index(drop=True)
    work["period"] = _period_key(work["date"], frequency)
    style_cols = [col for col in work.columns if col not in {"date", "period"}]
    chosen: list[str] = []
    rotated_returns: list[float] = []
    for period, sl in work.groupby("period", sort=True):
        period_start = pd.Timestamp(sl["date"].min())
        hist = work.loc[work["date"] < period_start].tail(int(lookback_days))
        if hist.empty:
            style = style_cols[0]
        else:
            style_scores: list[tuple[str, float, float]] = []
            for col in style_cols:
                m = summarize_returns(hist[col], annual_days=252)
                style_scores.append((col, float(m.get("sharpe", 0.0)), float(m.get("annual_return", 0.0))))
            style_scores.sort(key=lambda item: (item[1], item[2]), reverse=True)
            style = style_scores[0][0]
        chosen.extend([style] * len(sl))
        rotated_returns.extend(pd.to_numeric(sl[style], errors="coerce").fillna(0.0).tolist())
    out = work[["date"]].copy()
    out["chosen_style"] = chosen
    out["net_return"] = rotated_returns
    metrics = summarize_returns(out["net_return"], annual_days=252)
    return out, metrics


if __name__ == "__main__":
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix)
    requested_styles = [item.strip() for item in str(args.styles).split(",") if item.strip()]
    selected_styles = {name: DEFAULT_STYLES[name] for name in requested_styles if name in DEFAULT_STYLES}
    if not selected_styles:
        raise ValueError(f"No valid styles selected from: {args.styles}")
    manifest = {
        "data_path": str(args.data_path),
        "backtest_config": str(args.backtest_config),
        "adjust": str(args.adjust),
        "rotation_frequency": str(args.rotation_frequency),
        "lookback_days": int(args.lookback_days),
        "styles": list(selected_styles.keys()),
        "data_sha256": _sha256(args.data_path),
        "backtest_config_sha256": _sha256(args.backtest_config),
        "script_sha256": _sha256(__file__),
        "style_config_sha256": {style: _sha256(cfg_path) for style, cfg_path in selected_styles.items()},
    }
    print(json.dumps({"stage": "style_compare:start", **manifest}, ensure_ascii=False), flush=True)

    summaries: list[dict[str, object]] = []
    merged_returns: pd.DataFrame | None = None
    for style_name, cfg_path in selected_styles.items():
        print(json.dumps({"stage": "style_compare:style_start", "style": style_name, "config": str(cfg_path)}, ensure_ascii=False), flush=True)
        result, metrics, _, _ = run_research_pipeline(
            data_path=args.data_path,
            research_config_path=str(cfg_path),
            backtest_config_path=args.backtest_config,
            data_adjust=args.adjust,
        )
        summaries.append(
            {
                "style": style_name,
                "research_config": str(cfg_path),
                "annual_return": float(metrics.get("annual_return", 0.0)),
                "sharpe": float(metrics.get("sharpe", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "hit_rate": float(metrics.get("hit_rate", 0.0)),
                "payoff_ratio": float(metrics.get("payoff_ratio", 0.0)),
                "avg_turnover": float(metrics.get("avg_turnover", 0.0)),
            }
        )
        style_returns = result[["date", "net_return"]].copy().rename(columns={"net_return": style_name})
        merged_returns = style_returns if merged_returns is None else merged_returns.merge(style_returns, on="date", how="inner")
        print(
            json.dumps(
                {
                    "stage": "style_compare:style_done",
                    "style": style_name,
                    "annual_return": float(metrics.get("annual_return", 0.0)),
                    "sharpe": float(metrics.get("sharpe", 0.0)),
                    "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary_df = pd.DataFrame(summaries).sort_values(["sharpe", "annual_return"], ascending=False).reset_index(drop=True)
    rotation_df, rotation_metrics = _rotation_backtest(
        merged_returns if merged_returns is not None else pd.DataFrame(columns=["date"]),
        frequency=str(args.rotation_frequency),
        lookback_days=int(args.lookback_days),
    )
    rotation_choice_df = (
        rotation_df.groupby(_period_key(rotation_df["date"], str(args.rotation_frequency)))["chosen_style"]
        .last()
        .reset_index()
        .rename(columns={"date": "period"})
    )
    summary_path = output_dir / f"{prefix}_summary.csv"
    rotation_path = output_dir / f"{prefix}_rotation.csv"
    rotation_metrics_path = output_dir / f"{prefix}_rotation_metrics.json"
    manifest_path = output_dir / f"{prefix}_lock_manifest.json"
    md_path = output_dir / f"{prefix}_summary.md"
    summary_df.to_csv(summary_path, index=False)
    rotation_choice_df.to_csv(rotation_path, index=False)
    rotation_metrics_path.write_text(json.dumps(rotation_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = [
        "# Style Parallel Compare",
        "",
        "## Locked Inputs",
        "",
        f"- data_path: `{args.data_path}`",
        f"- backtest_config: `{args.backtest_config}`",
        f"- adjust: `{args.adjust}`",
        f"- rotation_frequency: `{args.rotation_frequency}`",
        f"- lookback_days: `{int(args.lookback_days)}`",
        f"- data_sha256: `{manifest['data_sha256']}`",
        f"- backtest_config_sha256: `{manifest['backtest_config_sha256']}`",
        f"- script_sha256: `{manifest['script_sha256']}`",
        f"- manifest_path: `{manifest_path}`",
        "",
        "## Style Summary",
        "",
        "| style | annual_return | sharpe | max_drawdown | hit_rate | payoff_ratio | avg_turnover |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_df.itertuples(index=False):
        md_lines.append(
            "| {style} | {annual_return:.4f} | {sharpe:.4f} | {max_drawdown:.4f} | {hit_rate:.4f} | {payoff_ratio:.4f} | {avg_turnover:.4f} |".format(
                style=getattr(row, "style"),
                annual_return=float(getattr(row, "annual_return", 0.0)),
                sharpe=float(getattr(row, "sharpe", 0.0)),
                max_drawdown=float(getattr(row, "max_drawdown", 0.0)),
                hit_rate=float(getattr(row, "hit_rate", 0.0)),
                payoff_ratio=float(getattr(row, "payoff_ratio", 0.0)),
                avg_turnover=float(getattr(row, "avg_turnover", 0.0)),
            )
        )
    md_lines.extend(
        [
            "",
            "## Rotation Metrics",
            "",
            f"- frequency: `{args.rotation_frequency}`",
            f"- lookback_days: `{int(args.lookback_days)}`",
            f"- annual_return: `{float(rotation_metrics.get('annual_return', 0.0)):.4f}`",
            f"- sharpe: `{float(rotation_metrics.get('sharpe', 0.0)):.4f}`",
            f"- max_drawdown: `{float(rotation_metrics.get('max_drawdown', 0.0)):.4f}`",
            "",
            f"- summary_csv: `{summary_path}`",
            f"- rotation_csv: `{rotation_path}`",
            f"- rotation_metrics_json: `{rotation_metrics_path}`",
            f"- lock_manifest_json: `{manifest_path}`",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({
        "stage": "style_compare:done",
        "summary_csv": str(summary_path),
        "rotation_csv": str(rotation_path),
        "rotation_metrics_json": str(rotation_metrics_path),
        "lock_manifest_json": str(manifest_path),
        "markdown_path": str(md_path),
    }, ensure_ascii=False, indent=2))
