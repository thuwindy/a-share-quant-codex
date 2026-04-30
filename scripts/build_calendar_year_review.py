from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.backtest.metrics import summarize_returns
from ashare_quant.data.csv_adapter import CSVDataSource
from ashare_quant.pipeline import run_research_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a calendar-year review pack for the current production strategy.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--research-config", default=str(ROOT / "configs" / "research_production_default.json"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_liquidity_stress_15bps.json"))
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "yearly_review"))
    parser.add_argument("--output-prefix", default="production_year_review")
    return parser


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _year_mask(series: pd.Series, year: int) -> pd.Series:
    return pd.to_datetime(series).dt.year.eq(int(year))


def _pick_representative_stock(targets: pd.DataFrame, *, year: int) -> str | None:
    if targets is None or targets.empty:
        return None
    sl = targets.loc[_year_mask(targets["signal_date"], year)].copy()
    if sl.empty:
        return None
    agg = (
        sl.groupby(["code"], dropna=False)
        .agg(
            selection_count=("code", "size"),
            avg_score=("score", "mean"),
            avg_weight=("target_weight", "mean"),
            top_rank_share=("rank", lambda s: float((pd.to_numeric(s, errors="coerce") <= 10).mean())),
        )
        .reset_index()
    )
    agg["composite"] = (
        agg["selection_count"].astype(float) * 0.45
        + agg["avg_score"].rank(pct=True).astype(float) * 0.30
        + agg["avg_weight"].rank(pct=True).astype(float) * 0.15
        + agg["top_rank_share"].astype(float) * 0.10
    )
    agg = agg.sort_values(["composite", "selection_count", "avg_score"], ascending=False)
    return None if agg.empty else str(agg.iloc[0]["code"])


def _trade_signal_dates_for_code(targets: pd.DataFrame, code: str, *, year: int) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    if targets is None or targets.empty:
        return [], []
    sl = targets.copy()
    sl["execution_date"] = pd.to_datetime(sl["execution_date"])
    year_targets = sl.loc[_year_mask(sl["execution_date"], year)].copy()
    if year_targets.empty:
        return [], []
    execution_dates = sorted(year_targets["execution_date"].drop_duplicates())
    weight_by_date = (
        year_targets.loc[year_targets["code"].astype(str) == str(code)]
        .groupby("execution_date")["target_weight"]
        .sum()
        .to_dict()
    )
    buys: list[pd.Timestamp] = []
    sells: list[pd.Timestamp] = []
    prev_weight = 0.0
    for dt in execution_dates:
        weight = float(weight_by_date.get(dt, 0.0))
        if weight > 1e-12 and prev_weight <= 1e-12:
            buys.append(pd.Timestamp(dt))
        elif weight <= 1e-12 and prev_weight > 1e-12:
            sells.append(pd.Timestamp(dt))
        prev_weight = weight
    if prev_weight > 1e-12:
        last_dt = pd.Timestamp(execution_dates[-1])
        sells.append(last_dt)
    return buys, sells


def _plot_equity_curve(year_result: pd.DataFrame, output_path: Path) -> str:
    if year_result.empty:
        return ""
    out = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(year_result["date"], year_result["gross_equity"], color="#9ca3af", linewidth=1.5, label="gross")
    ax.plot(year_result["date"], year_result["equity"], color="#2563eb", linewidth=2.0, label="net")
    ax.set_title("2025 Net Value Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _plot_drawdown_curve(year_result: pd.DataFrame, output_path: Path) -> str:
    if year_result.empty:
        return ""
    out = _ensure_parent(output_path)
    peak = year_result["equity"].cummax()
    drawdown = year_result["equity"] / peak - 1.0
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(year_result["date"], drawdown, 0.0, color="#dc2626", alpha=0.35)
    ax.plot(year_result["date"], drawdown, color="#b91c1c", linewidth=1.6)
    ax.set_title("2025 Drawdown Curve")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _plot_return_distribution(year_result: pd.DataFrame, output_path: Path) -> str:
    if year_result.empty:
        return ""
    out = _ensure_parent(output_path)
    returns = pd.to_numeric(year_result["net_return"], errors="coerce").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.hist(returns, bins=32, color="#0891b2", alpha=0.85, edgecolor="white")
    ax.axvline(float(returns.mean()), color="#b91c1c", linestyle="--", linewidth=1.6, label="mean")
    ax.axvline(float(returns.median()), color="#1d4ed8", linestyle=":", linewidth=1.6, label="median")
    ax.set_title("2025 Daily Return Distribution")
    ax.grid(alpha=0.20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _plot_candles_with_signals(
    stock_df: pd.DataFrame,
    buys: list[pd.Timestamp],
    sells: list[pd.Timestamp],
    title: str,
    output_path: Path,
) -> str:
    if stock_df.empty:
        return ""
    out = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    data = stock_df.sort_values("date").copy()
    x = mdates.date2num(pd.to_datetime(data["date"]))
    width = 0.6
    for _, row in data.iterrows():
        dt = mdates.date2num(pd.Timestamp(row["date"]))
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        color = "#16a34a" if c >= o else "#dc2626"
        ax.vlines(dt, l, h, color=color, linewidth=1.0, alpha=0.8)
        lower = min(o, c)
        height = max(abs(c - o), 1e-4)
        rect = Rectangle((dt - width / 2, lower), width, height, facecolor=color, edgecolor=color, alpha=0.55)
        ax.add_patch(rect)
    if buys:
        buy_df = data.loc[data["date"].isin(pd.to_datetime(buys))]
        ax.scatter(
            mdates.date2num(pd.to_datetime(buy_df["date"])),
            buy_df["low"] * 0.98,
            marker="^",
            color="#1d4ed8",
            s=80,
            label="buy",
            zorder=5,
        )
    if sells:
        sell_df = data.loc[data["date"].isin(pd.to_datetime(sells))]
        ax.scatter(
            mdates.date2num(pd.to_datetime(sell_df["date"])),
            sell_df["high"] * 1.02,
            marker="v",
            color="#7c3aed",
            s=80,
            label="sell",
            zorder=5,
        )
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.xaxis_date()
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _build_plot_ohlc_frame(stock_df: pd.DataFrame) -> pd.DataFrame:
    price_map = {
        "open": "research_open" if "research_open" in stock_df.columns else "open",
        "high": "research_high" if "research_high" in stock_df.columns else "high",
        "low": "research_low" if "research_low" in stock_df.columns else "low",
        "close": "research_close" if "research_close" in stock_df.columns else "close",
    }
    cols = ["date", *price_map.values()]
    out = stock_df.loc[:, cols].copy()
    out = out.rename(columns={v: k for k, v in price_map.items()})
    out["date"] = pd.to_datetime(out["date"])
    return out.dropna()


if __name__ == "__main__":
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.output_prefix}_{args.year}"

    print(
        json.dumps(
            {
                "stage": "build_calendar_year_review:start",
                "year": int(args.year),
                "data_path": str(args.data_path),
                "research_config": str(args.research_config),
                "backtest_config": str(args.backtest_config),
                "adjust": str(args.adjust),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    result, metrics, _score_details, targets = run_research_pipeline(
        data_path=args.data_path,
        research_config_path=args.research_config,
        backtest_config_path=args.backtest_config,
        data_adjust=args.adjust,
    )
    result = result.copy()
    result["date"] = pd.to_datetime(result["date"])
    year_result = result.loc[_year_mask(result["date"], args.year)].copy().reset_index(drop=True)
    if year_result.empty:
        raise SystemExit(f"No result rows found for calendar year {args.year}.")
    year_metrics = summarize_returns(
        year_result["net_return"],
        turnover=year_result["turnover"],
        gross_returns=year_result["gross_return"],
        annual_days=int(metrics.get("annual_trading_days", 252)),
        concentration=year_result["max_position_weight"],
        cost=year_result["cost"],
    )
    representative_code = _pick_representative_stock(targets, year=args.year)
    stock_chart_path = ""
    stock_payload: dict[str, object] = {}
    if representative_code:
        source = CSVDataSource(args.data_path, adjust=args.adjust)
        panel = source.load_daily_bars(start=f"{args.year}-01-01", end=f"{args.year}-12-31")
        stock_df = panel.loc[panel["code"].astype(str) == representative_code].copy()
        stock_name = str(stock_df["name"].dropna().iloc[-1]) if "name" in stock_df.columns and not stock_df["name"].dropna().empty else representative_code
        buys, sells = _trade_signal_dates_for_code(targets, representative_code, year=args.year)
        plot_df = _build_plot_ohlc_frame(stock_df)
        stock_chart_path = _plot_candles_with_signals(
            plot_df,
            buys,
            sells,
            title=f"{args.year} Representative Stock: {stock_name} ({representative_code})",
            output_path=output_dir / f"{prefix}_{representative_code.replace('.', '_')}_signals.png",
        )
        stock_payload = {
            "code": representative_code,
            "name": stock_name,
            "buy_dates": [pd.Timestamp(v).strftime("%Y-%m-%d") for v in buys],
            "sell_dates": [pd.Timestamp(v).strftime("%Y-%m-%d") for v in sells],
            "signal_chart_path": stock_chart_path,
        }

    equity_path = _plot_equity_curve(year_result, output_dir / f"{prefix}_equity_curve.png")
    drawdown_path = _plot_drawdown_curve(year_result, output_dir / f"{prefix}_drawdown_curve.png")
    dist_path = _plot_return_distribution(year_result, output_dir / f"{prefix}_return_distribution.png")

    summary = {
        "year": int(args.year),
        "research_config": str(args.research_config),
        "backtest_config": str(args.backtest_config),
        "data_path": str(args.data_path),
        "adjust": str(args.adjust),
        "lock_manifest": {
            "data_sha256": _sha256(args.data_path),
            "research_config_sha256": _sha256(args.research_config),
            "backtest_config_sha256": _sha256(args.backtest_config),
            "script_sha256": _sha256(__file__),
        },
        "metrics": year_metrics,
        "equity_curve_path": equity_path,
        "drawdown_curve_path": drawdown_path,
        "return_distribution_path": dist_path,
        "representative_stock": stock_payload,
    }
    json_path = output_dir / f"{prefix}_summary.json"
    md_path = output_dir / f"{prefix}_summary.md"
    manifest_path = output_dir / f"{prefix}_lock_manifest.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary["lock_manifest"], ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = [
        f"# {args.year} Calendar-Year Review",
        "",
        "## Locked Inputs",
        "",
        f"- data_path: `{args.data_path}`",
        f"- research_config: `{args.research_config}`",
        f"- backtest_config: `{args.backtest_config}`",
        f"- adjust: `{args.adjust}`",
        f"- data_sha256: `{summary['lock_manifest']['data_sha256']}`",
        f"- research_config_sha256: `{summary['lock_manifest']['research_config_sha256']}`",
        f"- backtest_config_sha256: `{summary['lock_manifest']['backtest_config_sha256']}`",
        f"- script_sha256: `{summary['lock_manifest']['script_sha256']}`",
        f"- manifest_path: `{manifest_path}`",
        "",
        "## Metrics",
        "",
        f"- annual_return: `{float(year_metrics.get('annual_return', 0.0)):.4f}`",
        f"- sharpe: `{float(year_metrics.get('sharpe', 0.0)):.4f}`",
        f"- max_drawdown: `{float(year_metrics.get('max_drawdown', 0.0)):.4f}`",
        f"- hit_rate: `{float(year_metrics.get('hit_rate', 0.0)):.4f}`",
        f"- payoff_ratio: `{float(year_metrics.get('payoff_ratio', 0.0)):.4f}`",
        f"- equity_curve_path: `{equity_path}`",
        f"- drawdown_curve_path: `{drawdown_path}`",
        f"- return_distribution_path: `{dist_path}`",
        "",
    ]
    if stock_payload:
        md_lines.extend(
            [
                "## Representative Stock",
                "",
                f"- code: `{stock_payload.get('code', '')}`",
                f"- name: `{stock_payload.get('name', '')}`",
                f"- buy_dates: `{', '.join(stock_payload.get('buy_dates', []))}`",
                f"- sell_dates: `{', '.join(stock_payload.get('sell_dates', []))}`",
                f"- signal_chart_path: `{stock_payload.get('signal_chart_path', '')}`",
                "",
            ]
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "build_calendar_year_review:done",
                "summary_json": str(json_path),
                "summary_md": str(md_path),
                "lock_manifest_json": str(manifest_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(json.dumps({"summary_path": str(json_path), "markdown_path": str(md_path), **summary}, ensure_ascii=False, indent=2))
