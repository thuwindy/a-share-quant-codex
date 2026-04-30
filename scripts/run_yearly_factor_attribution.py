from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.performance_eval import build_calendar_metrics
from ashare_quant.pipeline import load_json, run_research_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run natural-year single-factor attribution from an existing research config.")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "daily_monitor_slice.csv"))
    parser.add_argument("--research-config", default=str(ROOT / "configs" / "research_production_default.json"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_liquidity_stress_15bps.json"))
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "factor_attribution"))
    parser.add_argument("--output-prefix", default="yearly_factor_attribution")
    return parser


def _raw_name(column_name: str) -> str:
    return str(column_name).removesuffix("_neu")


if __name__ == "__main__":
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    research_cfg = load_json(args.research_config)
    factor_columns = list(research_cfg.get("factor_columns") or [])
    penalty_columns = set(research_cfg.get("score_penalty_columns") or [])
    rows: list[dict[str, object]] = []

    for factor in factor_columns:
        cfg = dict(research_cfg)
        cfg["factor_set"] = "custom"
        cfg["factor_columns"] = [factor]
        cfg["score_penalty_columns"] = []
        raw_factor = _raw_name(factor)
        cfg["horizon_factor_map"] = {str(h): [raw_factor] for h in cfg.get("use_horizons", cfg.get("label_horizons", [cfg.get("label_horizon", 5)]))}
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
            tmp_path = Path(handle.name)
            handle.write(json.dumps(cfg, ensure_ascii=False, indent=2))
        try:
            result, metrics, _score_details, _targets = run_research_pipeline(
                data_path=args.data_path,
                research_config_path=tmp_path,
                backtest_config_path=args.backtest_config,
                data_adjust=args.adjust,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        yearly = build_calendar_metrics(result, freq="Y", annual_days=int(metrics.get("annual_trading_days", 252)))
        role = "penalty" if factor in penalty_columns else "alpha"
        for record in yearly.to_dict(orient="records"):
            record["factor"] = factor
            record["role"] = role
            rows.append(record)

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{args.output_prefix}.csv"
    json_path = output_dir / f"{args.output_prefix}.json"
    md_path = output_dir / f"{args.output_prefix}.md"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    md_lines = [
        "# Yearly Factor Attribution",
        "",
        f"- research_config: `{args.research_config}`",
        f"- backtest_config: `{args.backtest_config}`",
        "",
        "| factor | role | period | annual_return | sharpe | max_drawdown | hit_rate | payoff_ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in df.itertuples(index=False):
        md_lines.append(
            "| {factor} | {role} | {period} | {annual_return:.4f} | {sharpe:.4f} | {max_drawdown:.4f} | {hit_rate:.4f} | {payoff_ratio:.4f} |".format(
                factor=getattr(row, "factor", ""),
                role=getattr(row, "role", ""),
                period=getattr(row, "period", ""),
                annual_return=float(getattr(row, "annual_return", 0.0)),
                sharpe=float(getattr(row, "sharpe", 0.0)),
                max_drawdown=float(getattr(row, "max_drawdown", 0.0)),
                hit_rate=float(getattr(row, "hit_rate", 0.0)),
                payoff_ratio=float(getattr(row, "payoff_ratio", 0.0)),
            )
        )
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"rows": len(df), "csv_path": str(csv_path), "json_path": str(json_path), "md_path": str(md_path)}, ensure_ascii=False, indent=2))
