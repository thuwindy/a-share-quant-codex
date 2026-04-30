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


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _latest_monitor_json(monitor_dir: str | Path) -> Path:
    files = sorted(Path(monitor_dir).glob("*_picks.json"))
    if not files:
        raise FileNotFoundError(f"No monitor pick payload found under {monitor_dir}.")
    return files[-1]


def _infer_metrics_path(monitor_json_path: Path) -> Path:
    return monitor_json_path.with_name(monitor_json_path.name.replace("_picks.json", "_metrics.json"))


def _build_report_markdown(summary: dict) -> str:
    checks = summary.get("checks", [])
    metrics_fresh = bool(summary.get("metrics_fresh", False))
    lines = [
        "# Risk Governor",
        "",
        f"- monitor_json: `{summary.get('monitor_json', '')}`",
        f"- metrics_json: `{summary.get('metrics_json', '')}`",
        f"- selection_date: `{summary.get('selection_date', '')}`",
        f"- status: `{summary.get('status', '')}`",
        f"- action_hint: `{summary.get('action_hint', '')}`",
        f"- metrics_fresh: `{metrics_fresh}`",
        "",
        "## Core Snapshot",
        "",
        f"- annual_return: `{summary.get('annual_return')}`",
        f"- sharpe: `{summary.get('sharpe')}`",
        f"- max_drawdown: `{summary.get('max_drawdown')}`",
        f"- avg_turnover: `{summary.get('avg_turnover')}`",
        f"- selected_count: `{summary.get('selected_count', 0)}`",
        f"- top_industry_weight: `{summary.get('top_industry_weight', 0.0):.4f}`",
        f"- max_single_weight: `{summary.get('max_single_weight', 0.0):.4f}`",
        "",
        "## Checks",
        "",
        "| check | value | threshold | ok | severity |",
        "| --- | --- | --- | --- | --- |",
    ]
    if checks:
        for item in checks:
            lines.append(
                "| {name} | {value} | {threshold} | {ok} | {severity} |".format(
                    name=item.get("name", ""),
                    value=item.get("value", ""),
                    threshold=item.get("threshold", ""),
                    ok="yes" if item.get("ok", False) else "no",
                    severity=item.get("severity", "warn"),
                )
            )
    else:
        lines.append("| - | - | - | - | - |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate latest monitor result with risk-governor thresholds.")
    parser.add_argument("--monitor-json", default="")
    parser.add_argument("--monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_auto"))
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--risk-config", default=str(ROOT / "configs" / "risk_governor.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "risk_governor"))
    parser.add_argument("--output-prefix", default="risk_gate")
    return parser


def _check_metric(
    *,
    name: str,
    value: float,
    threshold: float,
    op: str,
    severity: str = "warn",
) -> dict:
    if op == "ge":
        ok = value >= threshold
    elif op == "le":
        ok = value <= threshold
    else:
        raise ValueError(f"Unsupported op: {op}")
    return {
        "name": name,
        "value": round(float(value), 6),
        "threshold": round(float(threshold), 6),
        "ok": bool(ok),
        "severity": severity,
    }


def main() -> None:
    args = build_parser().parse_args()
    monitor_json_path = Path(args.monitor_json) if args.monitor_json else _latest_monitor_json(args.monitor_dir)
    metrics_json_path = Path(args.metrics_json) if args.metrics_json else _infer_metrics_path(monitor_json_path)

    payload = _load_json(monitor_json_path)
    metrics = _load_json(metrics_json_path) if metrics_json_path.exists() else {}
    risk_cfg = _load_json(args.risk_config)

    strategy_snapshot = pd.DataFrame(payload.get("strategy_snapshot", []))
    if strategy_snapshot.empty:
        strategy_snapshot = pd.DataFrame(payload.get("observation_pool", []))
    if strategy_snapshot.empty:
        raise ValueError(f"No strategy_snapshot or observation_pool found in {monitor_json_path}.")
    strategy_snapshot["target_weight"] = pd.to_numeric(strategy_snapshot["target_weight"], errors="coerce").fillna(0.0)
    strategy_snapshot["industry"] = strategy_snapshot.get("industry", pd.Series("Unknown", index=strategy_snapshot.index)).fillna("Unknown").astype(str)

    selected = strategy_snapshot.loc[strategy_snapshot["target_weight"] > 0].copy()
    selected_count = int(len(selected))
    max_single_weight = float(selected["target_weight"].max()) if not selected.empty else 0.0
    if selected.empty:
        top_industry_weight = 0.0
    else:
        industry_weight = selected.groupby("industry")["target_weight"].sum().sort_values(ascending=False)
        top_industry_weight = float(industry_weight.iloc[0]) if not industry_weight.empty else 0.0

    metrics_fresh = metrics_json_path.exists() and bool(metrics)
    annual_return = float(metrics.get("annual_return")) if metrics_fresh and metrics.get("annual_return") is not None else None
    sharpe = float(metrics.get("sharpe")) if metrics_fresh and metrics.get("sharpe") is not None else None
    max_drawdown = float(metrics.get("max_drawdown")) if metrics_fresh and metrics.get("max_drawdown") is not None else None
    avg_turnover = float(metrics.get("avg_turnover")) if metrics_fresh and metrics.get("avg_turnover") is not None else None

    checks = []
    if annual_return is not None:
        checks.append(
            _check_metric(
                name="annual_return",
                value=annual_return,
                threshold=float(risk_cfg.get("min_annual_return", 0.0)),
                op="ge",
                severity="warn",
            )
        )
    if sharpe is not None:
        checks.append(
            _check_metric(
                name="sharpe",
                value=sharpe,
                threshold=float(risk_cfg.get("min_sharpe", 0.2)),
                op="ge",
                severity="warn",
            )
        )
    if max_drawdown is not None:
        checks.append(
            _check_metric(
                name="max_drawdown",
                value=max_drawdown,
                threshold=float(risk_cfg.get("max_drawdown", -0.3)),
                op="ge",
                severity="block" if bool(risk_cfg.get("block_drawdown_breach", True)) else "warn",
            )
        )
    if avg_turnover is not None:
        checks.append(
            _check_metric(
                name="avg_turnover",
                value=avg_turnover,
                threshold=float(risk_cfg.get("max_avg_turnover", 0.2)),
                op="le",
                severity="warn",
            )
        )
    checks.extend([
        _check_metric(
            name="top_industry_weight",
            value=top_industry_weight,
            threshold=float(risk_cfg.get("max_top_industry_weight", 0.2)),
            op="le",
            severity="block" if bool(risk_cfg.get("block_concentration_breach", True)) else "warn",
        ),
        _check_metric(
            name="max_single_weight",
            value=max_single_weight,
            threshold=float(risk_cfg.get("max_single_weight", 0.12)),
            op="le",
            severity="warn",
        ),
        _check_metric(
            name="selected_count",
            value=float(selected_count),
            threshold=float(risk_cfg.get("min_selected_count", 12)),
            op="ge",
            severity="block",
        ),
    ])

    failed_block = [item for item in checks if not item["ok"] and item["severity"] == "block"]
    failed_warn = [item for item in checks if not item["ok"] and item["severity"] == "warn"]
    if failed_block:
        status = "BLOCK"
    elif failed_warn:
        status = "WARN"
    else:
        status = "PASS"
    action_hint = {
        "PASS": "normal sizing",
        "WARN": "reduce sizing / review",
        "BLOCK": "review only",
    }.get(status, "review only")

    selection_date = str(payload.get("summary", {}).get("selection_date", ""))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = selection_date.replace("-", "") if selection_date else pd.Timestamp.today().strftime("%Y%m%d")
    prefix = f"{args.output_prefix}_{suffix}"

    summary = {
        "selection_date": selection_date,
        "status": status,
        "action_hint": action_hint,
        "monitor_json": str(monitor_json_path),
        "metrics_json": str(metrics_json_path),
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_turnover": avg_turnover,
        "metrics_fresh": metrics_fresh,
        "selected_count": selected_count,
        "top_industry_weight": top_industry_weight,
        "max_single_weight": max_single_weight,
        "checks": checks,
    }

    json_path = out_dir / f"{prefix}.json"
    md_path = out_dir / f"{prefix}.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_report_markdown(summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] risk governor outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
