from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _require_fastapi():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    return FastAPI, HTMLResponse


def _latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_table(path: Path | None, limit: int = 12) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, dict) and isinstance(payload.get("observation_pool"), list):
        return payload["observation_pool"][:limit]
    if isinstance(payload, list):
        return payload[:limit]
    return []


def _dashboard_html(root_dir: Path) -> str:
    monitor_path = _latest_file(root_dir / "outputs" / "daily_monitor_auto", "*_metrics.json")
    picks_path = _latest_file(root_dir / "outputs" / "daily_monitor_auto", "*_picks.json")
    under20_path = _latest_file(root_dir / "outputs" / "daily_monitor_under20_elastic", "*_picks.json")
    risk_path = _latest_file(root_dir / "outputs" / "risk_governor", "risk_gate_*.json")
    monitor = _load_json(monitor_path)
    risk = _load_json(risk_path)
    main_rows = _load_table(picks_path, limit=12)
    under20_rows = _load_table(under20_path, limit=12)
    risk_status = str(risk.get("status", "n/a"))
    risk_color = {"PASS": "#0a7f2e", "WARN": "#b77800", "BLOCK": "#b00020"}.get(risk_status, "#555555")

    def _rows_html(rows: list[dict[str, Any]]) -> str:
        body = []
        for i, row in enumerate(rows, start=1):
            body.append(
                "<tr>"
                f"<td>{i}</td>"
                f"<td>{row.get('code', '')}</td>"
                f"<td>{row.get('name', '')}</td>"
                f"<td>{float(row.get('target_weight', 0.0)):.4f}</td>"
                f"<td>{float(row.get('score', 0.0)):.4f}</td>"
                f"<td>{row.get('industry', '')}</td>"
                "</tr>"
            )
        return "".join(body) or "<tr><td colspan='6'>no rows</td></tr>"

    return f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>A-share Quant Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; background: #f5f7fb; color: #1c2430; }}
    .grid {{ display:grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 14px; margin-bottom: 24px; }}
    .card {{ background:#fff; border-radius:14px; padding:16px 18px; box-shadow:0 6px 18px rgba(0,0,0,0.06); }}
    .k {{ font-size:12px; color:#6b7280; margin-bottom:8px; }}
    .v {{ font-size:24px; font-weight:700; }}
    .section {{ background:#fff; border-radius:16px; padding:18px; margin-top:18px; box-shadow:0 6px 18px rgba(0,0,0,0.06); }}
    table {{ width:100%; border-collapse: collapse; }}
    th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #e5e7eb; font-size:13px; }}
    .status {{ color:{risk_color}; font-weight:700; }}
    .wrap {{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
  </style>
</head>
<body>
  <h1>A-share Quant Desk</h1>
  <div class="grid">
    <div class="card"><div class="k">Risk Status</div><div class="v status">{risk_status}</div></div>
    <div class="card"><div class="k">Annual Return</div><div class="v">{float(monitor.get("annual_return", 0.0)):.2%}</div></div>
    <div class="card"><div class="k">Sharpe</div><div class="v">{float(monitor.get("sharpe", 0.0)):.4f}</div></div>
    <div class="card"><div class="k">Max Drawdown</div><div class="v">{float(monitor.get("max_drawdown", 0.0)):.2%}</div></div>
  </div>
  <div class="wrap">
    <div class="section">
      <h2>Main Strategy Top 12</h2>
      <table>
        <thead><tr><th>#</th><th>Code</th><th>Name</th><th>Weight</th><th>Score</th><th>Industry</th></tr></thead>
        <tbody>{_rows_html(main_rows)}</tbody>
      </table>
    </div>
    <div class="section">
      <h2>Under20 Elastic Top 12</h2>
      <table>
        <thead><tr><th>#</th><th>Code</th><th>Name</th><th>Weight</th><th>Score</th><th>Industry</th></tr></thead>
        <tbody>{_rows_html(under20_rows)}</tbody>
      </table>
    </div>
  </div>
  <div class="section">
    <h2>Source Files</h2>
    <ul>
      <li>monitor_metrics: {monitor_path or "n/a"}</li>
      <li>main_picks: {picks_path or "n/a"}</li>
      <li>under20_picks: {under20_path or "n/a"}</li>
      <li>risk_gate: {risk_path or "n/a"}</li>
    </ul>
  </div>
</body>
</html>
"""


def create_app(root_dir: str | Path | None = None):
    FastAPI, HTMLResponse = _require_fastapi()
    root = Path(root_dir or Path(__file__).resolve().parents[3])
    app = FastAPI(title="A-share Quant Dashboard", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"ok": True, "root_dir": str(root)}

    @app.get("/latest/monitor")
    def latest_monitor() -> dict[str, Any]:
        return _load_json(_latest_file(root / "outputs" / "daily_monitor_auto", "*_metrics.json"))

    @app.get("/latest/under20")
    def latest_under20() -> dict[str, Any]:
        return _load_json(_latest_file(root / "outputs" / "daily_monitor_under20_elastic", "*_picks.json"))

    @app.get("/latest/risk")
    def latest_risk() -> dict[str, Any]:
        return _load_json(_latest_file(root / "outputs" / "risk_governor", "risk_gate_*.json"))

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return _dashboard_html(root)

    return app
