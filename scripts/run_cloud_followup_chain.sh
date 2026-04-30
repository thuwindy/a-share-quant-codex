#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
LOG="${LOG:-$ROOT/outputs/ops_logs/manual_followup_chain.log}"

mkdir -p "$ROOT/outputs/ops_logs"
: > "$LOG"

echo "[INFO] followup chain start $(date +'%F %T %Z')" >> "$LOG"

ELASTIC_PATTERN="run_daily_monitor.py --skip-update --skip-backtest --data-path data/a_share_daily_industry.csv --slice-path data/daily_monitor_elastic_slice.csv --adjust qfq --research-config configs/research_under20_elastic_top20.json"

while pgrep -f "$ELASTIC_PATTERN" >/dev/null; do
  echo "[WAIT] elastic direct still running $(date +'%F %T')" >> "$LOG"
  sleep 60
done

if ls "$ROOT"/outputs/daily_monitor_under20_elastic_direct/*_picks.json >/dev/null 2>&1; then
  echo "[STEP] elastic paper $(date +'%F %T')" >> "$LOG"
  rm -rf "$ROOT/outputs/paper_monitor_elastic_direct"
  mkdir -p "$ROOT/outputs/paper_monitor_elastic_direct"
  PYTHONPATH="$ROOT/src" "$PYTHON_BIN" "$ROOT/scripts/run_paper_monitor.py" \
    --monitor-dir "$ROOT/outputs/daily_monitor_under20_elastic_direct" \
    --slice-path "$ROOT/data/daily_monitor_elastic_slice.csv" \
    --adjust qfq \
    --paper-config "$ROOT/configs/paper_trade_guardrails_elastic.json" \
    --backtest-config "$ROOT/configs/backtest_production_managed_15bps.json" \
    --state-path "$ROOT/outputs/paper_monitor_elastic_direct/paper_state.json" \
    --output-dir "$ROOT/outputs/paper_monitor_elastic_direct" >> "$LOG" 2>&1

  echo "[STEP] dual report $(date +'%F %T')" >> "$LOG"
  rm -rf "$ROOT/outputs/daily_monitor_dual_direct"
  mkdir -p "$ROOT/outputs/daily_monitor_dual_direct"
  PYTHONPATH="$ROOT/src" "$PYTHON_BIN" "$ROOT/scripts/build_dual_board_report.py" \
    --main-dir "$ROOT/outputs/daily_monitor_auto_direct" \
    --elastic-dir "$ROOT/outputs/daily_monitor_under20_elastic_direct" \
    --risk-dir "$ROOT/outputs/risk_governor_direct" \
    --output-dir "$ROOT/outputs/daily_monitor_dual_direct" \
    --output-prefix dual_monitor >> "$LOG" 2>&1
else
  echo "[WARN] elastic direct produced no picks; skip elastic paper and dual report" >> "$LOG"
fi

echo "[STEP] locked benchmark $(date +'%F %T')" >> "$LOG"
PYTHONPATH="$ROOT/src" "$PYTHON_BIN" "$ROOT/scripts/run_locked_2025_benchmark.py" >> "$LOG" 2>&1 || echo "[WARN] locked benchmark failed" >> "$LOG"

echo "[STEP] style compare $(date +'%F %T')" >> "$LOG"
PYTHONPATH="$ROOT/src" "$PYTHON_BIN" "$ROOT/scripts/run_style_parallel_compare.py" \
  --data-path "$ROOT/data/a_share_daily_industry.csv" \
  --output-dir "$ROOT/outputs/style_compare_cloud" \
  --output-prefix style_compare_cloud >> "$LOG" 2>&1 || echo "[WARN] style compare failed" >> "$LOG"

echo "[DONE] followup chain end $(date +'%F %T %Z')" >> "$LOG"
