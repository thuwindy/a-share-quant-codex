#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
ADJUST_MODE="${ADJUST_MODE:-qfq}"
MAIN_SLICE_PATH="${MAIN_SLICE_PATH:-$ROOT/data/daily_monitor_main_slice.csv}"
ELASTIC_SLICE_PATH="${ELASTIC_SLICE_PATH:-$ROOT/data/daily_monitor_elastic_slice.csv}"

echo "[INFO] run split paper monitors"
echo "[INFO] repo_root: $ROOT"
echo "[INFO] python: $PYTHON_BIN"

mkdir -p \
  "$ROOT/outputs/paper_monitor_main" \
  "$ROOT/outputs/paper_monitor_elastic"

echo "[TASK] main strategy paper monitor"
"$PYTHON_BIN" "$ROOT/scripts/run_paper_monitor.py" \
  --monitor-dir "$ROOT/outputs/daily_monitor_auto" \
  --slice-path "$MAIN_SLICE_PATH" \
  --adjust "$ADJUST_MODE" \
  --paper-config "$ROOT/configs/paper_trade_guardrails_main.json" \
  --backtest-config "$ROOT/configs/backtest_production_managed_15bps.json" \
  --state-path "$ROOT/outputs/paper_monitor_main/paper_state.json" \
  --output-dir "$ROOT/outputs/paper_monitor_main"

echo "[TASK] elastic strategy paper monitor"
"$PYTHON_BIN" "$ROOT/scripts/run_paper_monitor.py" \
  --monitor-dir "$ROOT/outputs/daily_monitor_under20_elastic" \
  --slice-path "$ELASTIC_SLICE_PATH" \
  --adjust "$ADJUST_MODE" \
  --paper-config "$ROOT/configs/paper_trade_guardrails_elastic.json" \
  --backtest-config "$ROOT/configs/backtest_production_managed_15bps.json" \
  --state-path "$ROOT/outputs/paper_monitor_elastic/paper_state.json" \
  --output-dir "$ROOT/outputs/paper_monitor_elastic"

echo "[OK] split paper monitors finished"
