#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
DATA_PATH="${DATA_PATH:-$ROOT/data/daily_monitor_main_slice.csv}"
BACKTEST_CONFIG="${BACKTEST_CONFIG:-$ROOT/configs/backtest_production_managed_15bps.json}"
BENCH_OUTPUT_DIR="${BENCH_OUTPUT_DIR:-$ROOT/outputs/locked_benchmark_2025}"
STYLE_OUTPUT_DIR="${STYLE_OUTPUT_DIR:-$ROOT/outputs/style_compare_cloud}"
RESEARCH_WEEKDAY_MODE="${RESEARCH_WEEKDAY_MODE:-rotate}"
STYLE_SET_TUE="${STYLE_SET_TUE:-trend,rule}"
STYLE_SET_THU="${STYLE_SET_THU:-ml_classification,hybrid}"

mkdir -p "$ROOT/outputs/ops_logs" "$BENCH_OUTPUT_DIR" "$STYLE_OUTPUT_DIR"

echo "[INFO] cloud research pack start"
echo "[INFO] repo_root: $ROOT"
echo "[INFO] data_path: $DATA_PATH"

weekday="$(date +%u)"
task_mode="all"
if [[ "$RESEARCH_WEEKDAY_MODE" == "rotate" ]]; then
  case "$weekday" in
    1|3|5) task_mode="benchmark" ;;
    2) task_mode="style_tue" ;;
    4) task_mode="style_thu" ;;
    *) task_mode="none" ;;
  esac
fi

case "$task_mode" in
  benchmark|all)
    echo "[TASK] locked 2025 benchmark"
    "$PYTHON_BIN" -u "$ROOT/scripts/run_locked_2025_benchmark.py" \
      --data-path "$DATA_PATH" \
      > "$ROOT/outputs/ops_logs/manual_locked_2025_benchmark.log" 2>&1
    [[ "$task_mode" == "benchmark" ]] && { echo "[OK] cloud research pack finished"; exit 0; }
    ;;&
  style_tue)
    echo "[TASK] style compare: $STYLE_SET_TUE"
    "$PYTHON_BIN" -u "$ROOT/scripts/run_style_parallel_compare.py" \
      --data-path "$DATA_PATH" \
      --backtest-config "$BACKTEST_CONFIG" \
      --styles "$STYLE_SET_TUE" \
      --output-dir "$STYLE_OUTPUT_DIR" \
      --output-prefix style_compare_cloud \
      > "$ROOT/outputs/ops_logs/manual_style_compare.log" 2>&1
    ;;
  style_thu)
    echo "[TASK] style compare: $STYLE_SET_THU"
    "$PYTHON_BIN" -u "$ROOT/scripts/run_style_parallel_compare.py" \
      --data-path "$DATA_PATH" \
      --backtest-config "$BACKTEST_CONFIG" \
      --styles "$STYLE_SET_THU" \
      --output-dir "$STYLE_OUTPUT_DIR" \
      --output-prefix style_compare_cloud \
      > "$ROOT/outputs/ops_logs/manual_style_compare.log" 2>&1
    ;;
  none)
    echo "[INFO] cloud research pack skip on weekday=$weekday"
    ;;
esac

echo "[OK] cloud research pack finished"
