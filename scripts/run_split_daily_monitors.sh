#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
DATA_PATH="${DATA_PATH:-$ROOT/data/a_share_daily_industry.csv}"
ADJUST_MODE="${ADJUST_MODE:-qfq}"

MAIN_SLICE_PATH="${MAIN_SLICE_PATH:-$ROOT/data/daily_monitor_main_slice.csv}"
ELASTIC_SLICE_PATH="${ELASTIC_SLICE_PATH:-$ROOT/data/daily_monitor_elastic_slice.csv}"

MAIN_OUTPUT_DIR="${MAIN_OUTPUT_DIR:-$ROOT/outputs/daily_monitor_auto}"
ELASTIC_OUTPUT_DIR="${ELASTIC_OUTPUT_DIR:-$ROOT/outputs/daily_monitor_under20_elastic}"

MAIN_RESEARCH_CONFIG="${MAIN_RESEARCH_CONFIG:-$ROOT/configs/research_production_default.json}"
ELASTIC_RESEARCH_CONFIG="${ELASTIC_RESEARCH_CONFIG:-$ROOT/configs/research_under20_elastic_top20.json}"
BACKTEST_CONFIG="${BACKTEST_CONFIG:-$ROOT/configs/backtest_production_managed_15bps.json}"
TARGET_DATE="${TARGET_DATE:-}"
FORCE_REBUILD_SLICES="${FORCE_REBUILD_SLICES:-0}"
SLICE_START_DATE="${SLICE_START_DATE:-2019-01-01}"
ELASTIC_SLICE_START_DATE="${ELASTIC_SLICE_START_DATE:-2023-01-01}"
ELASTIC_MAX_CODES="${ELASTIC_MAX_CODES:-200}"
ELASTIC_CHAIN_MODE="${ELASTIC_CHAIN_MODE:-observation_only}"

mkdir -p "$ROOT/outputs/ops_logs" "$MAIN_OUTPUT_DIR" "$ELASTIC_OUTPUT_DIR"

is_true() {
  [[ "${1,,}" =~ ^(1|true|yes|y|on)$ ]]
}

latest_csv_date() {
  tail -n 1 "$1" | cut -d',' -f1 || true
}

if [[ -z "$TARGET_DATE" ]]; then
  TARGET_DATE="$(latest_csv_date "$DATA_PATH")"
fi
if [[ -z "$TARGET_DATE" ]]; then
  echo "[ERROR] failed to resolve target date from data path: $DATA_PATH" >&2
  exit 1
fi

echo "[INFO] split daily monitors start"
echo "[INFO] repo_root: $ROOT"
echo "[INFO] data_path: $DATA_PATH"
echo "[INFO] main_slice: $MAIN_SLICE_PATH"
echo "[INFO] elastic_slice: $ELASTIC_SLICE_PATH"
echo "[INFO] target_date: $TARGET_DATE"

if is_true "$FORCE_REBUILD_SLICES"; then
  echo "[STEP] force rebuild slices"
  rm -f "$MAIN_SLICE_PATH" "$ELASTIC_SLICE_PATH"
fi

echo "[TASK] main daily monitor"
"$PYTHON_BIN" -u "$ROOT/scripts/run_daily_monitor.py" \
  --skip-update \
  --skip-backtest \
  --data-path "$DATA_PATH" \
  --slice-path "$MAIN_SLICE_PATH" \
  --end-date "$TARGET_DATE" \
  --adjust "$ADJUST_MODE" \
  --research-config "$MAIN_RESEARCH_CONFIG" \
  --backtest-config "$BACKTEST_CONFIG" \
  --output-dir "$MAIN_OUTPUT_DIR"

MAIN_SLICE_DATE="$(latest_csv_date "$MAIN_SLICE_PATH")"
if [[ "$MAIN_SLICE_DATE" != "$TARGET_DATE" ]]; then
  echo "[ERROR] main slice stale: expected $TARGET_DATE got ${MAIN_SLICE_DATE:-unknown}" >&2
  exit 1
fi

echo "[TASK] elastic daily monitor"
if [[ "$ELASTIC_CHAIN_MODE" == "observation_only" ]]; then
  echo "[INFO] elastic chain mode: observation_only"
  "$PYTHON_BIN" -u "$ROOT/scripts/build_monitor_observation_output.py" \
    --input-path "$DATA_PATH" \
    --slice-path "$ELASTIC_SLICE_PATH" \
    --start-date "$ELASTIC_SLICE_START_DATE" \
    --end-date "$TARGET_DATE" \
    --max-codes "$ELASTIC_MAX_CODES" \
    --research-config "$ELASTIC_RESEARCH_CONFIG" \
    --output-dir "$ELASTIC_OUTPUT_DIR" \
    --output-prefix under20_elastic \
    --prediction-date "$TARGET_DATE" \
    --adjust "$ADJUST_MODE"
else
  if ! "$PYTHON_BIN" -u "$ROOT/scripts/run_daily_monitor.py" \
    --skip-update \
    --skip-backtest \
    --data-path "$DATA_PATH" \
    --slice-path "$ELASTIC_SLICE_PATH" \
    --end-date "$TARGET_DATE" \
    --adjust "$ADJUST_MODE" \
    --research-config "$ELASTIC_RESEARCH_CONFIG" \
    --backtest-config "$BACKTEST_CONFIG" \
    --output-dir "$ELASTIC_OUTPUT_DIR" \
    --output-prefix under20_elastic; then
    echo "[WARN] elastic daily monitor failed, fallback to observation-only output"
    ELASTIC_SLICE_DATE="$(latest_csv_date "$ELASTIC_SLICE_PATH")"
    if [[ "$ELASTIC_SLICE_DATE" != "$TARGET_DATE" ]]; then
      echo "[ERROR] elastic fallback aborted: expected slice $TARGET_DATE got ${ELASTIC_SLICE_DATE:-unknown}" >&2
      exit 1
    fi
    "$PYTHON_BIN" -u "$ROOT/scripts/build_monitor_observation_output.py" \
      --data-path "$ELASTIC_SLICE_PATH" \
      --research-config "$ELASTIC_RESEARCH_CONFIG" \
      --output-dir "$ELASTIC_OUTPUT_DIR" \
      --output-prefix under20_elastic \
      --prediction-date "$TARGET_DATE" \
      --adjust "$ADJUST_MODE"
  fi
fi

ELASTIC_SLICE_DATE="$(latest_csv_date "$ELASTIC_SLICE_PATH")"
if [[ "$ELASTIC_SLICE_DATE" != "$TARGET_DATE" ]]; then
  echo "[ERROR] elastic slice stale: expected $TARGET_DATE got ${ELASTIC_SLICE_DATE:-unknown}" >&2
  exit 1
fi

echo "[OK] split daily monitors finished"
