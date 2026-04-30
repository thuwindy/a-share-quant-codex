#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
DATA_PATH="${DATA_PATH:-$ROOT/data/a_share_daily_industry.csv}"
ADJUST_MODE="${ADJUST_MODE:-qfq}"
RUN_DATA_UPDATE="${RUN_DATA_UPDATE:-1}"
ENV_FILE="${ENV_FILE:-$ROOT/.localhome/post_close_monitor.env}"

MAIN_OUTPUT_DIR="${MAIN_OUTPUT_DIR:-$ROOT/outputs/daily_monitor_auto}"
ELASTIC_OUTPUT_DIR="${ELASTIC_OUTPUT_DIR:-$ROOT/outputs/daily_monitor_under20_elastic}"
RISK_OUTPUT_DIR="${RISK_OUTPUT_DIR:-$ROOT/outputs/risk_governor}"
DUAL_OUTPUT_DIR="${DUAL_OUTPUT_DIR:-$ROOT/outputs/daily_monitor_dual}"
SHORTLINE_OUTPUT_DIR="${SHORTLINE_OUTPUT_DIR:-$ROOT/outputs/shortline_opportunities}"
MAIN_SLICE_PATH="${MAIN_SLICE_PATH:-$ROOT/data/daily_monitor_main_slice.csv}"
ELASTIC_SLICE_PATH="${ELASTIC_SLICE_PATH:-$ROOT/data/daily_monitor_elastic_slice.csv}"
MAIN_RESEARCH_CONFIG="${MAIN_RESEARCH_CONFIG:-$ROOT/configs/research_production_default.json}"
ELASTIC_RESEARCH_CONFIG="${ELASTIC_RESEARCH_CONFIG:-$ROOT/configs/research_under20_elastic_top20.json}"
BACKTEST_CONFIG="${BACKTEST_CONFIG:-$ROOT/configs/backtest_production_managed_15bps.json}"
RUN_SHORTLINE_REPORT="${RUN_SHORTLINE_REPORT:-1}"
DATA_UPDATE_RETRIES="${DATA_UPDATE_RETRIES:-3}"
DATA_UPDATE_SLEEP_SECONDS="${DATA_UPDATE_SLEEP_SECONDS:-180}"
ALLOW_UNCHANGED_MASTER_DATE="${ALLOW_UNCHANGED_MASTER_DATE:-0}"

mkdir -p \
  "$ROOT/outputs/ops_logs" \
  "$MAIN_OUTPUT_DIR" \
  "$ELASTIC_OUTPUT_DIR" \
  "$RISK_OUTPUT_DIR" \
  "$DUAL_OUTPUT_DIR" \
  "$SHORTLINE_OUTPUT_DIR"

export HOME="${HOME:-$ROOT/.localhome}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/matplotlib}"
mkdir -p "$HOME" "$MPLCONFIGDIR"

echo "[INFO] cloud post-close stable workflow start"
echo "[INFO] repo_root: $ROOT"
echo "[INFO] data_path: $DATA_PATH"

latest_csv_date() {
  tail -n 1 "$1" | cut -d',' -f1 || true
}

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$ENV_FILE"
  set +a
fi

if [[ "${RUN_DATA_UPDATE,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  BEFORE_DATE="$(latest_csv_date "$DATA_PATH")"
  echo "[STEP] update daily master"
  echo "[INFO] pre-update latest date: ${BEFORE_DATE:-unknown}"
  ATTEMPT=1
  AFTER_DATE="${BEFORE_DATE:-}"
  while [[ "$ATTEMPT" -le "$DATA_UPDATE_RETRIES" ]]; do
    echo "[INFO] update attempt: $ATTEMPT/$DATA_UPDATE_RETRIES"
    "$PYTHON_BIN" -u "$ROOT/scripts/update_tushare_daily_dataset.py" \
      --existing-path "$DATA_PATH" \
      --bypass-system-proxy
    AFTER_DATE="$(latest_csv_date "$DATA_PATH")"
    if [[ -n "${AFTER_DATE:-}" && "${AFTER_DATE:-}" != "${BEFORE_DATE:-}" ]]; then
      break
    fi
    if [[ "$ATTEMPT" -lt "$DATA_UPDATE_RETRIES" ]]; then
      echo "[WARN] master date unchanged after attempt $ATTEMPT: ${AFTER_DATE:-unknown}"
      echo "[INFO] sleep ${DATA_UPDATE_SLEEP_SECONDS}s before retry"
      sleep "$DATA_UPDATE_SLEEP_SECONDS"
    fi
    ATTEMPT=$((ATTEMPT + 1))
  done
  echo "[INFO] post-update latest date: ${AFTER_DATE:-unknown}"
  if [[ "${ALLOW_UNCHANGED_MASTER_DATE,,}" != "1" && "${AFTER_DATE:-}" == "${BEFORE_DATE:-}" ]]; then
    echo "[ERROR] master data did not advance after update attempts; refusing to build stale post-close outputs" >&2
    exit 2
  fi
fi

TARGET_DATE="${AFTER_DATE:-$(latest_csv_date "$DATA_PATH")}"
if [[ -z "${TARGET_DATE:-}" ]]; then
  echo "[ERROR] failed to resolve target date from master data: $DATA_PATH" >&2
  exit 1
fi
export TARGET_DATE
export FORCE_REBUILD_SLICES=1
echo "[INFO] target_date: $TARGET_DATE"

echo "[STEP] split daily monitors"
bash "$ROOT/scripts/run_split_daily_monitors.sh"

echo "[STEP] refresh main metrics"
if ! "$PYTHON_BIN" -u "$ROOT/scripts/refresh_daily_monitor_metrics.py" \
  --monitor-dir "$MAIN_OUTPUT_DIR" \
  --slice-path "$MAIN_SLICE_PATH" \
  --research-config "$MAIN_RESEARCH_CONFIG" \
  --backtest-config "$BACKTEST_CONFIG" \
  --adjust "$ADJUST_MODE"; then
  echo "[WARN] refresh main metrics failed; continue downstream artifacts and let preflight repair metrics later" >&2
fi

echo "[STEP] risk governor"
"$PYTHON_BIN" -u "$ROOT/scripts/run_risk_governor.py" \
  --monitor-dir "$MAIN_OUTPUT_DIR" \
  --output-dir "$RISK_OUTPUT_DIR"

echo "[STEP] refresh main report"
"$PYTHON_BIN" -u "$ROOT/scripts/refresh_daily_monitor_report.py" \
  --monitor-dir "$MAIN_OUTPUT_DIR" \
  --risk-dir "$RISK_OUTPUT_DIR"

echo "[STEP] dual board report"
"$PYTHON_BIN" -u "$ROOT/scripts/build_dual_board_report.py" \
  --main-dir "$MAIN_OUTPUT_DIR" \
  --elastic-dir "$ELASTIC_OUTPUT_DIR" \
  --risk-dir "$RISK_OUTPUT_DIR" \
  --output-dir "$DUAL_OUTPUT_DIR"

if [[ "${RUN_SHORTLINE_REPORT,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  echo "[STEP] shortline opportunity report"
  SHORTLINE_LOG="$ROOT/outputs/ops_logs/shortline_${TARGET_DATE//-/}.log"
  if ! "$PYTHON_BIN" -u "$ROOT/scripts/run_shortline_opportunity_report.py" \
    --data-path "$DATA_PATH" \
    --premium-dir "$ROOT/data/premium_v22" \
    --config "$ROOT/configs/research_shortline_opportunity.json" \
    --output-dir "$SHORTLINE_OUTPUT_DIR" \
    --prediction-date "$TARGET_DATE" \
    --sync-limit-data \
    --bypass-system-proxy 2>&1 | tee "$SHORTLINE_LOG"; then
    if grep -q "No under-20 shortline candidates matched the filters" "$SHORTLINE_LOG"; then
      echo "[INFO] no shortline candidates for target_date=$TARGET_DATE; writing empty same-day artifact"
      "$PYTHON_BIN" - "$SHORTLINE_OUTPUT_DIR" "$TARGET_DATE" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
target_date = sys.argv[2]
date_key = target_date.replace("-", "")
reason = f"No under-20 shortline candidates matched the filters on {target_date}."
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / f"shortline_opportunity_{date_key}_cards.json").write_text(
    json.dumps(
        {
            "summary": {
                "selection_date": target_date,
                "selected_count": 0,
                "candidate_count": 0,
                "data_latest_date": target_date,
                "limit_latest_date": target_date,
                "report_type": "shortline_opportunity_under20",
                "status": "no_candidates",
                "reason": reason,
            },
            "cards": [],
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
(output_dir / f"shortline_opportunity_{date_key}_cards.csv").write_text(
    "rank,code,name,close,shortline_score,risk_level,operation_logic\n",
    encoding="utf-8",
)
(output_dir / f"shortline_opportunity_{date_key}_report.md").write_text(
    f"# 精选短线机会 {target_date}\n\n{reason}\n",
    encoding="utf-8",
)
PY
    else
      echo "[WARN] shortline opportunity report failed for target_date=$TARGET_DATE; preflight will retry later" >&2
    fi
  fi
fi

echo "[STEP] split paper monitors"
if ! bash "$ROOT/scripts/run_split_paper_monitors.sh"; then
  echo "[WARN] split paper monitors failed; evening brief artifacts are already complete, continue without blocking post-close workflow" >&2
fi

echo "[OK] cloud post-close stable workflow finished"
