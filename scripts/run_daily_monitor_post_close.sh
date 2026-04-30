#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.localhome/post_close_monitor.env"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

: "${TUSHARE_TOKEN:?TUSHARE_TOKEN is required}"
: "${TUSHARE_HTTP_URL:?TUSHARE_HTTP_URL is required}"

export HOME="$ROOT/.localhome"
export TUSHARE_BYPASS_SYSTEM_PROXY="${TUSHARE_BYPASS_SYSTEM_PROXY:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/matplotlib}"
RUN_DATA_QUALITY="${RUN_DATA_QUALITY:-1}"
RUN_DATA_UPDATE="${RUN_DATA_UPDATE:-1}"
RUN_SHORTLINE_REPORT="${RUN_SHORTLINE_REPORT:-1}"
DATA_PATH="${DATA_PATH:-data/a_share_daily_industry.csv}"

mkdir -p "$HOME" "$MPLCONFIGDIR"

to_lower() {
  printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]'
}

UNDER20_SKIP_BACKTEST_RAW="${UNDER20_SKIP_BACKTEST:-1}"
UNDER20_SKIP_BACKTEST_NORMALIZED_INPUT="$(to_lower "$UNDER20_SKIP_BACKTEST_RAW")"
case "$UNDER20_SKIP_BACKTEST_NORMALIZED_INPUT" in
  1|true|yes|y|on)
    UNDER20_SKIP_BACKTEST_NORMALIZED=1
    ;;
  0|false|no|n|off)
    UNDER20_SKIP_BACKTEST_NORMALIZED=0
    ;;
  *)
    echo "[WARN] Invalid UNDER20_SKIP_BACKTEST=${UNDER20_SKIP_BACKTEST_RAW}; fallback to 1 (skip under20 backtest)"
    UNDER20_SKIP_BACKTEST_NORMALIZED=1
    ;;
esac

UNDER20_EXTRA_ARGS=()
if [[ "$UNDER20_SKIP_BACKTEST_NORMALIZED" == "1" ]]; then
  UNDER20_EXTRA_ARGS+=(--skip-backtest)
fi

cd "$ROOT"

if [[ "$(to_lower "$RUN_DATA_UPDATE")" =~ ^(1|true|yes|y|on)$ ]]; then
  BEFORE_DATE="$(tail -n 1 "$DATA_PATH" | cut -d',' -f1 || true)"
  echo "[INFO] pre-update latest date: ${BEFORE_DATE:-unknown}"
  .venv/bin/python scripts/update_tushare_daily_dataset.py \
    --existing-path "$DATA_PATH" \
    --bypass-system-proxy
  AFTER_DATE="$(tail -n 1 "$DATA_PATH" | cut -d',' -f1 || true)"
  echo "[INFO] post-update latest date: ${AFTER_DATE:-unknown}"
fi

if [[ "$(to_lower "$RUN_DATA_QUALITY")" =~ ^(1|true|yes|y|on)$ ]]; then
  .venv/bin/python scripts/build_daily_data_quality_report.py \
    --data-path "$DATA_PATH" \
    --output-dir outputs/data_quality \
    --output-prefix daily_quality
fi

.venv/bin/python scripts/run_daily_monitor.py \
  --data-path "$DATA_PATH" \
  --slice-path data/daily_monitor_slice.csv \
  --adjust qfq \
  --research-config configs/research_v2_2_mid40_vwap_indcap_top40.json \
  --backtest-config configs/backtest_liquidity_stress_15bps.json \
  --output-dir outputs/daily_monitor_auto

.venv/bin/python scripts/run_risk_governor.py \
  --monitor-dir outputs/daily_monitor_auto \
  --risk-config configs/risk_governor.json \
  --output-dir outputs/risk_governor

.venv/bin/python scripts/refresh_daily_monitor_report.py \
  --monitor-dir outputs/daily_monitor_auto \
  --risk-dir outputs/risk_governor

.venv/bin/python scripts/run_daily_monitor.py \
  --data-path "$DATA_PATH" \
  --slice-path data/daily_monitor_slice_under20_elastic.csv \
  --adjust qfq \
  --research-config configs/research_under20_elastic_top20.json \
  --backtest-config configs/backtest_liquidity_stress_15bps.json \
  --output-dir outputs/daily_monitor_under20_elastic \
  --output-prefix under20_elastic \
  "${UNDER20_EXTRA_ARGS[@]}"

.venv/bin/python scripts/build_dual_board_report.py \
  --main-dir outputs/daily_monitor_auto \
  --elastic-dir outputs/daily_monitor_under20_elastic \
  --output-dir outputs/daily_monitor_dual \
  --output-prefix dual_monitor

if [[ "$(to_lower "$RUN_SHORTLINE_REPORT")" =~ ^(1|true|yes|y|on)$ ]]; then
  .venv/bin/python scripts/run_shortline_opportunity_report.py \
    --data-path "$DATA_PATH" \
    --premium-dir data/premium_v22 \
    --config configs/research_shortline_opportunity.json \
    --output-dir outputs/shortline_opportunities \
    --sync-limit-data \
    --bypass-system-proxy
fi

.venv/bin/python scripts/run_paper_monitor.py \
  --monitor-dir outputs/daily_monitor_auto \
  --slice-path data/daily_monitor_slice.csv \
  --adjust qfq \
  --paper-config configs/paper_trade_guardrails.json \
  --backtest-config configs/backtest_liquidity_stress_15bps.json \
  --state-path outputs/paper_monitor_auto/paper_state.json \
  --output-dir outputs/paper_monitor_auto
