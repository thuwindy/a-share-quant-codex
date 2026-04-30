#!/bin/bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.localhome/post_close_monitor.env"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

: "${TUSHARE_TOKEN:?TUSHARE_TOKEN is required}"
: "${TUSHARE_HTTP_URL:?TUSHARE_HTTP_URL is required}"

export HOME="$ROOT/.localhome"
export PYTHONPATH="${PYTHONPATH:-$ROOT/src}"
export TUSHARE_BYPASS_SYSTEM_PROXY="${TUSHARE_BYPASS_SYSTEM_PROXY:-1}"

to_lower() {
  printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]'
}

set_default_var() {
  local name="$1"
  local default="$2"
  if [[ -z "${!name+x}" ]]; then
    printf -v "$name" "%s" "$default"
  fi
}

is_true() {
  local value="${1:-0}"
  value="$(to_lower "$value")"
  [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "y" || "$value" == "on" ]]
}

WORKFLOW_MODE="$(to_lower "${WORKFLOW_MODE:-standard}")"
case "$WORKFLOW_MODE" in
  fast)
    set_default_var RUN_DATA_SYNC 1
    set_default_var RUN_DATA_QUALITY 1
    set_default_var RUN_POINT_IN_TIME_CHECK 1
    set_default_var RUN_EXECUTION_ALIGNMENT_CHECK 0
    set_default_var RUN_COST_MODEL_CHECK 0
    set_default_var RUN_METRIC_PACK 1
    set_default_var RUN_QUICK_GRID 0
    set_default_var RUN_PAPER_GATE 1
    set_default_var RUN_PREMIUM_SYNC 0
    set_default_var RUN_ANALYSIS_ALIGN 0
    set_default_var RUN_MAIN_MONITOR 1
    set_default_var RUN_ELASTIC_MONITOR 0
    set_default_var RUN_RISK_GOVERNOR 1
    set_default_var RUN_DAILY_RISK_REFRESH 1
    set_default_var RUN_EVAL_ABLATION 0
    set_default_var RUN_DUAL_REPORT 0
    set_default_var RUN_PAPER_LIVE 1
    set_default_var RUN_FAILURE_NOTIFY 1
    ;;
  standard)
    set_default_var RUN_DATA_SYNC 1
    set_default_var RUN_DATA_QUALITY 1
    set_default_var RUN_POINT_IN_TIME_CHECK 1
    set_default_var RUN_EXECUTION_ALIGNMENT_CHECK 0
    set_default_var RUN_COST_MODEL_CHECK 0
    set_default_var RUN_METRIC_PACK 1
    set_default_var RUN_QUICK_GRID 0
    set_default_var RUN_PAPER_GATE 1
    set_default_var RUN_PREMIUM_SYNC 0
    set_default_var RUN_ANALYSIS_ALIGN 0
    set_default_var RUN_MAIN_MONITOR 1
    set_default_var RUN_ELASTIC_MONITOR 1
    set_default_var RUN_RISK_GOVERNOR 1
    set_default_var RUN_DAILY_RISK_REFRESH 1
    set_default_var RUN_EVAL_ABLATION 0
    set_default_var RUN_DUAL_REPORT 1
    set_default_var RUN_PAPER_LIVE 1
    set_default_var RUN_FAILURE_NOTIFY 1
    ;;
  full)
    set_default_var RUN_DATA_SYNC 1
    set_default_var RUN_DATA_QUALITY 1
    set_default_var RUN_POINT_IN_TIME_CHECK 1
    set_default_var RUN_EXECUTION_ALIGNMENT_CHECK 0
    set_default_var RUN_COST_MODEL_CHECK 1
    set_default_var RUN_METRIC_PACK 1
    set_default_var RUN_QUICK_GRID 1
    set_default_var RUN_PAPER_GATE 1
    set_default_var RUN_PREMIUM_SYNC 1
    set_default_var RUN_ANALYSIS_ALIGN 1
    set_default_var RUN_MAIN_MONITOR 1
    set_default_var RUN_ELASTIC_MONITOR 1
    set_default_var RUN_RISK_GOVERNOR 1
    set_default_var RUN_DAILY_RISK_REFRESH 1
    set_default_var RUN_EVAL_ABLATION 1
    set_default_var RUN_DUAL_REPORT 1
    set_default_var RUN_PAPER_LIVE 1
    set_default_var RUN_FAILURE_NOTIFY 1
    ;;
  *)
    echo "[WARN] unknown WORKFLOW_MODE=$WORKFLOW_MODE, fallback to standard"
    WORKFLOW_MODE="standard"
    set_default_var RUN_DATA_SYNC 1
    set_default_var RUN_DATA_QUALITY 1
    set_default_var RUN_POINT_IN_TIME_CHECK 1
    set_default_var RUN_EXECUTION_ALIGNMENT_CHECK 0
    set_default_var RUN_COST_MODEL_CHECK 0
    set_default_var RUN_METRIC_PACK 1
    set_default_var RUN_QUICK_GRID 0
    set_default_var RUN_PAPER_GATE 1
    set_default_var RUN_PREMIUM_SYNC 0
    set_default_var RUN_ANALYSIS_ALIGN 0
    set_default_var RUN_MAIN_MONITOR 1
    set_default_var RUN_ELASTIC_MONITOR 1
    set_default_var RUN_RISK_GOVERNOR 1
    set_default_var RUN_DAILY_RISK_REFRESH 1
    set_default_var RUN_EVAL_ABLATION 0
    set_default_var RUN_DUAL_REPORT 1
    set_default_var RUN_PAPER_LIVE 1
    set_default_var RUN_FAILURE_NOTIFY 1
    ;;
esac

set_default_var UNDER20_SKIP_BACKTEST 1
set_default_var PREMIUM_OVERLAP_DAYS 5
set_default_var PREMIUM_FULL_REFRESH 0
set_default_var PREMIUM_TASK_MAX_RETRIES 2
set_default_var PREMIUM_TASK_RETRY_SLEEP_SECONDS 5
set_default_var PREMIUM_DEGRADE_REPORT_RC_ON_ERROR 1
set_default_var PIT_DATA_PATH data/daily_monitor_slice_fundamental.csv
set_default_var RESEARCH_CONFIG_MAIN configs/research_production_default.json
set_default_var RESEARCH_CONFIG_ELASTIC configs/research_under20_elastic_top20.json
set_default_var BACKTEST_CONFIG configs/backtest_liquidity_stress_15bps.json
set_default_var QUICK_GRID_DATA_PATH data/a_share_daily_industry.csv
set_default_var QUICK_GRID_RESEARCH_CONFIG "$RESEARCH_CONFIG_MAIN"
set_default_var QUICK_GRID_BACKTEST_CONFIG "$BACKTEST_CONFIG"
set_default_var QUICK_GRID_OUTPUT_DIR outputs/quick_grid
set_default_var QUICK_GRID_OUTPUT_PREFIX quick_grid
set_default_var QUICK_GRID_TOP_N_GRID 20,30,40
set_default_var QUICK_GRID_WEIGHT_CHANGE_GRID 0.03,0.05
set_default_var QUICK_GRID_RANK_CHANGE_GRID 8,15,20
set_default_var QUICK_GRID_SLEEVE_GRID ""
set_default_var EXEC_ALIGNMENT_DATA_PATH data/daily_monitor_slice_fundamental.csv
set_default_var METRIC_PACK_MONITOR_DIR outputs/daily_monitor_auto
set_default_var DAILY_RISK_REFRESH_MONITOR_DIR outputs/daily_monitor_auto
set_default_var DAILY_RISK_REFRESH_RISK_DIR outputs/risk_governor
set_default_var PAPER_GATE_RISK_DIR outputs/risk_governor
set_default_var PAPER_GATE_OUTPUT_DIR outputs/paper_monitor_auto
set_default_var PAPER_GATE_FLAG_PATH outputs/paper_monitor_auto/paper_gate_allow.flag
set_default_var FAILURE_NOTIFY_OUTPUT_DIR outputs/ops_logs/alerts
set_default_var FAILURE_NOTIFY_WEBHOOK_URL ""

cd "$ROOT"
mkdir -p outputs/ops_logs
LOG_PATH="outputs/ops_logs/post_close_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[INFO] workflow start at $(date '+%F %T %Z')"
echo "[INFO] log: $LOG_PATH"
echo "[INFO] workflow mode: $WORKFLOW_MODE"
echo "[INFO] switches: data_sync=$RUN_DATA_SYNC data_quality=$RUN_DATA_QUALITY pit=$RUN_POINT_IN_TIME_CHECK execution_alignment=$RUN_EXECUTION_ALIGNMENT_CHECK cost_model=$RUN_COST_MODEL_CHECK metric_pack=$RUN_METRIC_PACK quick_grid=$RUN_QUICK_GRID premium_sync=$RUN_PREMIUM_SYNC analysis_align=$RUN_ANALYSIS_ALIGN main_monitor=$RUN_MAIN_MONITOR elastic_monitor=$RUN_ELASTIC_MONITOR risk_governor=$RUN_RISK_GOVERNOR daily_risk_refresh=$RUN_DAILY_RISK_REFRESH paper_gate=$RUN_PAPER_GATE eval_ablation=$RUN_EVAL_ABLATION dual_report=$RUN_DUAL_REPORT paper_live=$RUN_PAPER_LIVE failure_notify=$RUN_FAILURE_NOTIFY"
echo "[INFO] configs: research_main=$RESEARCH_CONFIG_MAIN research_elastic=$RESEARCH_CONFIG_ELASTIC backtest=$BACKTEST_CONFIG"

LAST_TASK_ID="bootstrap"
WORKFLOW_NAME="post_close_daily_agents"
on_error() {
  local exit_code=$?
  local failed_line="${BASH_LINENO[0]:-0}"
  trap - ERR
  set +e

  mkdir -p outputs/ops_logs
  printf '{"status":"error","failed_at":"%s","task_id":"%s","exit_code":%s,"failed_line":%s,"log_path":"%s"}\n' \
    "$(date '+%F %T %Z')" "$LAST_TASK_ID" "$exit_code" "$failed_line" "$LOG_PATH" \
    > outputs/ops_logs/post_close_last_error.json

  if is_true "$RUN_FAILURE_NOTIFY"; then
    .venv/bin/python scripts/notify_workflow_failure.py \
      --workflow-name "$WORKFLOW_NAME" \
      --task-id "$LAST_TASK_ID" \
      --exit-code "$exit_code" \
      --log-path "$LOG_PATH" \
      --error-message "task_failed_at_line_${failed_line}" \
      --output-dir "$FAILURE_NOTIFY_OUTPUT_DIR" \
      --webhook-url "$FAILURE_NOTIFY_WEBHOOK_URL" \
      || true
  fi
  echo "[ERROR] workflow failed at task=$LAST_TASK_ID line=$failed_line exit_code=$exit_code"
  exit "$exit_code"
}
trap on_error ERR

run_task() {
  local task_id="$1"
  shift
  LAST_TASK_ID="$task_id"
  echo "[TASK:$task_id] START"
  "$@"
  echo "[TASK:$task_id] DONE"
}

if is_true "$RUN_DATA_SYNC"; then
  run_task "T1_data_sync" .venv/bin/python scripts/update_tushare_daily_dataset.py \
    --existing-path data/a_share_daily_industry.csv \
    --bypass-system-proxy
fi

if is_true "$RUN_DATA_QUALITY"; then
  run_task "T1b_data_quality" .venv/bin/python scripts/build_daily_data_quality_report.py \
    --data-path data/a_share_daily_industry.csv \
    --output-dir outputs/data_quality \
    --output-prefix daily_quality
fi

if is_true "$RUN_POINT_IN_TIME_CHECK"; then
  PIT_SOURCE="$PIT_DATA_PATH"
  if [[ ! -f "$PIT_SOURCE" ]]; then
    PIT_SOURCE="data/a_share_daily_industry.csv"
  fi
  run_task "T1c_point_in_time" .venv/bin/python scripts/check_point_in_time_alignment.py \
    --data-path "$PIT_SOURCE" \
    --output-dir outputs/data_quality \
    --output-prefix t13_point_in_time
fi

if is_true "$RUN_EXECUTION_ALIGNMENT_CHECK"; then
  EXEC_ALIGNMENT_SOURCE="$EXEC_ALIGNMENT_DATA_PATH"
  if [[ ! -f "$EXEC_ALIGNMENT_SOURCE" ]]; then
    EXEC_ALIGNMENT_SOURCE="data/daily_monitor_slice.csv"
  fi
  if [[ ! -f "$EXEC_ALIGNMENT_SOURCE" ]]; then
    EXEC_ALIGNMENT_SOURCE="data/a_share_daily_industry.csv"
  fi
  run_task "T21_execution_alignment" .venv/bin/python scripts/check_execution_alignment_variants.py \
    --data-path "$EXEC_ALIGNMENT_SOURCE" \
    --research-config "$RESEARCH_CONFIG_MAIN" \
    --adjust qfq \
    --output-dir outputs/data_quality \
    --output-prefix t21_execution_alignment \
    --execution-lag 1
fi

if is_true "$RUN_COST_MODEL_CHECK"; then
  run_task "T22_cost_model" .venv/bin/python scripts/check_cost_model_consistency.py \
    --backtest-config "$BACKTEST_CONFIG" \
    --output-dir outputs/data_quality \
    --output-prefix t22_cost_model_consistency
fi

if is_true "$RUN_PREMIUM_SYNC"; then
  PREMIUM_ARGS=(--overlap-days "$PREMIUM_OVERLAP_DAYS")
  if is_true "$PREMIUM_FULL_REFRESH"; then
    PREMIUM_ARGS+=(--full-refresh)
  fi
  PREMIUM_ARGS+=(--task-max-retries "$PREMIUM_TASK_MAX_RETRIES")
  PREMIUM_ARGS+=(--task-retry-sleep-seconds "$PREMIUM_TASK_RETRY_SLEEP_SECONDS")
  if is_true "$PREMIUM_DEGRADE_REPORT_RC_ON_ERROR"; then
    PREMIUM_ARGS+=(--degrade-report-rc-on-error)
  else
    PREMIUM_ARGS+=(--no-degrade-report-rc-on-error)
  fi
  run_task "T2_premium_sync" .venv/bin/python scripts/sync_tushare_premium_tables.py \
    --daily-path data/a_share_daily_industry.csv \
    --output-dir data/premium_v22 \
    --bypass-system-proxy \
    "${PREMIUM_ARGS[@]}"
fi

if is_true "$RUN_ANALYSIS_ALIGN"; then
  run_task "T3_analysis_align" .venv/bin/python scripts/build_aligned_feature_panels.py \
    --daily-path data/a_share_daily_industry.csv \
    --flow-path data/premium_v22/tushare_flow_daily.csv \
    --chip-path data/premium_v22/tushare_chip_daily.csv \
    --report-rc-path data/premium_v22/tushare_report_rc_events.csv \
    --hk-hold-path data/premium_v22/tushare_hk_hold_daily.csv \
    --limit-sentiment-path data/premium_v22/tushare_limit_sentiment_daily.csv \
    --output-path data/a_share_daily_premium.csv
fi

if is_true "$RUN_MAIN_MONITOR"; then
  run_task "T4_main_monitor" .venv/bin/python scripts/run_daily_monitor.py \
    --data-path data/a_share_daily_industry.csv \
    --slice-path data/daily_monitor_slice.csv \
    --adjust qfq \
    --research-config "$RESEARCH_CONFIG_MAIN" \
    --backtest-config "$BACKTEST_CONFIG" \
    --output-dir outputs/daily_monitor_auto \
    --skip-update
fi

if is_true "$RUN_METRIC_PACK"; then
  run_task "T30_metric_pack" .venv/bin/python scripts/build_standard_metric_pack.py \
    --monitor-dir "$METRIC_PACK_MONITOR_DIR"
fi

if is_true "$RUN_QUICK_GRID"; then
  QUICK_GRID_ARGS=()
  if [[ -n "${QUICK_GRID_SLEEVE_GRID}" ]]; then
    QUICK_GRID_ARGS+=(--sleeve-grid "$QUICK_GRID_SLEEVE_GRID")
  fi
  run_task "T23_quick_grid" .venv/bin/python scripts/run_quick_topn_notrade_grid.py \
    --data-path "$QUICK_GRID_DATA_PATH" \
    --research-config "$QUICK_GRID_RESEARCH_CONFIG" \
    --backtest-config "$QUICK_GRID_BACKTEST_CONFIG" \
    --adjust qfq \
    --output-dir "$QUICK_GRID_OUTPUT_DIR" \
    --output-prefix "$QUICK_GRID_OUTPUT_PREFIX" \
    --top-n-grid "$QUICK_GRID_TOP_N_GRID" \
    --weight-change-grid "$QUICK_GRID_WEIGHT_CHANGE_GRID" \
    --rank-change-grid "$QUICK_GRID_RANK_CHANGE_GRID" \
    "${QUICK_GRID_ARGS[@]}"
fi

if is_true "$RUN_ELASTIC_MONITOR"; then
  UNDER20_ARGS=()
  if is_true "$UNDER20_SKIP_BACKTEST"; then
    UNDER20_ARGS+=(--skip-backtest)
  fi
  run_task "T5_under20_elastic" .venv/bin/python scripts/run_daily_monitor.py \
    --data-path data/a_share_daily_industry.csv \
    --slice-path data/daily_monitor_slice_under20_elastic.csv \
    --adjust qfq \
    --research-config "$RESEARCH_CONFIG_ELASTIC" \
    --backtest-config "$BACKTEST_CONFIG" \
    --output-dir outputs/daily_monitor_under20_elastic \
    --output-prefix under20_elastic \
    --skip-update \
    "${UNDER20_ARGS[@]}"
fi

if is_true "$RUN_RISK_GOVERNOR"; then
  run_task "T6_risk_governor" .venv/bin/python scripts/run_risk_governor.py \
    --monitor-dir outputs/daily_monitor_auto \
    --risk-config configs/risk_governor.json \
    --output-dir outputs/risk_governor
fi

if is_true "$RUN_DAILY_RISK_REFRESH" && is_true "$RUN_RISK_GOVERNOR" && is_true "$RUN_MAIN_MONITOR"; then
  run_task "T33_daily_risk_refresh" .venv/bin/python scripts/refresh_daily_monitor_report.py \
    --monitor-dir "$DAILY_RISK_REFRESH_MONITOR_DIR" \
    --risk-dir "$DAILY_RISK_REFRESH_RISK_DIR"
fi

if is_true "$RUN_EVAL_ABLATION"; then
  run_task "T7_eval_ablation" .venv/bin/python scripts/run_ablation_suite.py \
    --data-path data/a_share_daily_industry.csv \
    --research-config configs/recommended_default_config.json \
    --backtest-config configs/backtest_liquidity.json \
    --adjust qfq \
    --output-dir outputs/ablation_suite
fi

if is_true "$RUN_DUAL_REPORT"; then
  run_task "T8_dual_report" .venv/bin/python scripts/build_dual_board_report.py \
    --main-dir outputs/daily_monitor_auto \
    --elastic-dir outputs/daily_monitor_under20_elastic \
    --output-dir outputs/daily_monitor_dual \
    --output-prefix dual_monitor
fi

PAPER_GATE_ALLOW="1"
if is_true "$RUN_PAPER_GATE"; then
  run_task "T42_paper_gate" .venv/bin/python scripts/check_paper_trade_gate.py \
    --risk-dir "$PAPER_GATE_RISK_DIR" \
    --output-dir "$PAPER_GATE_OUTPUT_DIR" \
    --allow-flag-path "$PAPER_GATE_FLAG_PATH" \
    --block-statuses BLOCK
  PAPER_GATE_ALLOW="$(cat "$PAPER_GATE_FLAG_PATH" 2>/dev/null || echo "1")"
  if [[ "$PAPER_GATE_ALLOW" != "1" ]]; then
    echo "[TASK:T42_paper_gate] BLOCK status detected, paper live step will be skipped."
  fi
fi

if is_true "$RUN_PAPER_LIVE"; then
  if [[ "$PAPER_GATE_ALLOW" == "1" ]]; then
    run_task "T9_paper_live" .venv/bin/python scripts/run_paper_monitor.py \
      --monitor-dir outputs/daily_monitor_auto \
      --slice-path data/daily_monitor_slice.csv \
      --adjust qfq \
      --paper-config configs/paper_trade_guardrails.json \
      --backtest-config "$BACKTEST_CONFIG" \
      --state-path outputs/paper_monitor_auto/paper_state.json \
      --output-dir outputs/paper_monitor_auto
  else
    echo "[TASK:T9_paper_live] SKIPPED by paper gate (allow_flag=$PAPER_GATE_ALLOW)."
  fi
fi

mkdir -p outputs/ops_logs
rm -f outputs/ops_logs/post_close_last_error.json
printf '{"status":"ok","finished_at":"%s","log_path":"%s"}\n' \
  "$(date '+%F %T %Z')" "$LOG_PATH" \
  > outputs/ops_logs/post_close_last_success.json

echo "[INFO] workflow completed at $(date '+%F %T %Z')"
