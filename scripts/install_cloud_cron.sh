#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$ROOT"
RUN_HOUR=16
RUN_MINUTE=30
ROTATE_HOUR=18
ROTATE_MINUTE=30
RUN_WEEKDAYS="1-5"
WORKFLOW_MODE="standard"
RUN_PREMIUM_SYNC=1
PREMIUM_SYNC_HOUR=18
PREMIUM_SYNC_MINUTE=10
KEEP_PLAIN_DAYS=7
KEEP_GZIP_DAYS=45
CRON_TZ_NAME="Asia/Shanghai"
BENCH_HOUR=1
BENCH_MINUTE=30
STYLE_HOUR=3
STYLE_MINUTE=30
RUN_BENCHMARK=1
RUN_STYLE_COMPARE=0
PUSH_HOUR=20
PUSH_MINUTE=0
PREFLIGHT_HOUR=19
PREFLIGHT_MINUTE=45
RUN_EVENING_PREFLIGHT=1
RUN_PUSHPLUS_SHORTLINE=1
MORNING_HOUR=8
MORNING_MINUTE=58
RUN_MORNING_BRIEF=1
PREFETCH_HOUR=8
PREFETCH_MINUTE=40
RUN_MORNING_PREFETCH=1
WEEKLY_RESEARCH_HOUR=21
WEEKLY_RESEARCH_MINUTE=0
WEEKLY_RESEARCH_DAY=5
RUN_WEEKLY_RESEARCH_BRIEF=1
OBSERVATION_CHECK_HOUR=20
OBSERVATION_CHECK_MINUTE=20
RUN_OBSERVATION_DAILY_CHECK=1
DRY_RUN=0
UNINSTALL=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/install_cloud_cron.sh [options]

Options:
  --repo-root PATH              Repo root path (default: script parent)
  --run-hour N                  Post-close run hour (default: 16)
  --run-minute N                Post-close run minute (default: 30)
  --rotate-hour N               Log rotation hour (default: 18)
  --rotate-minute N             Log rotation minute (default: 30)
  --run-weekdays SPEC           Cron weekday spec for strategy run (default: 1-5)
  --workflow-mode MODE          fast|standard|full (default: standard)
  --run-premium-sync 0|1        Whether to include premium sync in cron run (default: 0)
  --premium-sync-hour N         Premium Tushare sync hour (default: 18)
  --premium-sync-minute N       Premium Tushare sync minute (default: 10)
  --keep-plain-days N           Keep uncompressed post_close logs for N days (default: 7)
  --keep-gzip-days N            Keep compressed post_close logs for N days (default: 45)
  --cron-tz TZ                  Cron timezone (default: Asia/Shanghai)
  --bench-hour N                Locked 2025 benchmark hour (default: 1)
  --bench-minute N              Locked 2025 benchmark minute (default: 30)
  --style-hour N                Legacy extra cron hour (default: 3)
  --style-minute N              Legacy extra cron minute (default: 30)
  --run-benchmark 0|1           Install locked benchmark cron (default: 1)
  --run-style-compare 0|1       Install legacy extra cron (default: 0)
  --push-hour N                 PushPlus evening brief push hour (default: 20)
  --push-minute N               PushPlus evening brief push minute (default: 0)
  --preflight-hour N            Evening brief preflight hour (default: 19)
  --preflight-minute N          Evening brief preflight minute (default: 45)
  --run-evening-preflight 0|1   Install evening preflight cron (default: 1)
  --run-pushplus-shortline 0|1  Install daily PushPlus shortline push cron (default: 1)
  --prefetch-hour N             Morning brief cache prefetch hour (default: 8)
  --prefetch-minute N           Morning brief cache prefetch minute (default: 40)
  --run-morning-prefetch 0|1    Install morning brief cache prefetch cron (default: 1)
  --morning-hour N              Morning brief push hour (default: 8)
  --morning-minute N            Morning brief push minute (default: 58)
  --run-morning-brief 0|1       Install daily 08:58 morning brief cron (default: 1)
  --weekly-research-hour N      Weekly observation review hour (default: 21)
  --weekly-research-minute N    Weekly observation review minute (default: 0)
  --run-weekly-research-brief 0|1 Install Friday weekly observation review cron (default: 1)
  --observation-check-hour N    Daily stable observation check hour (default: 20)
  --observation-check-minute N  Daily stable observation check minute (default: 20)
  --run-observation-daily-check 0|1 Install weekday stable observation check cron (default: 1)
  --dry-run                     Print resulting crontab without applying
  --uninstall                   Remove managed cron block and exit
  -h, --help                    Show help

Managed crontab block markers:
  # >>> a_share_quant_codex managed block >>>
  # <<< a_share_quant_codex managed block <<<
EOF
}

is_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"; shift 2 ;;
    --run-hour)
      RUN_HOUR="$2"; shift 2 ;;
    --run-minute)
      RUN_MINUTE="$2"; shift 2 ;;
    --rotate-hour)
      ROTATE_HOUR="$2"; shift 2 ;;
    --rotate-minute)
      ROTATE_MINUTE="$2"; shift 2 ;;
    --run-weekdays)
      RUN_WEEKDAYS="$2"; shift 2 ;;
    --workflow-mode)
      WORKFLOW_MODE="$2"; shift 2 ;;
    --run-premium-sync)
      RUN_PREMIUM_SYNC="$2"; shift 2 ;;
    --premium-sync-hour)
      PREMIUM_SYNC_HOUR="$2"; shift 2 ;;
    --premium-sync-minute)
      PREMIUM_SYNC_MINUTE="$2"; shift 2 ;;
    --keep-plain-days)
      KEEP_PLAIN_DAYS="$2"; shift 2 ;;
    --keep-gzip-days)
      KEEP_GZIP_DAYS="$2"; shift 2 ;;
    --cron-tz)
      CRON_TZ_NAME="$2"; shift 2 ;;
    --bench-hour)
      BENCH_HOUR="$2"; shift 2 ;;
    --bench-minute)
      BENCH_MINUTE="$2"; shift 2 ;;
    --style-hour)
      STYLE_HOUR="$2"; shift 2 ;;
    --style-minute)
      STYLE_MINUTE="$2"; shift 2 ;;
    --run-benchmark)
      RUN_BENCHMARK="$2"; shift 2 ;;
    --run-style-compare)
      RUN_STYLE_COMPARE="$2"; shift 2 ;;
    --prefetch-hour)
      PREFETCH_HOUR="$2"; shift 2 ;;
    --prefetch-minute)
      PREFETCH_MINUTE="$2"; shift 2 ;;
    --run-morning-prefetch)
      RUN_MORNING_PREFETCH="$2"; shift 2 ;;
    --push-hour)
      PUSH_HOUR="$2"; shift 2 ;;
    --push-minute)
      PUSH_MINUTE="$2"; shift 2 ;;
    --preflight-hour)
      PREFLIGHT_HOUR="$2"; shift 2 ;;
    --preflight-minute)
      PREFLIGHT_MINUTE="$2"; shift 2 ;;
    --run-evening-preflight)
      RUN_EVENING_PREFLIGHT="$2"; shift 2 ;;
    --run-pushplus-shortline)
      RUN_PUSHPLUS_SHORTLINE="$2"; shift 2 ;;
    --morning-hour)
      MORNING_HOUR="$2"; shift 2 ;;
    --morning-minute)
      MORNING_MINUTE="$2"; shift 2 ;;
    --run-morning-brief)
      RUN_MORNING_BRIEF="$2"; shift 2 ;;
    --weekly-research-hour)
      WEEKLY_RESEARCH_HOUR="$2"; shift 2 ;;
    --weekly-research-minute)
      WEEKLY_RESEARCH_MINUTE="$2"; shift 2 ;;
    --run-weekly-research-brief)
      RUN_WEEKLY_RESEARCH_BRIEF="$2"; shift 2 ;;
    --observation-check-hour)
      OBSERVATION_CHECK_HOUR="$2"; shift 2 ;;
    --observation-check-minute)
      OBSERVATION_CHECK_MINUTE="$2"; shift 2 ;;
    --run-observation-daily-check)
      RUN_OBSERVATION_DAILY_CHECK="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --uninstall)
      UNINSTALL=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[ERROR] repo root does not exist: $REPO_ROOT" >&2
  exit 2
fi

if [[ ! -x "$REPO_ROOT/scripts/run_agent_workflow_template.sh" ]]; then
  echo "[WARN] $REPO_ROOT/scripts/run_agent_workflow_template.sh is not executable; trying /bin/bash invocation."
fi

for val in "$RUN_HOUR" "$RUN_MINUTE" "$PREMIUM_SYNC_HOUR" "$PREMIUM_SYNC_MINUTE" "$ROTATE_HOUR" "$ROTATE_MINUTE" "$BENCH_HOUR" "$BENCH_MINUTE" "$STYLE_HOUR" "$STYLE_MINUTE" "$PUSH_HOUR" "$PUSH_MINUTE" "$PREFLIGHT_HOUR" "$PREFLIGHT_MINUTE" "$PREFETCH_HOUR" "$PREFETCH_MINUTE" "$MORNING_HOUR" "$MORNING_MINUTE" "$WEEKLY_RESEARCH_HOUR" "$WEEKLY_RESEARCH_MINUTE" "$WEEKLY_RESEARCH_DAY" "$OBSERVATION_CHECK_HOUR" "$OBSERVATION_CHECK_MINUTE" "$RUN_PREMIUM_SYNC" "$KEEP_PLAIN_DAYS" "$KEEP_GZIP_DAYS" "$RUN_BENCHMARK" "$RUN_STYLE_COMPARE" "$RUN_PUSHPLUS_SHORTLINE" "$RUN_MORNING_PREFETCH" "$RUN_MORNING_BRIEF" "$RUN_EVENING_PREFLIGHT" "$RUN_WEEKLY_RESEARCH_BRIEF" "$RUN_OBSERVATION_DAILY_CHECK"; do
  if ! is_int "$val"; then
    echo "[ERROR] integer expected but got: $val" >&2
    exit 2
  fi
done

if [[ "$WORKFLOW_MODE" != "fast" && "$WORKFLOW_MODE" != "standard" && "$WORKFLOW_MODE" != "full" ]]; then
  echo "[ERROR] WORKFLOW_MODE must be fast|standard|full, got: $WORKFLOW_MODE" >&2
  exit 2
fi

if [[ "$RUN_PREMIUM_SYNC" != "0" && "$RUN_PREMIUM_SYNC" != "1" ]]; then
  echo "[ERROR] RUN_PREMIUM_SYNC must be 0 or 1, got: $RUN_PREMIUM_SYNC" >&2
  exit 2
fi
if [[ "$RUN_BENCHMARK" != "0" && "$RUN_BENCHMARK" != "1" ]]; then
  echo "[ERROR] RUN_BENCHMARK must be 0 or 1, got: $RUN_BENCHMARK" >&2
  exit 2
fi
if [[ "$RUN_STYLE_COMPARE" != "0" && "$RUN_STYLE_COMPARE" != "1" ]]; then
  echo "[ERROR] RUN_STYLE_COMPARE must be 0 or 1, got: $RUN_STYLE_COMPARE" >&2
  exit 2
fi
if [[ "$RUN_PUSHPLUS_SHORTLINE" != "0" && "$RUN_PUSHPLUS_SHORTLINE" != "1" ]]; then
  echo "[ERROR] RUN_PUSHPLUS_SHORTLINE must be 0 or 1, got: $RUN_PUSHPLUS_SHORTLINE" >&2
  exit 2
fi
if [[ "$RUN_EVENING_PREFLIGHT" != "0" && "$RUN_EVENING_PREFLIGHT" != "1" ]]; then
  echo "[ERROR] RUN_EVENING_PREFLIGHT must be 0 or 1, got: $RUN_EVENING_PREFLIGHT" >&2
  exit 2
fi
if [[ "$RUN_MORNING_BRIEF" != "0" && "$RUN_MORNING_BRIEF" != "1" ]]; then
  echo "[ERROR] RUN_MORNING_BRIEF must be 0 or 1, got: $RUN_MORNING_BRIEF" >&2
  exit 2
fi
if [[ "$RUN_MORNING_PREFETCH" != "0" && "$RUN_MORNING_PREFETCH" != "1" ]]; then
  echo "[ERROR] RUN_MORNING_PREFETCH must be 0 or 1, got: $RUN_MORNING_PREFETCH" >&2
  exit 2
fi
if [[ "$RUN_WEEKLY_RESEARCH_BRIEF" != "0" && "$RUN_WEEKLY_RESEARCH_BRIEF" != "1" ]]; then
  echo "[ERROR] RUN_WEEKLY_RESEARCH_BRIEF must be 0 or 1, got: $RUN_WEEKLY_RESEARCH_BRIEF" >&2
  exit 2
fi
if [[ "$RUN_OBSERVATION_DAILY_CHECK" != "0" && "$RUN_OBSERVATION_DAILY_CHECK" != "1" ]]; then
  echo "[ERROR] RUN_OBSERVATION_DAILY_CHECK must be 0 or 1, got: $RUN_OBSERVATION_DAILY_CHECK" >&2
  exit 2
fi

mkdir -p "$REPO_ROOT/outputs/ops_logs"

BEGIN_MARKER="# >>> a_share_quant_codex managed block >>>"
END_MARKER="# <<< a_share_quant_codex managed block <<<"

current_crontab="$(crontab -l 2>/dev/null || true)"
clean_crontab="$(printf '%s\n' "$current_crontab" | awk -v b="$BEGIN_MARKER" -v e="$END_MARKER" '
  $0==b {skip=1; next}
  $0==e {skip=0; next}
  !skip {print}
')"

if [[ "$UNINSTALL" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] would uninstall managed cron block:"
    printf '%s\n' "$clean_crontab"
  else
    printf '%s\n' "$clean_crontab" | crontab -
    echo "[OK] removed managed cron block."
  fi
  exit 0
fi

POST_CLOSE_CMD="cd $REPO_ROOT && /bin/bash scripts/run_cloud_post_close_stable.sh >> $REPO_ROOT/outputs/ops_logs/cron_post_close.log 2>&1"
PREMIUM_SYNC_CMD="cd $REPO_ROOT && /bin/bash -lc 'set -a && source .localhome/post_close_monitor.env && set +a && .venv/bin/python scripts/sync_tushare_premium_tables.py --daily-path data/a_share_daily_industry.csv --output-dir data/premium_v22 --overlap-days 3 --bypass-system-proxy' >> $REPO_ROOT/outputs/ops_logs/cron_premium_sync.log 2>&1"
BENCH_CMD="cd $REPO_ROOT && /bin/bash scripts/run_cloud_research_pack.sh >> $REPO_ROOT/outputs/ops_logs/cron_locked_2025_benchmark.log 2>&1"
STYLE_CMD="cd $REPO_ROOT && /bin/true"
PUSH_CMD="cd $REPO_ROOT && /bin/bash -lc 'set -a && source .localhome/post_close_monitor.env && set +a && .venv/bin/python scripts/send_pushplus_evening_brief.py --template html' >> $REPO_ROOT/outputs/ops_logs/cron_pushplus_shortline.log 2>&1"
PREFLIGHT_CMD="cd $REPO_ROOT && /bin/bash -lc 'set -a && source .localhome/post_close_monitor.env && set +a && .venv/bin/python scripts/preflight_evening_brief.py' >> $REPO_ROOT/outputs/ops_logs/cron_evening_preflight.log 2>&1"
MORNING_PREFETCH_CMD="cd $REPO_ROOT && /bin/bash -lc 'set -a && source .localhome/post_close_monitor.env && set +a && .venv/bin/python scripts/send_pushplus_morning_brief.py --template html --cache-only' >> $REPO_ROOT/outputs/ops_logs/cron_morning_brief.log 2>&1"
MORNING_CMD="cd $REPO_ROOT && /bin/bash -lc 'set -a && source .localhome/post_close_monitor.env && set +a && .venv/bin/python scripts/send_pushplus_morning_brief.py --template html --use-latest-cache' >> $REPO_ROOT/outputs/ops_logs/cron_morning_brief.log 2>&1"
WEEKLY_RESEARCH_CMD="cd $REPO_ROOT && /bin/bash -lc 'set -a && source .localhome/post_close_monitor.env && set +a && .venv/bin/python scripts/send_pushplus_research_weekly.py --template html' >> $REPO_ROOT/outputs/ops_logs/cron_research_weekly.log 2>&1"
OBSERVATION_DAILY_CMD="cd $REPO_ROOT && /bin/bash -lc 'set -a && source .localhome/post_close_monitor.env && set +a && STAMP=\$(date +\\%Y\\%m\\%d) && .venv/bin/python scripts/run_stable_observation_daily_check.py --data-path data/a_share_daily_industry.csv --research-config configs/research_production_default.json --output-prefix candidate_pool_quality_dashboard --date-key \${STAMP}' >> $REPO_ROOT/outputs/ops_logs/cron_observation_daily.log 2>&1"
ROTATE_CMD="cd $REPO_ROOT && KEEP_PLAIN_DAYS=$KEEP_PLAIN_DAYS KEEP_GZIP_DAYS=$KEEP_GZIP_DAYS /bin/bash scripts/rotate_ops_logs.sh >> $REPO_ROOT/outputs/ops_logs/cron_rotate.log 2>&1"

managed_lines="$(cat <<EOF
$BEGIN_MARKER
CRON_TZ=$CRON_TZ_NAME
$RUN_MINUTE $RUN_HOUR * * $RUN_WEEKDAYS $POST_CLOSE_CMD
EOF
)"
if [[ "$RUN_BENCHMARK" == "1" ]]; then
  managed_lines="${managed_lines}"$'\n'"$BENCH_MINUTE $BENCH_HOUR * * $RUN_WEEKDAYS $BENCH_CMD"
fi
if [[ "$RUN_PREMIUM_SYNC" == "1" ]]; then
  managed_lines="${managed_lines}"$'\n'"$PREMIUM_SYNC_MINUTE $PREMIUM_SYNC_HOUR * * $RUN_WEEKDAYS $PREMIUM_SYNC_CMD"
fi
if [[ "$RUN_STYLE_COMPARE" == "1" ]]; then
  managed_lines="${managed_lines}"$'\n'"$STYLE_MINUTE $STYLE_HOUR * * $RUN_WEEKDAYS $STYLE_CMD"
fi
if [[ "$RUN_PUSHPLUS_SHORTLINE" == "1" ]]; then
  managed_lines="${managed_lines}"$'\n'"$PUSH_MINUTE $PUSH_HOUR * * $RUN_WEEKDAYS $PUSH_CMD"
fi
if [[ "$RUN_EVENING_PREFLIGHT" == "1" ]]; then
  managed_lines="${managed_lines}"$'\n'"$PREFLIGHT_MINUTE $PREFLIGHT_HOUR * * $RUN_WEEKDAYS $PREFLIGHT_CMD"
fi
if [[ "$RUN_MORNING_PREFETCH" == "1" ]]; then
  managed_lines="${managed_lines}"$'\n'"$PREFETCH_MINUTE $PREFETCH_HOUR * * $RUN_WEEKDAYS $MORNING_PREFETCH_CMD"
fi
if [[ "$RUN_MORNING_BRIEF" == "1" ]]; then
  managed_lines="${managed_lines}"$'\n'"$MORNING_MINUTE $MORNING_HOUR * * $RUN_WEEKDAYS $MORNING_CMD"
fi
if [[ "$RUN_OBSERVATION_DAILY_CHECK" == "1" ]]; then
  managed_lines="${managed_lines}"$'\n'"$OBSERVATION_CHECK_MINUTE $OBSERVATION_CHECK_HOUR * * $RUN_WEEKDAYS $OBSERVATION_DAILY_CMD"
fi
if [[ "$RUN_WEEKLY_RESEARCH_BRIEF" == "1" ]]; then
  managed_lines="${managed_lines}"$'\n'"$WEEKLY_RESEARCH_MINUTE $WEEKLY_RESEARCH_HOUR * * $WEEKLY_RESEARCH_DAY $WEEKLY_RESEARCH_CMD"
fi
managed_block="$(printf '%s\n%s\n%s\n' "$managed_lines" "$ROTATE_MINUTE $ROTATE_HOUR * * * $ROTATE_CMD" "$END_MARKER")"

new_crontab="$(printf '%s\n\n%s\n' "$clean_crontab" "$managed_block" | awk 'NF{blank=0; print; next} !blank{print; blank=1}')"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY-RUN] resulting crontab:"
  printf '%s\n' "$new_crontab"
else
  printf '%s\n' "$new_crontab" | crontab -
  echo "[OK] installed managed cron block."
  echo "[INFO] verify with: crontab -l"
fi
