#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(basename "$SCRIPT_DIR")" = "scripts" ]; then
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  ROOT="$SCRIPT_DIR"
fi
OUT="$ROOT/outputs"

latest_match() {
  local pattern="$1"
  shopt -s nullglob
  local files=($pattern)
  shopt -u nullglob
  if [ ${#files[@]} -eq 0 ]; then
    return 1
  fi
  printf '%s\n' "${files[@]}" | sort | tail -n 1
}

print_header() {
  local title="$1"
  echo
  echo "============================================================"
  echo "$title"
  echo "============================================================"
}

print_file_excerpt() {
  local path="$1"
  local lines="${2:-120}"
  echo "[FILE] $path"
  sed -n "1,${lines}p" "$path"
}

print_log_tail() {
  local path="$1"
  local lines="${2:-60}"
  if [ -f "$path" ]; then
    echo "[LOG] $path"
    tail -n "$lines" "$path"
  fi
}

BENCH_MD="$(latest_match "$OUT/locked_benchmark_2025/*summary.md" || true)"
BENCH_JSON="$(latest_match "$OUT/locked_benchmark_2025/*summary.json" || true)"
STYLE_MD="$(latest_match "$OUT/style_compare_cloud/*summary.md" || true)"
STYLE_CSV="$(latest_match "$OUT/style_compare_cloud/*summary.csv" || true)"
STYLE_ROT_JSON="$(latest_match "$OUT/style_compare_cloud/*rotation_metrics.json" || true)"
DAILY_MD="$(latest_match "$OUT/daily_monitor_auto/daily_monitor_[0-9]*_report.md" || true)"
RISK_JSON="$(latest_match "$OUT/risk_governor/risk_gate_*.json" || true)"
DUAL_MD="$(latest_match "$OUT/daily_monitor_dual/dual_monitor_[0-9]*_report.md" || true)"
SHORTLINE_MD="$(latest_match "$OUT/shortline_opportunities/shortline_opportunity_*_report.md" || true)"

print_header "Cloud Result Snapshot"
echo "repo_root: $ROOT"
echo "now: $(date '+%Y-%m-%d %H:%M:%S %Z')"

print_header "Locked 2025 Benchmark"
if [ -n "${BENCH_MD:-}" ] && [ -f "$BENCH_MD" ]; then
  print_file_excerpt "$BENCH_MD" 140
elif [ -n "${BENCH_JSON:-}" ] && [ -f "$BENCH_JSON" ]; then
  print_file_excerpt "$BENCH_JSON" 120
else
  echo "No locked benchmark summary file found."
  print_log_tail "$OUT/ops_logs/manual_locked_2025_benchmark.log" 80
  print_log_tail "$OUT/ops_logs/cron_locked_2025_benchmark.log" 80
fi

print_header "Style Compare"
if [ -n "${STYLE_MD:-}" ] && [ -f "$STYLE_MD" ]; then
  print_file_excerpt "$STYLE_MD" 160
  if [ -n "${STYLE_CSV:-}" ] && [ -f "$STYLE_CSV" ]; then
    echo
    echo "[FILE] $STYLE_CSV"
    column -s, -t < "$STYLE_CSV" | sed -n '1,30p' || sed -n '1,30p' "$STYLE_CSV"
  fi
  if [ -n "${STYLE_ROT_JSON:-}" ] && [ -f "$STYLE_ROT_JSON" ]; then
    echo
    print_file_excerpt "$STYLE_ROT_JSON" 80
  fi
else
  echo "No style compare summary file found."
  print_log_tail "$OUT/ops_logs/manual_style_compare.log" 80
  print_log_tail "$OUT/ops_logs/cron_style_compare.log" 80
fi

print_header "Daily Monitor Summary"
if [ -n "${DAILY_MD:-}" ] && [ -f "$DAILY_MD" ]; then
  print_file_excerpt "$DAILY_MD" 200
else
  echo "No daily monitor report file found."
  print_log_tail "$OUT/ops_logs/cron_post_close.log" 80
fi

print_header "Risk Governor"
if [ -n "${RISK_JSON:-}" ] && [ -f "$RISK_JSON" ]; then
  print_file_excerpt "$RISK_JSON" 120
else
  echo "No risk governor json found."
fi

print_header "Dual Board Summary"
if [ -n "${DUAL_MD:-}" ] && [ -f "$DUAL_MD" ]; then
  print_file_excerpt "$DUAL_MD" 140
else
  echo "No dual board report found."
fi

print_header "Shortline Opportunities"
if [ -n "${SHORTLINE_MD:-}" ] && [ -f "$SHORTLINE_MD" ]; then
  print_file_excerpt "$SHORTLINE_MD" 180
else
  echo "No shortline opportunity report found."
fi

print_header "Latest Log Hints"
print_log_tail "$OUT/ops_logs/post_close_last_success.json" 40
print_log_tail "$OUT/ops_logs/post_close_last_error.json" 40
print_log_tail "$OUT/ops_logs/cron_post_close.log" 40
print_log_tail "$OUT/ops_logs/cron_pushplus_shortline.log" 40
