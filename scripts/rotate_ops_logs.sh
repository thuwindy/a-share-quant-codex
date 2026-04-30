#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/ops_logs}"
KEEP_PLAIN_DAYS="${KEEP_PLAIN_DAYS:-7}"
KEEP_GZIP_DAYS="${KEEP_GZIP_DAYS:-45}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_DIR"

to_int() {
  local value="$1"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] expected integer, got: $value" >&2
    exit 2
  fi
}

to_int "$KEEP_PLAIN_DAYS"
to_int "$KEEP_GZIP_DAYS"
to_int "$DRY_RUN"

echo "[INFO] rotate_ops_logs start: $(date '+%F %T %Z')"
echo "[INFO] LOG_DIR=$LOG_DIR KEEP_PLAIN_DAYS=$KEEP_PLAIN_DAYS KEEP_GZIP_DAYS=$KEEP_GZIP_DAYS DRY_RUN=$DRY_RUN"

compress_count=0
delete_count=0

while IFS= read -r -d '' file; do
  compress_count=$((compress_count + 1))
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] gzip -f $file"
  else
    gzip -f "$file"
  fi
done < <(find "$LOG_DIR" -type f -name 'post_close_*.log' -mtime +"$KEEP_PLAIN_DAYS" -print0)

while IFS= read -r -d '' file; do
  delete_count=$((delete_count + 1))
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] rm -f $file"
  else
    rm -f "$file"
  fi
done < <(find "$LOG_DIR" -type f -name 'post_close_*.log.gz' -mtime +"$KEEP_GZIP_DAYS" -print0)

echo "[INFO] rotate_ops_logs done: compressed=$compress_count deleted=$delete_count"
