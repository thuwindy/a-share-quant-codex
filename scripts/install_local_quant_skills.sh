#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT_DIR/codex_skills"
TARGET_DIR="${HOME}/.codex/skills"

mkdir -p "$TARGET_DIR"

for skill in qlib-alpha factor-check fast-ablation alpha-prototype; do
  if [[ ! -d "$SRC_DIR/$skill" ]]; then
    echo "missing skill: $skill" >&2
    exit 1
  fi
  rsync -az --delete "$SRC_DIR/$skill/" "$TARGET_DIR/$skill/"
  echo "installed: $skill"
done

echo "done"
