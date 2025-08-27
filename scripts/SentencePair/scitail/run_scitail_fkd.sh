#!/usr/bin/env bash
set -euo pipefail

# Runs the three banking77 FKD scripts sequentially and writes per-script logs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS=(
  "scitail_fkd_final.sh"
  "scitail_fkd_final_1.sh"
)

for s in "${SCRIPTS[@]}"; do
  path="$SCRIPT_DIR/$s"
  if [ ! -f "$path" ]; then
    echo "ERROR: script not found: $path" >&2
    exit 2
  fi
  echo "---- Running: $s ----"
  # run each script with bash; print output to screen only
  bash "$path"
  echo "---- Finished: $s (exit $?) ----"
  echo
done

echo "All scripts completed."

