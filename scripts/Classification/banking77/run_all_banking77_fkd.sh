#!/usr/bin/env bash
set -euo pipefail

# Runs the three banking77 FKD scripts sequentially and writes per-script logs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS=(
  "banking77_fkd.sh"
  "banking77_fkd_1.sh"
)

for s in "${SCRIPTS[@]}"; do
  path="$SCRIPT_DIR/$s"
  if [ ! -f "$path" ]; then
    echo "ERROR: script not found: $path" >&2
    exit 2
  fi
  echo "---- Running: $s ----"
  # run each script with bash; capture stdout+stderr to a log file next to the script
  log="$SCRIPT_DIR/${s%.sh}.log"
  echo "Logging to: $log"
  bash "$path" 2>&1 | tee "$log"
  echo "---- Finished: $s (exit $?) ----"
  echo
done

echo "All scripts completed. Logs are in $SCRIPT_DIR/*.log"
