#!/usr/bin/env bash
set -uo pipefail
ROOT="$HOME/fourcastnet_us_mpca"
cd "$ROOT"
for attempt in $(seq 1 12); do
  echo "Starting resumable RH shard $1 of 8; attempt $attempt"
  if .venv/bin/python -u revision_2026/scripts/prepare_2022_native_rh_dap.py \
    --out revision_2026/data/native_rh_2022 --start-day "$1" --step 8; then
    echo "RH shard $1 completed successfully"
    exit 0
  fi
  echo "RH shard $1 stopped after bounded HTTP retries; backing off 60 seconds"
  sleep 60
done
echo "RH shard $1 exhausted 12 resumable process attempts"
exit 1
