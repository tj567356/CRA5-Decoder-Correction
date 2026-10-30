#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/fourcastnet_us_mpca"
cd "$ROOT"
for shard in 0 1 2 3 4 5 6 7; do
  nohup bash revision_2026/scripts/worker_2022_native_rh.sh "$shard" \
    >> "revision_2026/logs/prepare_2022_native_rh_${shard}.log" 2>&1 < /dev/null &
  echo "shard $shard PID $!"
done
