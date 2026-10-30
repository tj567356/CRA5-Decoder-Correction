#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/fourcastnet_us_mpca"
cd "$ROOT"
for shard in 0 1 2 3 4 5 6 7; do
  nohup .venv/bin/python -u revision_2026/scripts/prepare_2022_era5.py \
    --out revision_2026/data/era5_2022 --skip-rh --workers 9 \
    --start-index "$shard" --stride 8 \
    > "revision_2026/logs/prepare_2022_era5_${shard}.log" 2>&1 < /dev/null &
  echo "shard $shard PID $!"
done
