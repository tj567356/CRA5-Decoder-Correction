#!/usr/bin/env bash
set -euo pipefail
cd /home/tong/fourcastnet_us_mpca
run=results/revision_2026/rf_ablation_d23_seed0_20261029
code=revision_2026/rf_ablation_20261029
# Training owns the GPU until frozen tests and comparable cost profiling finish.
while [[ ! -f "$run/performance_complete.json" ]]; do
  if ! pgrep -f '[p]ython -u revision_2026/rf_ablation_20261029/rf_ablation.py' > /dev/null; then
    echo 'Training stopped before completion; no postprocessing performed.' >&2
    exit 1
  fi
  sleep 30
done
.venv/bin/python -u "$code/rf_erf.py" --model "$code/model.py" --run "$run"
.venv/bin/python -u "$code/rf_postprocess.py" --run "$run"
.venv/bin/python "$code/bootstrap_gain.py" "$run/eval2018/sufficient_statistics.npz" --out "$run/bootstrap2018"
.venv/bin/python "$code/bootstrap_gain.py" "$run/eval2022/sufficient_statistics.npz" --out "$run/bootstrap2022"
.venv/bin/python "$code/rf_paired_uncertainty.py" --run "$run"
