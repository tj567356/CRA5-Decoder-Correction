#!/usr/bin/env bash
# End-to-end verification of 12-bit quantised single-stage at the 99% threshold.
#
# The bit accounting is already measured (205.5x, coefficient perturbation
# 0.00368, predicted total 0.08199 against the unquantised 0.08191). What is not
# measured is the CNN: quantisation shifts its input distribution, so the network
# is retrained from scratch on the quantised reconstruction rather than reused.
#
# Predicted end state: 0.0820 at 205x, against CRA5's 0.0875 at 472x -- the error
# crossing survives while the compression gap narrows from 10x to 2.3x.
set -eu
BITS=${1:-12}
BASE=/home/tong/new_20
PY=$BASE/.venv/bin/python
POINT="s99_q${BITS}"
LOG=$BASE/logs/${POINT}.log
MODEL=$BASE/mpca_train_basis_bundle/models/mpca_spatial_s99
CORES=/home/tong/fourcastnet_us_mpca      # the 99% cores live in the original tree

step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

echo "=== $POINT started $(date) ===" | tee -a "$LOG"
cd "$BASE"

step "1/3 quantise to ${BITS} bit and write the reconstruction"
$PY -u scripts/quantise_cores.py --cores "$CORES" --model-dir "$MODEL" \
    --bits "$BITS" --output-dir "decoded/${POINT}" >> "$LOG" 2>&1

step "2/3 pre-CNN nRMSE (should sit just above the unquantised 0.1233)"
$PY -u scripts/eval_decoded_nrmse.py --decoded-dir "decoded/${POINT}" \
    --label "${POINT}" --year 2018 >> "$LOG" 2>&1

step "3/3 memmap + champion U-Net, same recipe as every other point"
cd "$BASE/scripts/cnn"
$PY -u prepare_memmap.py --decoded_dir "$BASE/decoded/${POINT}" \
    --orig_dir $BASE/data/processed/fourcastnet20_conus \
    --stats_path $BASE/mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz \
    --out_dir "$BASE/data/memmap_${POINT}" >> "$LOG" 2>&1
CK="$BASE/checkpoints/${POINT}/unet_d2_c64_lr3e-4_clip1"
mkdir -p "$CK"
$PY -u train_cnn.py --energy "${POINT}" --arch unet --mid_channels 64 --depth 2 \
    --bottleneck_channels 128 --tag unet_d2_c64_lr3e-4_clip1 \
    --epochs 200 --patience 12 --lr 3e-4 --grad_clip 1.0 --weight_decay 1e-4 \
    --seed 0 --amp --cache_dir "$BASE/data/memmap_${POINT}" > "$CK/train.log" 2>&1

echo "=== $POINT finished $(date) ===" | tee -a "$LOG"
grep -E "coefficients|perturbation|mean nRMSE" "$LOG" | tail -8 | tee -a "$LOG"
