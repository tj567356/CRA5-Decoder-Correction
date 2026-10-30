#!/usr/bin/env bash
# Single-stage pipeline: multilinear PCA -> CNN, no global PCA in between.
#
#   ./run_stage1only.sh 0.953 s953
#
# The comparison this exists for, at a matched storage budget:
#
#   two-stage    98% + n=2011   2,011 stored   232.5x   0.13575  (measured)
#   single-stage 95.3%          2,030 stored   230.3x   ?
#
# Within 1% on both the stored count and the compression ratio, so the answer
# needs no extrapolation: is the second stage earning its place, or is it only
# adding error the CNN cannot repair?
#
# Everything about the CNN is unchanged from all 13 other points -- same
# architecture, lr 3e-4, clip 1.0, AMP, epochs 200, patience 12, seed 0.
set -eu
E=${1:?usage: run_stage1only.sh <energy e.g. 0.953> <tag e.g. s953>}
TAG=${2:?}

BASE=/home/tong/new_20
PY=$BASE/.venv/bin/python
POINT="${TAG}_stage1only"
LOG=$BASE/logs/${POINT}.log
MODEL=$BASE/mpca_train_basis_bundle/models/mpca_spatial_${TAG}

step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

echo "=== $POINT (energy $E) started $(date) ===" | tee -a "$LOG"
cd "$BASE"

step "1/5 stage-1 basis at $E"
$PY -u scripts/fit_mpca_basis.py --energy "$E" --out "$MODEL" >> "$LOG" 2>&1

step "2/5 stage-1 floor (this is also the pre-CNN error -- there is no stage-2)"
$PY -u scripts/stage1_only_rmse.py --bases "mpca_spatial_${TAG}" --year 2018 >> "$LOG" 2>&1

step "3/5 write the stage-1-only reconstruction"
$PY -u scripts/reconstruct_stage1_only.py --model-dir "$MODEL" \
    --output-dir "decoded/${POINT}" >> "$LOG" 2>&1

step "4/5 confirm the written fields reproduce the floor, then memmap"
$PY -u scripts/eval_decoded_nrmse.py --decoded-dir "decoded/${POINT}" \
    --label "${POINT}" --year 2018 >> "$LOG" 2>&1
cd "$BASE/scripts/cnn"
$PY -u prepare_memmap.py --decoded_dir "$BASE/decoded/${POINT}" \
    --orig_dir $BASE/data/processed/fourcastnet20_conus \
    --stats_path $BASE/mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz \
    --out_dir "$BASE/data/memmap_${POINT}" >> "$LOG" 2>&1

step "5/5 champion U-Net, identical recipe to every other point"
CK="$BASE/checkpoints/${POINT}/unet_d2_c64_lr3e-4_clip1"
mkdir -p "$CK"
$PY -u train_cnn.py --energy "${POINT}" --arch unet --mid_channels 64 --depth 2 \
    --bottleneck_channels 128 --tag unet_d2_c64_lr3e-4_clip1 \
    --epochs 200 --patience 12 --lr 3e-4 --grad_clip 1.0 --weight_decay 1e-4 \
    --seed 0 --amp --cache_dir "$BASE/data/memmap_${POINT}" > "$CK/train.log" 2>&1

echo "=== $POINT finished $(date) ===" | tee -a "$LOG"
grep -E "coefficients|mean nRMSE" "$LOG" | tail -3 | tee -a "$LOG"
df -h / | tail -1 | tee -a "$LOG"
