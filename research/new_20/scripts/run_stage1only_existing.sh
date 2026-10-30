#!/usr/bin/env bash
# Single-stage run reusing a basis that already exists.
#
#   ./run_stage1only_existing.sh s98
#
# Same as run_stage1only.sh minus the fit: the threshold sweep already produced
# mpca_spatial_s98, and refitting would reproduce it bit for bit while burning
# two minutes and risking a mismatch with the cores that sweep already used.
#
# Why this point: the matched-compression comparison (single-stage 95.3% at 230x
# against two-stage 98%+n=2011 at 232x) changed two things at once -- the stage-1
# threshold moved from 98% to 95.3% as well as stage-2 disappearing. Holding the
# threshold at 98% isolates stage-2 as the only difference:
#
#   98% alone            5,458 stored    85.7x    ?
#   98% + n=2011         2,011 stored   232.5x    0.13575
#
# so the pair reads as "what does the second stage cost in error for the 2.7x of
# extra compression it buys".
set -eu
TAG=${1:?usage: run_stage1only_existing.sh <tag e.g. s98>}

BASE=/home/tong/new_20
PY=$BASE/.venv/bin/python
POINT="${TAG}_stage1only"
LOG=$BASE/logs/${POINT}.log
MODEL=$BASE/mpca_train_basis_bundle/models/mpca_spatial_${TAG}

[ -d "$MODEL" ] || { echo "missing basis $MODEL" >&2; exit 1; }

step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

echo "=== $POINT started $(date) ===" | tee -a "$LOG"
cd "$BASE"

step "1/4 stage-1 floor (also the pre-CNN error: there is no stage-2)"
$PY -u scripts/stage1_only_rmse.py --bases "mpca_spatial_${TAG}" --year 2018 >> "$LOG" 2>&1

step "2/4 write the stage-1-only reconstruction"
$PY -u scripts/reconstruct_stage1_only.py --model-dir "$MODEL" \
    --output-dir "decoded/${POINT}" >> "$LOG" 2>&1

step "3/4 cross-check the written fields, then memmap"
$PY -u scripts/eval_decoded_nrmse.py --decoded-dir "decoded/${POINT}" \
    --label "${POINT}" --year 2018 >> "$LOG" 2>&1
cd "$BASE/scripts/cnn"
$PY -u prepare_memmap.py --decoded_dir "$BASE/decoded/${POINT}" \
    --orig_dir $BASE/data/processed/fourcastnet20_conus \
    --stats_path $BASE/mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz \
    --out_dir "$BASE/data/memmap_${POINT}" >> "$LOG" 2>&1

step "4/4 champion U-Net, identical recipe to every other point"
CK="$BASE/checkpoints/${POINT}/unet_d2_c64_lr3e-4_clip1"
mkdir -p "$CK"
$PY -u train_cnn.py --energy "${POINT}" --arch unet --mid_channels 64 --depth 2 \
    --bottleneck_channels 128 --tag unet_d2_c64_lr3e-4_clip1 \
    --epochs 200 --patience 12 --lr 3e-4 --grad_clip 1.0 --weight_decay 1e-4 \
    --seed 0 --amp --cache_dir "$BASE/data/memmap_${POINT}" > "$CK/train.log" 2>&1

echo "=== $POINT finished $(date) ===" | tee -a "$LOG"
grep -E "coefficients|mean nRMSE" "$LOG" | tail -3 | tee -a "$LOG"
df -h / | tail -1 | tee -a "$LOG"
