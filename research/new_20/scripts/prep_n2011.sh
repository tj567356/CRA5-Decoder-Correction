#!/usr/bin/env bash
# Prepare the n=2011 operating point in new_20: stage-2 fit, decode, memmap.
#
# n=2011 is 232x end to end, which is exactly the original pipeline's best point
# (e995). Matching the component count rather than the energy threshold is what
# makes old-vs-new comparable: at equal energy the two stage-1 bases pick
# different n and the compression ratio drifts, so a difference could come from
# either the basis or from one side simply storing more.
#
# Refits rather than truncating the n=4384 model. PCA components are nested, so
# slicing would be mathematically identical, but the fit takes about a minute
# and surgery on a pickled sklearn object risks leaving one attribute stale.
#
# Also the point where the CNN can actually be tested. At n=4384 stage-2 is full
# rank on the training years, so the training residual is ~0 and the CNN has
# nothing to learn (train loss 0.0016 against val 0.0210). At 2011 the training
# residual is real.
set -eu
BASE=/home/tong/new_20
PY=$BASE/.venv/bin/python
LOG=$BASE/logs/prep_n2011.log

echo "=== n=2011 prep started $(date) ===" | tee -a "$LOG"

cd "$BASE"
echo "[$(date +%H:%M:%S)] stage-2 fit" | tee -a "$LOG"
$PY -u scripts/fit_second_stage_pca.py \
    --input-dir cores \
    --output-dir second_stage_pca/n2011 \
    --n-components 2011 >> "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] decode 5 years" | tee -a "$LOG"
$PY -u scripts/decode_second_stage_pca.py \
    --input-dir cores \
    --pca-dir second_stage_pca/n2011 \
    --output-dir decoded/n2011 \
    --mpca-dir mpca_train_basis_bundle/models/mpca_spatial_999energy \
    --years 2014 2015 2016 2017 2018 >> "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] pre-CNN nRMSE on 2018" | tee -a "$LOG"
$PY -u scripts/eval_decoded_nrmse.py \
    --decoded-dir decoded/n2011 --label n2011 --year 2018 >> "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] memmap" | tee -a "$LOG"
cd "$BASE/scripts/cnn"
$PY -u prepare_memmap.py \
    --decoded_dir $BASE/decoded/n2011 \
    --orig_dir $BASE/data/processed/fourcastnet20_conus \
    --stats_path $BASE/mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz \
    --out_dir $BASE/data/memmap_n2011 >> "$LOG" 2>&1

echo "=== n=2011 prep finished $(date) ===" | tee -a "$LOG"
grep -E "mean nRMSE|Total" "$LOG" | tail -3 | tee -a "$LOG"
