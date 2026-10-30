#!/bin/bash
# Fill in e90/e95/e97 x UNet so the compression curve is drawn with ONE
# architecture instead of four conv baselines plus one UNet.
#
# Config is identical to the current best (e99 unet_d2_c64_lr3e-4_clip1) --
# only --energy and --cache_dir change. Each level needs its own memmap because
# --cache_dir overrides --energy in train_cnn.py.
export PYTHONUNBUFFERED=1
BASE=/home/tong/fourcastnet_us_mpca
cd $BASE/scripts/cnn

LEVELS="e90 e95 e97"

# ── memmaps first: CPU/IO only, runs alongside the e995 job on the GPU ──
for LV in $LEVELS; do
    MM=$BASE/data/memmap_$LV
    if [ -f "$MM/target_2018.npy" ]; then
        echo "=== $(date +%H:%M:%S)  $MM present, skipping ==="
        continue
    fi
    echo "=== $(date +%H:%M:%S)  preparing $LV memmap ==="
    $BASE/.venv/bin/python -u prepare_memmap.py \
        --decoded_dir $BASE/decoded/$LV \
        --orig_dir $BASE/data/processed/fourcastnet20_conus \
        --stats_path $BASE/mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz \
        --out_dir $MM
done

echo "=== $(date +%H:%M:%S)  waiting for the e995 job to release the GPU ==="
while pgrep -f "run_e995.sh" > /dev/null; do sleep 60; done

TAG=unet_d2_c64_lr3e-4_clip1
for LV in $LEVELS; do
    DIR=$BASE/checkpoints/$LV/$TAG
    mkdir -p "$DIR"
    echo "=== $(date +%H:%M:%S)  training $LV x UNet ==="
    $BASE/.venv/bin/python -u train_cnn.py \
        --energy $LV \
        --arch unet \
        --mid_channels 64 \
        --depth 2 \
        --bottleneck_channels 128 \
        --tag $TAG \
        --epochs 100 \
        --patience 8 \
        --lr 3e-4 \
        --grad_clip 1.0 \
        --weight_decay 1e-4 \
        --cache_dir $BASE/data/memmap_$LV \
        --amp \
        > "$DIR/train.log" 2>&1
    echo "=== $(date +%H:%M:%S)  $LV done (exit $?) ==="
done
echo "MATRIX COMPLETE"
