#!/bin/bash
# e995 x UNet -- the missing cell in the energy x architecture matrix.
#
# NOTE: --cache_dir overrides --energy in train_cnn.py (the dataset reads the
# memmap and ignores decoded_dir entirely), so e995 needs its OWN memmap dir.
# Reusing data/memmap here would silently train on e99 data.
export PYTHONUNBUFFERED=1
BASE=/home/tong/fourcastnet_us_mpca
MM=$BASE/data/memmap_e995
cd $BASE/scripts/cnn

if [ ! -f "$MM/target_2018.npy" ]; then
    echo "=== $(date +%H:%M:%S)  preparing e995 memmap -> $MM ==="
    $BASE/.venv/bin/python -u prepare_memmap.py \
        --decoded_dir $BASE/decoded/e995 \
        --orig_dir $BASE/data/processed/fourcastnet20_conus \
        --stats_path $BASE/mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz \
        --out_dir $MM
else
    echo "=== $MM already present, skipping prepare ==="
fi

# Wait for the seed-repeat runs to finish so the two jobs do not halve each
# other's throughput on a single GPU.
echo "=== $(date +%H:%M:%S)  waiting for seed repeats to finish ==="
while pgrep -f "run_seeds.sh" > /dev/null; do sleep 60; done
echo "=== $(date +%H:%M:%S)  seed repeats done, starting e995 UNet ==="

TAG=unet_d2_c64_lr3e-4_clip1
DIR=$BASE/checkpoints/e995/$TAG
mkdir -p "$DIR"
$BASE/.venv/bin/python -u train_cnn.py \
    --energy e995 \
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
    --cache_dir $MM \
    --amp \
    > "$DIR/train.log" 2>&1
echo "=== $(date +%H:%M:%S)  e995 UNet done (exit $?) ==="
