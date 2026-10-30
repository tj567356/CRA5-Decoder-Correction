#!/bin/bash
# CRA5 decode + the champion UNet, same protocol as every PCA run.
#
# The question this answers: is the CNN gain a generic post-processing win, or
# is it specific to PCA's linear truncation error? PCA gain shrinks as the
# compression rises (e995 -19.2% -> e90 -6.2%) because the leftover error gets
# less structured. VAEformer's residual is entropy-coded quantisation noise --
# the far end of that same curve -- so the prediction is a small gain.
#
# The architecture is pre-committed: unet_d2_c64_lr3e-4_clip1 was chosen on PCA
# data, and nothing here is tuned on CRA5. That makes this row free of the
# "picked the winner on the same data" objection. If it underperforms, do NOT
# come back and tune it -- that would trade a clean result for a fished one.
#
# Split is dataset.py's, unchanged: train 2014-2016, early stop on 2017,
# report 2018. Same std file as everything else, so nRMSE is comparable to
# CRA5-raw 0.0875 and PCA-e995+UNet 0.1396 (both 2018).
#
# Caveat to carry into the writeup: tcwv is not a CRA5 channel, it is derived
# from the 37-level q integral and carries a systematic -0.63 kg/m^2 bias. The
# CNN can learn that bias off the other channels, so tcwv's improvement is
# partly self-repair, not decompression. Report it separately.
set -euo pipefail

ROOT=/home/tong/fourcastnet_us_mpca
TAG=unet_d2_c64_lr3e-4_clip1
CKPT=$ROOT/checkpoints/cra5/$TAG

export PYTHONUNBUFFERED=1
cd $ROOT/scripts/cnn
mkdir -p $CKPT

$ROOT/.venv/bin/python -u train_cnn.py \
    --energy cra5 \
    --arch unet \
    --mid_channels 64 \
    --depth 2 \
    --bottleneck_channels 128 \
    --tag $TAG \
    --epochs 200 \
    --patience 12 \
    --lr 3e-4 \
    --grad_clip 1.0 \
    --weight_decay 1e-4 \
    --seed 0 \
    --batch_size 8 \
    --cache_dir $ROOT/data/memmap_cra5 \
    --amp \
    > $CKPT/train.log 2>&1
