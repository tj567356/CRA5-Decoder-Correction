"""One-time conversion: zarr -> float16 memmap .npy (input/target) per year.

Training then memory-maps these files: data stays on disk, RSS stays low,
reads are served by the OS page cache (near-RAM speed, reclaimable).

Usage:
    python prepare_memmap.py --decoded_dir ../decoded/e99 \
        --orig_dir ../../data/processed/fourcastnet20_conus \
        --stats_path ../../mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz \
        --out_dir ../../data/memmap_e99
"""
import argparse
import os
import time

import numpy as np
import zarr

import memmap_manifest
from dataset import VARIABLE_ORDER, TRAIN_YEARS, VAL_YEARS, TEST_YEARS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoded_dir", required=True)
    parser.add_argument("--orig_dir", required=True)
    parser.add_argument("--stats_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--chunk_rows", type=int, default=256)
    args = parser.parse_args()

    stats = np.load(args.stats_path)
    mean = stats["mean"].astype(np.float32)[:, None, None]
    std = stats["std"].astype(np.float32)[:, None, None]
    os.makedirs(args.out_dir, exist_ok=True)

    years = TRAIN_YEARS + VAL_YEARS + TEST_YEARS
    for y in years:
        z_r = zarr.open(f"{args.decoded_dir}/{y}.zarr", mode="r")["fields"]
        z_o = zarr.open(f"{args.orig_dir}/{y}.zarr", mode="r")["fields"]
        shape = z_r.shape
        assert shape == z_o.shape, f"{y}: {shape} vs {z_o.shape}"
        T, C, H, W = shape

        in_path = os.path.join(args.out_dir, f"input_{y}.npy")
        tg_path = os.path.join(args.out_dir, f"target_{y}.npy")
        in_mm = np.lib.format.open_memmap(in_path, mode="w+", dtype=np.float16, shape=shape)
        tg_mm = np.lib.format.open_memmap(tg_path, mode="w+", dtype=np.float16, shape=shape)

        t0 = time.time()
        for s in range(0, T, args.chunk_rows):
            e = min(s + args.chunk_rows, T)
            recon = np.asarray(z_r[s:e], dtype=np.float32)
            orig = np.asarray(z_o[s:e], dtype=np.float32)
            in_mm[s:e] = ((recon - mean) / std).astype(np.float16)
            tg_mm[s:e] = ((orig - recon) / std).astype(np.float16)
            del recon, orig
        in_mm.flush()
        tg_mm.flush()
        print(f"{y}: {T}x{C}x{H}x{W} done in {time.time()-t0:.0f}s")

    # Provenance travels with the data: train_cnn.py reads this back and refuses to
    # train on a cache whose files no longer match it.
    memmap_manifest.write(args.out_dir, args.decoded_dir, args.orig_dir,
                          args.stats_path, years)
    print(f"Wrote {args.out_dir}/{memmap_manifest.MANIFEST} (source: {args.decoded_dir})")

    total = sum(
        os.path.getsize(os.path.join(args.out_dir, f))
        for f in os.listdir(args.out_dir)
    )
    print(f"\nDone. Total {total/2**30:.1f} GiB in {args.out_dir}")
    print("Variables:", ", ".join(VARIABLE_ORDER))


if __name__ == "__main__":
    main()
