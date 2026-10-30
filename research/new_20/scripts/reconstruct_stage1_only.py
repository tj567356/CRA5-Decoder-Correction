"""Write the stage-1-only reconstruction as a decoded-format zarr.

Everything downstream (prepare_memmap.py, eval_decoded_nrmse.py, train_cnn.py)
consumes `decoded/<point>/{year}.zarr` with a float32 `fields` array laid out
(time, channel, lat, lon). stage1_only_rmse.py already computes this round trip
but only keeps the error, so this script is the same arithmetic with the fields
written out -- letting a no-second-stage pipeline reuse the existing CNN path
unchanged.

    core = L^T X R        (project onto the per-variable lat/lon bases)
    X_hat = L core R^T    (and straight back, no global PCA in between)

Which is the point of the experiment: at a matched storage budget, is the second
stage earning its place? 95.3% keeps 2,030 coefficients = 230.3x end to end,
within 1% of the two-stage best (2,011 components, 232.5x, nRMSE 0.13575), so the
two are directly comparable without extrapolating anything.

    python scripts/reconstruct_stage1_only.py \
        --model-dir mpca_train_basis_bundle/models/mpca_spatial_s953 \
        --output-dir decoded/s953_stage1only
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr

BASE = Path("/home/tong/new_20")
ORIG = BASE / "data/processed/fourcastnet20_conus"
STATS = BASE / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2014, 2015, 2016, 2017, 2018])
    ap.add_argument("--chunk-time", type=int, default=256)
    args = ap.parse_args()

    names = [str(x) for x in np.load(STATS)["channel_names"]]
    models = {}
    budget = 0
    for v in names:
        m = np.load(args.model_dir / f"{v}.npz", allow_pickle=True)
        L = np.asarray(m["lat_components"], dtype=np.float32)   # (n_lat, r1)
        R = np.asarray(m["lon_components"], dtype=np.float32)   # (n_lon, r2)
        models[v] = (L, R, float(m["mean"]), float(m["std"]))
        budget += L.shape[1] * R.shape[1]
    raw = 20 * 97 * 241
    print(f"basis {args.model_dir.name}: {budget:,} coefficients "
          f"({raw / budget:.1f}x end to end, no second stage)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        src = zarr.open_group(str(ORIG / f"{year}.zarr"), mode="r")
        n_time, n_ch, n_lat, n_lon = src["fields"].shape
        out_path = args.output_dir / f"{year}.zarr"
        if out_path.exists():
            shutil.rmtree(out_path)
        g = zarr.open_group(str(out_path), mode="w")
        fields = g.create_array(
            "fields", shape=(n_time, n_ch, n_lat, n_lon),
            chunks=(min(args.chunk_time, n_time), n_ch, n_lat, n_lon),
            dtype="float32")
        g.create_array("channel", data=np.arange(n_ch, dtype=np.int16))
        g.create_array("time", data=np.asarray(src["time"][:], dtype=np.int64))
        g.attrs.update({
            "year": year,
            "layout": "time, channel, latitude, longitude",
            "n_components": None,          # no second stage in this pipeline
            "stage1_coefficients": budget,
            "stage1_basis": args.model_dir.name,
            "channel_names": names,
        })

        for ci, v in enumerate(names):
            L, R, mean, std = models[v]
            # chunk over time: the round trip holds two full-resolution copies
            for t0 in range(0, n_time, args.chunk_time):
                t1 = min(t0 + args.chunk_time, n_time)
                x = (np.asarray(src["fields"][t0:t1, ci], dtype=np.float32)
                     - mean) / std
                core = np.einsum("ha,thw,wb->tab", L, x, R, optimize=True)
                xh = np.einsum("ha,tab,wb->thw", L, core, R, optimize=True)
                fields[t0:t1, ci] = xh * std + mean
            print(f"  {year} {v}: done", flush=True)
        print(f"[INFO] {year}: saved {out_path}", flush=True)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
