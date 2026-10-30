"""Quantise stage-1 cores to a fixed bit depth and write the reconstruction.

The single-stage pipeline stores raw float32 coefficients, which is why its
compression ratio is only 45.9x while CRA5 reaches 471.6x on an entropy-coded
bitstream. Float32 coefficients barely compress losslessly (49x, +7%) because the
mantissa bits are near-random; quantising first is what unlocks the gain.

Measured on 2018, quantising per variable then LZMA:

    16 bit   129.5x   coefficient perturbation 0.00023   total 0.08191  (+0.00%)
    12 bit   205.5x                            0.00368         0.08199  (+0.10%)
    10 bit   294.7x                            0.01474         0.08323  (+1.61%)
     8 bit   538.4x                            0.05902         0.10096  (+23.3%)

Per variable, not globally: z50's four coefficients and u10's 1,404 span
different orders of magnitude, and sharing one scale wastes most of the range.
The per-variable min/max costs 2 floats per variable per year, negligible.

Because the lat and lon bases are orthonormal eigenvector matrices, a coefficient
perturbation maps to an equal-norm field perturbation, so the error cost above is
exact rather than estimated. What it does NOT capture is the effect on the CNN,
whose input distribution shifts slightly -- hence this script writes the fields
so the network can be retrained and the end-to-end number measured.

    python scripts/quantise_cores.py --cores <dir> --model-dir <dir> \
        --bits 12 --output-dir decoded/<point>
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
    ap.add_argument("--cores", type=Path, required=True,
                    help="dir holding {year}.zarr of per-variable cores")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bits", type=int, default=12)
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2014, 2015, 2016, 2017, 2018])
    ap.add_argument("--chunk-time", type=int, default=256)
    args = ap.parse_args()

    names = [str(x) for x in np.load(STATS)["channel_names"]]
    models = {}
    budget = 0
    for v in names:
        m = np.load(args.model_dir / f"{v}.npz", allow_pickle=True)
        L = np.asarray(m["lat_components"], dtype=np.float32)
        R = np.asarray(m["lon_components"], dtype=np.float32)
        models[v] = (L, R, float(m["mean"]), float(m["std"]))
        budget += L.shape[1] * R.shape[1]
    levels = 2 ** args.bits - 1
    print(f"{budget:,} coefficients/frame, quantised to {args.bits} bit "
          f"per variable", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        src = zarr.open_group(str(ORIG / f"{year}.zarr"), mode="r")
        cg = zarr.open_group(str(args.cores / f"{year}.zarr"), mode="r")
        n_time, n_ch, n_lat, n_lon = src["fields"].shape
        out = args.output_dir / f"{year}.zarr"
        if out.exists():
            shutil.rmtree(out)
        g = zarr.open_group(str(out), mode="w")
        fields = g.create_array(
            "fields", shape=(n_time, n_ch, n_lat, n_lon),
            chunks=(min(args.chunk_time, n_time), n_ch, n_lat, n_lon),
            dtype="float32")
        g.create_array("channel", data=np.arange(n_ch, dtype=np.int16))
        g.create_array("time", data=np.asarray(src["time"][:], dtype=np.int64))
        g.attrs.update({"year": year,
                        "layout": "time, channel, latitude, longitude",
                        "stage1_coefficients": budget,
                        "quantisation_bits": args.bits,
                        "stage1_basis": args.model_dir.name,
                        "channel_names": names})

        sq = 0.0
        for ci, v in enumerate(names):
            L, R, mean, std = models[v]
            core = np.asarray(cg[v][:], dtype=np.float32)
            lo, hi = float(core.min()), float(core.max())
            q = np.round((core - lo) / (hi - lo) * levels)
            deq = (q / levels * (hi - lo) + lo).astype(np.float32)
            sq += float(((core - deq) ** 2).sum())
            for t0 in range(0, n_time, args.chunk_time):
                t1 = min(t0 + args.chunk_time, n_time)
                xh = np.einsum("ha,tab,wb->thw", L, deq[t0:t1], R, optimize=True)
                fields[t0:t1, ci] = xh * std + mean
        pert = np.sqrt(sq / (n_time * n_ch * n_lat * n_lon))
        print(f"[INFO] {year}: saved {out}   quantisation perturbation "
              f"{pert:.5f} (normalised)", flush=True)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
