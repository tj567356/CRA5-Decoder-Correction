"""Fit the first-stage spatial MPCA basis, with per-variable ranks under our control.

The original fitting code was run on another machine and never came across
(summary.json still carries its C:\\Users\\user\\... paths), so this is a
reimplementation from the recorded method: "spatial MPCA via mode covariance
eigendecomposition", sample axis time, feature modes latitude and longitude.

Two modes:

  --validate      refit at the original 0.99 energy threshold and compare
                  against the existing models. Ranks and eigenvalues must
                  reproduce, otherwise the reimplementation differs from
                  whatever produced them and no downstream result can be
                  attributed to the allocation change.
  --ranks FILE    take per-variable (lat_rank, lon_rank) from a reallocation
                  plan instead of an energy threshold.

Output layout matches what encode_mpca_cores.py reads, so the rest of the
pipeline runs unchanged.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import zarr

BASE = Path(os.path.expanduser("~/new_20"))
BUNDLE = BASE / "mpca_train_basis_bundle"
INPUT_DIR = BASE / "data/processed/fourcastnet20_conus"
STATS = BUNDLE / "stats/train_2014_2016_channel_stats.npz"
REF_MODELS = BUNDLE / "models/mpca_spatial_99energy"
TRAIN_YEARS = [2014, 2015, 2016]


def mode_covariance(x, mode):
    """Covariance of one spatial mode, the other mode folded into the samples.

    x is (T, n_lat, n_lon) already channel-normalised. For mode=0 this unfolds
    to (n_lat, T*n_lon) and returns a (n_lat, n_lat) Gram matrix scaled by the
    sample count -- the scale only rescales eigenvalues, never the ranks or the
    energy fractions, both of which are ratios.
    """
    a = np.moveaxis(x, mode + 1, 0).reshape(x.shape[mode + 1], -1)
    return (a @ a.T) / a.shape[1]


def eig_desc(c):
    w, v = np.linalg.eigh(c)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


def rank_for_energy(evals, threshold):
    cum = np.cumsum(np.maximum(evals, 0)) / np.maximum(evals, 0).sum()
    return int(np.searchsorted(cum, threshold) + 1), cum


def load_train(channel_index):
    """Concatenate the training years for one channel, normalised."""
    st = np.load(STATS)
    mean = float(st["mean"][channel_index])
    std = float(st["std"][channel_index])
    parts = []
    for y in TRAIN_YEARS:
        g = zarr.open_group(str(INPUT_DIR / f"{y}.zarr"), mode="r")
        parts.append(np.asarray(g["fields"][:, channel_index], dtype=np.float64))
    x = np.concatenate(parts, axis=0)
    return (x - mean) / std, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", type=float, default=0.99)
    ap.add_argument("--ranks", type=Path, help="reallocation plan json")
    ap.add_argument("--plan-key", default="greedy_equal")
    ap.add_argument("--out", type=Path, default=BUNDLE / "models/mpca_refit_validate")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--channels", nargs="*")
    args = ap.parse_args()

    st = np.load(STATS)
    names = [str(x) for x in st["channel_names"]]
    plan = json.load(open(args.ranks))[args.plan_key] if args.ranks else None

    args.out.mkdir(parents=True, exist_ok=True)
    g0 = zarr.open_group(str(INPUT_DIR / f"{TRAIN_YEARS[0]}.zarr"), mode="r")
    lat = np.asarray(g0["lat"][:], dtype=np.float32)
    lon = np.asarray(g0["lon"][:], dtype=np.float32)

    summary_channels, worst = [], {"rank": 0, "eval": 0.0, "vec": 0.0}
    for i, name in enumerate(names):
        if args.channels and name not in args.channels:
            continue
        x, mean, std = load_train(i)
        c_lat = mode_covariance(x, 0)
        c_lon = mode_covariance(x, 1)
        w_lat, v_lat = eig_desc(c_lat)
        w_lon, v_lon = eig_desc(c_lon)

        if plan:
            r1, r2 = plan[name]
            cum_lat = np.cumsum(np.maximum(w_lat, 0)) / np.maximum(w_lat, 0).sum()
            cum_lon = np.cumsum(np.maximum(w_lon, 0)) / np.maximum(w_lon, 0).sum()
        else:
            r1, cum_lat = rank_for_energy(w_lat, args.energy)
            r2, cum_lon = rank_for_energy(w_lon, args.energy)

        np.savez(args.out / f"{name}.npz",
                 channel_name=np.array(name), channel_index=np.int16(i),
                 years=np.array(TRAIN_YEARS, dtype=np.int16),
                 energy_threshold=np.float64(args.energy),
                 mean=np.float64(mean), std=np.float64(std), lat=lat, lon=lon,
                 lat_rank=np.int16(r1), lon_rank=np.int16(r2),
                 lat_retained_energy=np.float64(cum_lat[r1 - 1]),
                 lon_retained_energy=np.float64(cum_lon[r2 - 1]),
                 lat_components=v_lat[:, :r1].astype(np.float32),
                 lon_components=v_lon[:, :r2].astype(np.float32),
                 lat_eigenvalues=w_lat, lon_eigenvalues=w_lon)

        summary_channels.append({"name": name, "model_path": f"{name}.npz",
                                 "lat_rank": r1, "lon_rank": r2,
                                 "lat_retained_energy": float(cum_lat[r1 - 1]),
                                 "lon_retained_energy": float(cum_lon[r2 - 1]),
                                 "spatial_core_elements": r1 * r2,
                                 "spatial_original_elements": len(lat) * len(lon),
                                 "spatial_core_ratio": r1 * r2 / (len(lat) * len(lon))})

        line = f"{name:7s} lat {r1:3d}  lon {r2:3d}  core {r1*r2:6d}"
        if args.validate:
            ref = np.load(REF_MODELS / f"{name}.npz", allow_pickle=True)
            R1, R2 = int(ref["lat_rank"]), int(ref["lon_rank"])
            # eigenvalues are scale-free only up to the covariance normalisation
            def relspec(a, b):
                a = np.maximum(a, 0); b = np.maximum(b, 0)
                return float(np.abs(a / a.sum() - b / b.sum()).max())
            de = max(relspec(w_lat, ref["lat_eigenvalues"]),
                     relspec(w_lon, ref["lon_eigenvalues"]))
            # eigenvectors carry an arbitrary sign
            k = min(r1, R1)
            dv = float(np.abs(np.abs((v_lat[:, :k] * np.asarray(
                ref["lat_components"])[:, :k]).sum(0)) - 1).max())
            ok = "OK " if (r1 == R1 and r2 == R2) else "MISMATCH"
            line += f"  | ref lat {R1:3d} lon {R2:3d}  {ok}  spec {de:.2e}  vec {dv:.2e}"
            worst["rank"] += (r1 != R1) + (r2 != R2)
            worst["eval"] = max(worst["eval"], de)
            worst["vec"] = max(worst["vec"], dv)
        print(line, flush=True)

    json.dump({"method": "spatial MPCA via mode covariance eigendecomposition",
               "sample_axis": "time", "feature_modes": ["latitude", "longitude"],
               "rank_selection": ("from reallocation plan" if plan else
                                  f"smallest rank with cumulative energy >= {args.energy}"),
               "energy_threshold": args.energy, "years": TRAIN_YEARS,
               "input_dir": str(INPUT_DIR), "stats_path": str(STATS),
               "n_lat": len(lat), "n_lon": len(lon),
               "channels": summary_channels},
              open(args.out / "summary.json", "w"), indent=2)

    if args.validate:
        print(f"\nrank mismatches: {worst['rank']}   "
              f"max normalised-spectrum diff {worst['eval']:.3e}   "
              f"max |<v,v_ref>|-1 {worst['vec']:.3e}")
        print("reproduced" if worst["rank"] == 0 and worst["eval"] < 1e-6
              else "DOES NOT reproduce -- method differs, do not proceed")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
