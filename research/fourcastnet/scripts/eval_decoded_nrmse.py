"""Per-channel nRMSE of a decoded (stage-1 + stage-2) reconstruction against the original.

This is the "+PCA, no CNN" row of the comparison table: it isolates whether a
stage-1 allocation gain survives the second-stage PCA, before any network is
trained. nRMSE is rmse / train-period channel std, the same definition used
everywhere else in this project.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import zarr

BASE = Path(os.path.expanduser("~/fourcastnet_us_mpca"))
ORIG = BASE / "data/processed/fourcastnet20_conus"
STATS = BASE / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoded-dir", type=Path, required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--out", type=Path, default=BASE / "results/stage2_compare.json")
    args = ap.parse_args()
    label = args.label or args.decoded_dir.name

    stats = np.load(STATS)
    names = [str(x) for x in stats["channel_names"]]
    std = stats["std"].astype(np.float64)

    go = zarr.open_group(str(ORIG / f"{args.year}.zarr"), mode="r")
    gd = zarr.open_group(str(args.decoded_dir / f"{args.year}.zarr"), mode="r")

    per = {}
    for i, name in enumerate(names):
        o = np.asarray(go["fields"][:, i], dtype=np.float64)
        d = np.asarray(gd["fields"][:, i], dtype=np.float64)
        rmse = float(np.sqrt(np.mean((o - d) ** 2)))
        per[name] = {"rmse": rmse, "nrmse": rmse / float(std[i])}
        print(f"  {name:7s} rmse {rmse:12.4f}   nRMSE {per[name]['nrmse']:.4f}", flush=True)

    mean_nrmse = float(np.mean([per[n]["nrmse"] for n in names]))
    print(f"\n{label}: mean nRMSE {mean_nrmse:.4f}")

    out = json.load(open(args.out)) if args.out.exists() else {}
    out[label] = {"year": args.year, "decoded_dir": str(args.decoded_dir),
                  "mean_nrmse": mean_nrmse, "per_variable": per}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
