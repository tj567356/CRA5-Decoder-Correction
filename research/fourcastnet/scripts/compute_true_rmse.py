"""
Compute true RMSE: two-stage reconstruction vs original processed fields.

Original fields are at: input_dir/data/processed/fourcastnet20_conus/{year}.zarr["fields"]
Decoded fields are at:   output_base_dir/{level}/{year}.zarr["fields"]
"""

import json
import time
from pathlib import Path

import numpy as np
import zarr


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--original-dir", type=Path, required=True)
    p.add_argument("--decoded-base-dir", type=Path, required=True)
    p.add_argument("--energy-levels", nargs="+", default=["e90", "e95", "e97", "e99", "e995"])
    p.add_argument("--years", type=int, nargs="+", default=[2014, 2015, 2016, 2017, 2018])
    p.add_argument("--variable-order", nargs="+", default=[
        "u10","v10","t2m","sp","msl","t850","u1000","v1000","z1000",
        "u850","v850","z850","u500","v500","z500","t500","z50","r500","r850","tcwv"
    ])
    args = p.parse_args()

    # Load all originals
    originals = {}
    for year in args.years:
        g = zarr.open_group(str(args.original_dir / f"{year}.zarr"), mode="r")
        originals[str(year)] = np.asarray(g["fields"][:], dtype=np.float32)
        print(f"[INFO] Original {year}: {originals[str(year)].shape}")

    variable_order = args.variable_order

    all_results = {}
    for level in args.energy_levels:
        print(f"\n{'='*60}")
        print(f"  {level}")
        print(f"{'='*60}")

        level_rmse = {}
        for year in args.years:
            g = zarr.open_group(str(args.decoded_base_dir / level / f"{year}.zarr"), mode="r")
            recon = np.asarray(g["fields"][:], dtype=np.float32)
            orig = originals[str(year)]

            assert recon.shape == orig.shape, f"Shape mismatch {year}: {recon.shape} vs {orig.shape}"

            # Overall RMSE
            overall = float(np.sqrt(np.mean((recon - orig) ** 2)))

            # Per-variable RMSE
            per_var = {}
            for ch_idx, v in enumerate(variable_order):
                diff = recon[:, ch_idx, :, :] - orig[:, ch_idx, :, :]
                per_var[v] = float(np.sqrt(np.mean(diff ** 2)))

            level_rmse[str(year)] = {"overall": overall, "per_variable": per_var}
            print(f"  {year}: overall RMSE = {overall:.4f}")

        # Per-variable table
        print(f"\n  Per-variable RMSE ({level}):")
        header = f"  {'variable':>10s}"
        for y in args.years:
            header += f"  {str(y):>10s}"
        print(header)
        for v in variable_order:
            row = f"  {v:>10s}"
            for y in args.years:
                row += f"  {level_rmse[str(y)]['per_variable'][v]:10.4f}"
            print(row)

        all_results[level] = level_rmse

    # Cross-level summary
    pca_base = args.decoded_base_dir.parent / "second_stage_pca"
    n_comps = {}
    for level in args.energy_levels:
        with open(pca_base / level / "metadata.json") as f:
            n_comps[level] = json.load(f)["n_components"]

    print(f"\n{'='*60}")
    print("  CROSS-LEVEL OVERALL RMSE (true RMSE vs original fields)")
    print(f"{'='*60}")
    header = f"  {'level':>6s} {'n_comp':>7s}"
    for y in args.years:
        header += f"  {str(y):>10s}"
    print(header)
    for level in args.energy_levels:
        row = f"  {level:>6s} {n_comps[level]:>7d}"
        for y in args.years:
            row += f"  {all_results[level][str(y)]['overall']:10.4f}"
        print(row)

    out_path = args.decoded_base_dir / "true_rmse_results.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Saved {out_path}")


if __name__ == "__main__":
    main()
