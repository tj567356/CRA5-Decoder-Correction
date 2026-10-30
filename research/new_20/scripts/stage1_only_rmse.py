"""Stage-1-only reconstruction error: project onto an MPCA basis and come straight back.

No stage-2 PCA, no CNN. Every basis compared here holds the same total coefficient
budget, so compression is pinned and the allocation is the only free variable --
this is the controlled comparison for the greedy reallocation.

The round trip is x_hat = L (L^T x R) R^T on channel-normalised data, so the
reported nRMSE is just the RMS of the normalised residual, directly comparable to
the sqrt(1 - E_lat*E_lon) estimates in results/stage1_reallocation.json.
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

DEFAULT_BASES = ["mpca_spatial_99energy", "mpca_greedy_equal", "mpca_greedy_nwp"]
RAW_PER_FRAME = 20 * 97 * 241


def evaluate(model_dir, names, stats, year):
    """Per-channel nRMSE of the stage-1 round trip, plus the coefficient budget."""
    out, budget = {}, 0
    for i, name in enumerate(names):
        m = np.load(model_dir / f"{name}.npz", allow_pickle=True)
        L = np.asarray(m["lat_components"], dtype=np.float32)   # (n_lat, r1)
        R = np.asarray(m["lon_components"], dtype=np.float32)   # (n_lon, r2)
        mean = float(stats["mean"][i])
        std = float(stats["std"][i])

        g = zarr.open_group(str(INPUT_DIR / f"{year}.zarr"), mode="r")
        x = (np.asarray(g["fields"][:, i], dtype=np.float32) - mean) / std

        core = np.einsum("ha,thw,wb->tab", L, x, R, optimize=True)
        xh = np.einsum("ha,tab,wb->thw", L, core, R, optimize=True)
        resid = x - xh
        nrmse = float(np.sqrt(np.mean(resid ** 2)))

        r1, r2 = L.shape[1], R.shape[1]
        budget += r1 * r2
        out[name] = {"lat_rank": r1, "lon_rank": r2, "coeffs": r1 * r2,
                     "nrmse": nrmse, "rmse_physical": nrmse * std}
        print(f"  {name:7s} {r1:3d}x{r2:<3d} = {r1*r2:5d}   nRMSE {nrmse:.4f}", flush=True)
    return out, budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--bases", nargs="*", default=DEFAULT_BASES)
    ap.add_argument("--out", type=Path, default=BASE / "results/stage1_only_rmse.json")
    args = ap.parse_args()

    stats = np.load(STATS)
    names = [str(x) for x in stats["channel_names"]]

    # The reallocation estimate only exists in the original tree; this script
    # is useful for any basis, so its absence must not be fatal.
    _est_p = BASE / "results/stage1_reallocation.json"
    est = json.load(open(_est_p))["nrmse_estimate"] if _est_p.exists() else {}
    est_key = {"mpca_spatial_99energy": "current", "mpca_greedy_equal": "greedy_equal",
               "mpca_greedy_nwp": "greedy_nwp"}

    results = {}
    for b in args.bases:
        d = BUNDLE / "models" / b
        if not d.is_dir():
            print(f"[skip] {d} missing")
            continue
        print(f"\n=== {b} ===", flush=True)
        per, budget = evaluate(d, names, stats, args.year)
        mean_nrmse = float(np.mean([per[n]["nrmse"] for n in names]))
        results[b] = {"per_variable": per, "budget": budget,
                      "compression_stage1": RAW_PER_FRAME / budget,
                      "mean_nrmse": mean_nrmse}
        print(f"  budget {budget} ({RAW_PER_FRAME/budget:.2f}x)   mean nRMSE {mean_nrmse:.4f}")

    ref = results.get("mpca_spatial_99energy")
    print(f"\n{'basis':22s} {'coeffs':>7s} {'x':>7s} {'measured':>9s} {'estimate':>9s} {'vs base':>8s}")
    for b, r in results.items():
        e = est.get(est_key.get(b, ""), None)
        em = f"{np.mean([e[n] for n in names]):9.4f}" if e else " " * 9
        rel = f"{100*(r['mean_nrmse']/ref['mean_nrmse'] - 1):+7.1f}%" if ref else " " * 8
        print(f"{b:22s} {r['budget']:7d} {r['compression_stage1']:6.2f}x "
              f"{r['mean_nrmse']:9.4f} {em} {rel}")

    if ref and len(results) > 1:
        print(f"\n{'chan':7s} {'base':>8s} " +
              " ".join(f"{b.replace('mpca_greedy_', 'g_'):>8s}" for b in results if b != "mpca_spatial_99energy"))
        for n in names:
            row = " ".join(f"{results[b]['per_variable'][n]['nrmse']:8.4f}"
                           for b in results if b != "mpca_spatial_99energy")
            print(f"{n:7s} {ref['per_variable'][n]['nrmse']:8.4f} {row}")

    # Merge, never replace: measuring one basis used to delete every basis
    # measured before it, and three downstream scripts read this file without
    # knowing what order the runs happened in.
    merged = {}
    if args.out.exists():
        try:
            merged = json.load(open(args.out)).get("bases", {})
        except Exception:
            merged = {}
    merged.update(results)
    json.dump({"year": args.year, "raw_per_frame": RAW_PER_FRAME, "bases": merged},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
