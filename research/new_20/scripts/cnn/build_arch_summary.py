"""Summarise every CNN architecture checkpoint into a CSV + console table.

Auto-discovers any run under checkpoints/<energy>/<tag>/ that has
test_results.json, so newly trained architectures appear automatically.

Two columns carry the reporting metric:

  nrmse   mean over the 20 channels of rmse / train-period channel std.
          THIS is what gets reported. It is the only cross-channel average
          that means anything here.
  pooled_rmse_phys
          sqrt(mean(rmse^2)) in physical units, i.e. the old `test_rmse`.
          Surface pressure (Pa, rmse ~700) is 84% of it, so it ranks models
          by how well they compress `sp`. Kept for continuity, not for use.

The `basis` column separates runs that share an energy label but not a
stage-1 basis: checkpoints/e995/greedy_equal_n2011/ is a greedy-reallocation
run parked under e995 only because --energy used to be a closed choice list.
Averaging or best-of-ing it together with the e995 runs would compare two
different pipelines. Downstream scripts filter on basis == "99energy".

Usage:
    python build_arch_summary.py
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path("/home/tong/new_20")
CKPT_DIR = BASE / "checkpoints"
STATS = BASE / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
OUT = BASE / "results" / "cnn_arch_summary.csv"

ENERGY_ORDER = ["e90", "e95", "e97", "e99", "e995"]
VARIABLE_ORDER = [
    "u10", "v10", "t2m", "sp", "msl", "t850",
    "u1000", "v1000", "z1000", "u850", "v850", "z850",
    "u500", "v500", "z500", "t500", "z50", "r500", "r850", "tcwv",
]


def _dil_to_str(d):
    if not d:
        return None
    if isinstance(d, str):
        return d.replace(" ", "")
    return ",".join(map(str, d))


def fmt_dilations(args):
    arch = args.get("arch", "conv")
    d = _dil_to_str(args.get("dilations"))
    if d:
        return d
    b = _dil_to_str(args.get("block_dilations"))
    if b:
        return "b:" + b
    return "1,1,1,1" if arch == "conv" else "-"


def depth_from_tag(tag, arch):
    m = re.search(r"depth(\d+)|_d(\d)_", tag or "")
    if m:
        return int(m.group(1) or m.group(2))
    if arch == "unet":
        return 2
    return 1


def basis_of(test):
    """Which stage-1 MPCA basis this run's data came from.

    Prefer the memmap manifest (recorded since 2026-08-12), fall back to the
    cache_dir string. Runs predating the rename still point at .../data/memmap,
    which was the 99energy cache before the per-energy split -- everything that
    is not explicitly greedy is the 99energy basis.
    """
    src = ""
    man = test.get("memmap_manifest") or {}
    if isinstance(man, dict):
        src = str(man.get("decoded_dir", ""))
    if not src:
        src = str(test.get("args", {}).get("cache_dir") or "")
    return "greedy_equal" if "greedy" in src else "99energy"


def main():
    stats = np.load(STATS)
    names = [str(x) for x in stats["channel_names"]]
    std = stats["std"].astype(np.float64)
    assert names == VARIABLE_ORDER, "channel order drifted from the stats file"

    rows = []
    for test_p in sorted(CKPT_DIR.glob("*/*/test_results.json")) + sorted(
            CKPT_DIR.glob("*/test_results.json")):
        rel = test_p.relative_to(CKPT_DIR).parts
        energy = rel[0]
        tag = rel[1] if len(rel) > 2 else "baseline"
        with open(test_p) as f:
            test = json.load(f)
        args = test.get("args", {})
        arch = args.get("arch", "conv")
        hist_p = test_p.parent / "history.json"
        best_val_rmse = None
        best_val_epoch = None
        if hist_p.exists():
            with open(hist_p) as f:
                history = json.load(f)
            if history:
                best = min(history, key=lambda x: x["val_rmse"])
                best_val_rmse = best["val_rmse"]
                best_val_epoch = best["epoch"]

        per_var = test.get("test_per_variable_rmse", {})
        vec = np.array([per_var.get(n, np.nan) for n in names], dtype=np.float64)
        nrmse_vec = vec / std

        row = {
            "energy": energy,
            "basis": basis_of(test),
            "arch": arch,
            "tag": tag,
            "kernel_size": args.get("kernel_size", ""),
            "mid_channels": args.get("mid_channels", ""),
            "branch_channels": args.get("branch_channels", "") or "",
            "dilations": fmt_dilations(args),
            "depth": depth_from_tag(tag, arch),
            "n_params": test.get("n_params", ""),
            "nrmse": round(float(np.nanmean(nrmse_vec)), 5),
            "pooled_rmse_phys": test.get("test_overall_rmse"),
            "best_val_pooled_rmse_phys": best_val_rmse,
            "best_val_epoch": best_val_epoch,
            "lr": args.get("lr", ""),
            "grad_clip": args.get("grad_clip", "") or "",
            "seed": args.get("seed", ""),
        }
        for v, x in zip(VARIABLE_ORDER, nrmse_vec):
            row[f"nrmse_{v}"] = None if np.isnan(x) else round(float(x), 5)
        for v in VARIABLE_ORDER:
            row[f"rmse_{v}"] = per_var.get(v)
        rows.append(row)

    rows.sort(key=lambda r: r["nrmse"] if r["nrmse"] == r["nrmse"] else 1e18)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {OUT} ({len(rows)} runs)\n")

    header = (f"{'energy':>6s} {'basis':<13s} {'arch':<10s} {'tag':<34s} "
              f"{'ch':>4s} {'dil':>9s} {'params':>10s} "
              f"{'nRMSE':>8s} {'pooled':>8s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['energy']:>6s} {r['basis']:<13s} {r['arch']:<10s} {r['tag']:<34s} "
              f"{str(r['mid_channels']):>4s} {r['dilations']:>9s} "
              f"{r['n_params']:>10,} "
              f"{r['nrmse']:>8.5f} {r['pooled_rmse_phys']:>8.2f}")

    print("\nreminder: rank on nRMSE. `pooled` is sp-dominated (84%) and is kept "
          "for continuity only.")


if __name__ == "__main__":
    main()
