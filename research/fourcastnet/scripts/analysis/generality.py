"""Does the near-surface / upper-air asymmetry belong to CRA5, or to compression?

Everything measured so far comes from one compressor. "CRA5 loses near-surface
detail" is a fact about CRA5; "lossy compression of this data loses near-surface
detail" is a fact about the problem, and only the second is worth a paper. The
difference is testable because a second compressor is already on disk, and it
could hardly be more different: a linear per-variable multilinear PCA followed by
a global PCA, versus a transformer VAE with a learned latent.

If both show the same split, the asymmetry is a property of the fields, not of
one architecture's inductive bias. If only CRA5 shows it, the finding is narrower
and must be written that way.

Identical code path for every configuration -- only the memmap directory and the
checkpoint change -- so nothing in the comparison can come from the analysis.

    python scripts/analysis/generality.py
    -> results/generality.{csv,json}
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(os.path.expanduser("~/fourcastnet_us_mpca"))
sys.path.insert(0, str(BASE / "scripts/cnn"))
from model import UNet2  # noqa: E402

STATS = BASE / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
OUT_CSV = BASE / "results/generality.csv"
OUT_JSON = BASE / "results/generality.json"
YEARS = [2017, 2018]
BATCH = 16
K = 5                       # the window error_scale.py found most stable
TAG = "unet_d2_c64_lr3e-4_clip1"

CONFIGS = [
    ("CRA5 (transformer VAE)", "memmap_cra5", f"checkpoints/cra5/{TAG}/best.pt"),
    ("MPCA+PCA e995 (linear)", "memmap_e995", f"checkpoints/e995/{TAG}/best.pt"),
    ("MPCA+PCA e99 (linear)", "memmap_e99", f"checkpoints/e99/{TAG}/best.pt"),
    ("MPCA+PCA e97 (linear)", "memmap_e97", f"checkpoints/e97/{TAG}/best.pt"),
]
NEAR = ["u10", "v10", "t2m", "sp", "msl", "u1000", "v1000", "z1000",
        "u850", "v850", "z850", "t850", "r850"]
UPPER = ["u500", "v500", "z500", "t500", "r500", "z50"]
DERIVED = "tcwv"


def measure(mem, ckpt, names, device):
    net = UNet2(20, 64, kernel_size=3, depth=2, bottleneck_channels=128).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    per_year = {}
    for year in YEARS:
        x = np.load(mem / f"input_{year}.npy", mmap_mode="r")
        y = np.load(mem / f"target_{year}.npy", mmap_mode="r")
        n_time = x.shape[0]
        n_ch = x.shape[1]
        acc = {k: torch.zeros(n_ch, device=device, dtype=torch.float64)
               for k in ("b", "a", "fb")}
        with torch.no_grad():
            for t0 in range(0, n_time, BATCH):
                t1 = min(t0 + BATCH, n_time)
                xb = torch.from_numpy(np.asarray(x[t0:t1],
                                                 dtype=np.float32)).to(device)
                yb = torch.from_numpy(np.asarray(y[t0:t1],
                                                 dtype=np.float32)).to(device)
                eb = (-yb).double()
                ea = (net(xb).float() - yb).double()
                acc["b"] += (eb ** 2).sum(dim=(0, 2, 3))
                acc["a"] += (ea ** 2).sum(dim=(0, 2, 3))
                sm = F.avg_pool2d(eb, K, stride=1, padding=K // 2,
                                  count_include_pad=False)
                acc["fb"] += ((eb - sm) ** 2).sum(dim=(0, 2, 3))
        ncell = n_time * x.shape[2] * x.shape[3]
        b = (acc["b"] / ncell).cpu().numpy()
        a = (acc["a"] / ncell).cpu().numpy()
        per_year[year] = {
            v: {"gain": 100 * (1 - float(np.sqrt(a[i] / b[i]))),
                "fine5": 100 * float(acc["fb"][i].cpu().numpy() / (b[i] * ncell)),
                "nrmse_before": float(np.sqrt(b[i]))}
            for i, v in enumerate(names)}
    return per_year


def main():
    names = [str(v) for v in np.load(STATS)["channel_names"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, summary = [], []

    for label, memdir, ck in CONFIGS:
        mem, ckpt = BASE / "data" / memdir, BASE / ck
        if not (mem.exists() and ckpt.exists()):
            print(f"skip {label}: missing {'memmap' if not mem.exists() else 'ckpt'}")
            continue
        res = measure(mem, ckpt, names, device)
        for v in names:
            rows.append({"config": label, "channel": v,
                         "group": "near" if v in NEAR
                                  else ("upper" if v in UPPER else "derived"),
                         **{f"{k}_{y}": round(res[y][v][k], 3)
                            for y in YEARS
                            for k in ("gain", "fine5", "nrmse_before")}})
        s = {"config": label}
        for y in YEARS:
            for g, chans in (("near", NEAR), ("upper", UPPER)):
                gs = [res[y][c]["gain"] for c in chans]
                fs = [res[y][c]["fine5"] for c in chans]
                s[f"{g}_gain_{y}"] = round(float(np.mean(gs)), 2)
                s[f"{g}_gain_min_{y}"] = round(min(gs), 2)
                s[f"{g}_gain_max_{y}"] = round(max(gs), 2)
                s[f"{g}_fine5_{y}"] = round(float(np.mean(fs)), 2)
            s[f"ratio_{y}"] = round(s[f"near_gain_{y}"] / s[f"upper_gain_{y}"], 1) \
                if s[f"upper_gain_{y}"] > 0 else None
        summary.append(s)
        print(f"{label}  近地面 {s['near_gain_2018']:.1f}%  "
              f"上層 {s['upper_gain_2018']:.1f}%  倍數 {s['ratio_2018']}")

    y = YEARS[-1]
    print(f"\n{'壓縮器':<26s}{'近地面平均':>10s}{'範圍':>16s}"
          f"{'上層平均':>10s}{'範圍':>16s}{'倍數':>7s}")
    print("-" * 86)
    for s in summary:
        print(f"{s['config']:<26s}{s[f'near_gain_{y}']:>9.1f}%"
              f"{s[f'near_gain_min_{y}']:>8.1f}–{s[f'near_gain_max_{y}']:<7.1f}"
              f"{s[f'upper_gain_{y}']:>9.1f}%"
              f"{s[f'upper_gain_min_{y}']:>8.1f}–{s[f'upper_gain_max_{y}']:<7.1f}"
              f"{str(s[f'ratio_{y}']):>7s}")

    print(f"\n細尺度誤差比例 (k={K}，CNN 前)")
    for s in summary:
        print(f"{s['config']:<26s}近地面 {s[f'near_fine5_{y}']:5.1f}%   "
              f"上層 {s[f'upper_fine5_{y}']:5.1f}%")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"years": YEARS, "k": K, "near": NEAR, "upper": UPPER,
               "derived": DERIVED, "summary": summary, "channels": rows},
              open(OUT_JSON, "w"), indent=2)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}\n      {OUT_JSON}")


if __name__ == "__main__":
    main()
