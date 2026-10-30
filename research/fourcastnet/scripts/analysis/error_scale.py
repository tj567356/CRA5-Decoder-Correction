"""S2: is CRA5's error fine-scale, and does that predict what a CNN removes?

error_static_fraction.py showed the terrain-locked static error is real, is
near-surface only, and is almost entirely removed -- but that it accounts for the
gain in only two channels. Strip sp out and static fraction stops predicting
anything (r falls from +0.71 to +0.26). So staticness is a property of the error,
not the mechanism behind the repair.

The candidate mechanism is scale. CRA5's latent grid is 10x coarser than the data,
so what it loses is small-scale structure -- whether or not that structure moves.
Terrain is small and still; a gust front is small and moving; a 500 hPa trough is
large and moving. That ordering predicts the observed pattern where staticness
does not.

    fine-scale fraction = mean((e - smooth_k(e))²) / mean(e²)

measured at k = 3, 5, 9 so the claim does not hinge on one arbitrary window.

The test is not just the correlation -- it is whether the correlation survives
dropping any single channel. That is precisely what static fraction failed, so
leave-one-out is reported as the headline, not as a footnote.

    python scripts/analysis/error_scale.py
    -> results/error_scale.{csv,json}
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

CKPT = BASE / "checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/best.pt"
MEM = BASE / "data/memmap_cra5"
STATS = BASE / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
OUT_CSV = BASE / "results/error_scale.csv"
OUT_JSON = BASE / "results/error_scale.json"
YEARS = [2017, 2018]
KS = [3, 5, 9]
BATCH = 16
DERIVED = "tcwv"


def fine(e, k):
    """Power left after removing what a k x k box average can represent."""
    sm = F.avg_pool2d(e, k, stride=1, padding=k // 2, count_include_pad=False)
    return ((e - sm) ** 2).sum(dim=(0, 2, 3))


def moments(year, net, device, n_ch):
    x = np.load(MEM / f"input_{year}.npy", mmap_mode="r")
    y = np.load(MEM / f"target_{year}.npy", mmap_mode="r")
    n_time = x.shape[0]
    tot = {"before": torch.zeros(n_ch, device=device, dtype=torch.float64),
           "after": torch.zeros(n_ch, device=device, dtype=torch.float64)}
    fi = {(w, k): torch.zeros(n_ch, device=device, dtype=torch.float64)
          for w in ("before", "after") for k in KS}
    with torch.no_grad():
        for t0 in range(0, n_time, BATCH):
            t1 = min(t0 + BATCH, n_time)
            xb = torch.from_numpy(np.asarray(x[t0:t1], dtype=np.float32)).to(device)
            yb = torch.from_numpy(np.asarray(y[t0:t1], dtype=np.float32)).to(device)
            pred = net(xb).float()
            for w, e in (("before", -yb), ("after", pred - yb)):
                e = e.double()
                tot[w] += (e ** 2).sum(dim=(0, 2, 3))
                for k in KS:
                    fi[(w, k)] += fine(e, k)
    n_cell = n_time * x.shape[2] * x.shape[3]
    return ({w: (tot[w] / n_cell).cpu().numpy() for w in tot},
            {key: (fi[key] / n_cell).cpu().numpy() for key in fi})


def loo(a, b):
    """Leave-one-out range of Pearson r -- the test static fraction failed."""
    rs = [float(np.corrcoef(np.delete(a, i), np.delete(b, i))[0, 1])
          for i in range(len(a))]
    return min(rs), max(rs), int(np.argmin(rs)), int(np.argmax(rs))


def main():
    st = np.load(STATS)
    names = [str(v) for v in st["channel_names"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet2(20, 64, kernel_size=3, depth=2, bottleneck_channels=128).to(device)
    net.load_state_dict(torch.load(CKPT, map_location=device))
    net.eval()

    res = {}
    for yr in YEARS:
        res[yr] = moments(yr, net, device, len(names))
        print(f"{yr} done")

    rows = []
    for i, v in enumerate(names):
        row = {"channel": v}
        for yr in YEARS:
            tot, fi = res[yr]
            row[f"gain_{yr}"] = round(
                100 * (1 - float(np.sqrt(tot["after"][i] / tot["before"][i]))), 2)
            for k in KS:
                row[f"fine{k}_before_{yr}"] = round(
                    100 * float(fi[("before", k)][i] / tot["before"][i]), 2)
                row[f"fine{k}_after_{yr}"] = round(
                    100 * float(fi[("after", k)][i] / tot["after"][i]), 2)
        rows.append(row)

    yr = YEARS[-1]
    print(f"\n{'chan':<7s}{'細尺度%(k=3)':>13s}{'k=5':>8s}{'k=9':>8s}"
          f"{'修掉%':>9s}{'修完細尺度%':>13s}")
    for r in rows:
        mark = "  (排除於擬合)" if r["channel"] == DERIVED else ""
        print(f"{r['channel']:<7s}{r[f'fine3_before_{yr}']:12.2f}"
              f"{r[f'fine5_before_{yr}']:8.2f}{r[f'fine9_before_{yr}']:8.2f}"
              f"{r[f'gain_{yr}']:9.2f}{r[f'fine3_after_{yr}']:12.2f}" + mark)

    fitrows = [r for r in rows if r["channel"] != DERIVED]
    fitnames = [r["channel"] for r in fitrows]
    corr = {}
    print()
    for y in YEARS:
        for k in KS:
            a = np.array([r[f"fine{k}_before_{y}"] for r in fitrows])
            b = np.array([r[f"gain_{y}"] for r in fitrows])
            r_all = float(np.corrcoef(a, b)[0, 1])
            lo, hi, ilo, ihi = loo(a, b)
            corr[f"{y}_k{k}"] = {"r": r_all, "loo_min": lo, "loo_max": hi,
                                 "loo_min_drop": fitnames[ilo],
                                 "loo_max_drop": fitnames[ihi], "n": len(a)}
            print(f"{y} k={k}  r = {r_all:+.4f}   留一法 [{lo:+.4f}, {hi:+.4f}]"
                  f"   最脆弱：拿掉 {fitnames[ilo]}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"years": YEARS, "kernels": KS, "excluded_from_fit": DERIVED,
               "correlation": corr, "channels": rows},
              open(OUT_JSON, "w"), indent=2)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}\n      {OUT_JSON}")


if __name__ == "__main__":
    main()
