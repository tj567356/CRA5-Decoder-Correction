"""The control the elevation result needs: is it terrain, or is it just the west?

In CONUS every high cell is a western cell, so "error grows with elevation" and
"error grows toward the Pacific storm track" fit the same numbers. The upper-air
channels already hint that something besides terrain is at work: at 5.5 km, where
no mountain reaches, error still rises 1.37x across the same bands.

A second confound is worse and was never stated: the 0-200 m band is largely
OCEAN. Comparing it against the Rockies compares sea with land, not low with high.

So this splits three ways and reports the contrasts that survive:

  land / sea     from the diurnal amplitude of 2 m temperature -- land swings
                 between day and night, water does not. 6-hourly data gives four
                 phases per day, which is enough to separate them and needs no
                 external mask.
  west / east    at 100 W, the conventional divide and roughly where the plains
                 begin to climb.
  elevation      the existing bands, but now computed within land and within west,
                 so the comparison holds the other two fixed.

If the elevation gradient survives inside western land alone, terrain is doing
real work. If it vanishes, the earlier result was longitude wearing a costume.

    python scripts/analysis/elevation_controls.py
    -> results/elevation_controls.{csv,json}
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

BASE = Path(os.path.expanduser("~/fourcastnet_us_mpca"))
sys.path.insert(0, str(BASE / "scripts/cnn"))
from model import UNet2  # noqa: E402

CKPT = BASE / "checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/best.pt"
MEM = BASE / "data/memmap_cra5"
STATS = BASE / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
ORIG = BASE / "data/processed/fourcastnet20_conus"
OUT_CSV = BASE / "results/elevation_controls.csv"
OUT_JSON = BASE / "results/elevation_controls.json"
YEAR = 2018
BATCH = 16
BANDS = [(0, 200), (200, 600), (600, 1200), (1200, 2000), (2000, 9999)]
WEST_OF = -100.0
NEAR = ["u10", "v10", "t2m", "sp", "msl", "u1000", "v1000", "z1000",
        "u850", "v850", "z850", "t850", "r850"]
UPPER = ["u500", "v500", "z500", "t500", "r500", "z50"]


def main():
    st = np.load(STATS)
    names = [str(v) for v in st["channel_names"]]
    std, mean = st["std"].astype(np.float64), st["mean"].astype(np.float64)
    sp_i, t2_i = names.index("sp"), names.index("t2m")
    lon = np.asarray(zarr.open_group(str(ORIG / f"{YEAR}.zarr"), mode="r")["lon"])

    x = np.load(MEM / f"input_{YEAR}.npy", mmap_mode="r")
    y = np.load(MEM / f"target_{YEAR}.npy", mmap_mode="r")
    n_time, n_ch, n_lat, n_lon = x.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet2(20, 64, kernel_size=3, depth=2, bottleneck_channels=128).to(device)
    net.load_state_dict(torch.load(CKPT, map_location=device))
    net.eval()

    sq_b = np.zeros((n_ch, n_lat, n_lon))
    sq_a = np.zeros((n_ch, n_lat, n_lon))
    sp_sum = np.zeros((n_lat, n_lon))
    t2_phase = np.zeros((4, n_lat, n_lon))     # 6-hourly -> four phases per day
    t2_cnt = np.zeros(4)
    with torch.no_grad():
        for t0 in range(0, n_time, BATCH):
            t1 = min(t0 + BATCH, n_time)
            xb = torch.from_numpy(np.asarray(x[t0:t1], dtype=np.float32)).to(device)
            yb = np.asarray(y[t0:t1], dtype=np.float64)
            pred = net(xb).float().cpu().numpy().astype(np.float64)
            sq_b += (yb ** 2).sum(axis=0)
            sq_a += ((pred - yb) ** 2).sum(axis=0)
            truth_sp = (np.asarray(x[t0:t1, sp_i], dtype=np.float64) * std[sp_i]
                        + mean[sp_i] + yb[:, sp_i] * std[sp_i])
            sp_sum += truth_sp.sum(axis=0)
            truth_t2 = (np.asarray(x[t0:t1, t2_i], dtype=np.float64) * std[t2_i]
                        + mean[t2_i] + yb[:, t2_i] * std[t2_i])
            for k in range(t1 - t0):
                p = (t0 + k) % 4
                t2_phase[p] += truth_t2[k]
                t2_cnt[p] += 1
    rmse_b = np.sqrt(sq_b / n_time)
    rmse_a = np.sqrt(sq_a / n_time)

    elev = np.clip(44330.0 * (1.0 - (sp_sum / n_time / 101325.0) ** (1 / 5.255)),
                   0, None)
    diurnal = (t2_phase / t2_cnt[:, None, None]).std(axis=0)
    # Bimodal by construction: water has enormous heat capacity, land does not.
    # A fixed 1 K sits in the trough between the two modes; a median split would
    # instead force land and sea to be half the domain each, which is an assumption
    # about the coastline rather than a measurement of it. Percentiles are printed
    # so the choice can be checked against the actual distribution.
    thr = 1.0
    land = diurnal > thr
    qs = np.percentile(diurnal, [5, 25, 40, 50, 60, 75, 95])
    print("日夜溫差分位數 (K): " + "  ".join(f"{p}%={v:.2f}" for p, v in
                                        zip([5, 25, 40, 50, 60, 75, 95], qs)))
    west = np.broadcast_to(lon[None, :] < WEST_OF, elev.shape)
    print(f"日夜溫差中位數 {thr:.3f} K   陸地格 {land.sum():,} / {land.size:,}")
    print(f"海格日夜溫差 {diurnal[~land].mean():.3f} K   "
          f"陸格 {diurnal[land].mean():.3f} K")
    print(f"西部格 {west.sum():,}   東部格 {(~west).sum():,}\n")

    def agg(mask, chans):
        idx = [names.index(c) for c in chans]
        if mask.sum() == 0:
            return None
        b = float(np.sqrt((rmse_b[idx][:, mask] ** 2).mean(axis=1).mean()))
        a = float(np.sqrt((rmse_a[idx][:, mask] ** 2).mean(axis=1).mean()))
        return {"n": int(mask.sum()), "before": b, "after": a,
                "gain_pct": 100 * (1 - a / b)}

    rows = []

    def add(label, mask):
        for grp, chans in (("near", NEAR), ("upper", UPPER)):
            r = agg(mask, chans)
            if r:
                rows.append({"stratum": label, "group": grp, **r})
                print(f"{label:<28s}{grp:<7s}{r['n']:>7,}"
                      f"{r['before']:>10.4f}{r['after']:>10.4f}"
                      f"{r['gain_pct']:>8.1f}%")

    print(f"{'分層':<28s}{'組':<7s}{'格數':>7s}{'CNN前':>10s}{'CNN後':>10s}{'修掉':>9s}")
    print("-" * 71)
    add("全部", np.ones_like(land))
    add("海", ~land)
    add("陸", land)
    print()
    for lo, hi in ((0, 200), (200, 600)):
        m = (elev >= lo) & (elev < hi) & land
        add(f"陸 {lo}-{hi}m 西", m & west)
        add(f"陸 {lo}-{hi}m 東", m & ~west)
    print()
    for lo, hi in BANDS:
        add(f"陸+西 {lo}-{hi if hi < 9999 else '+'}m",
            (elev >= lo) & (elev < hi) & land & west)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"year": YEAR, "west_of": WEST_OF, "land_threshold_K": thr,
               "rows": rows}, open(OUT_JSON, "w"), indent=2)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}\n      {OUT_JSON}")


if __name__ == "__main__":
    main()
