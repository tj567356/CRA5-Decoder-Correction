"""Re-test the surface-pressure terrain claim under the controls that broke the rest.

why_sp.py reported CRA5's sp error climbing 3.9x from plain to mountain against
2.2x for the linear pipeline, and that contrast is currently doing mechanism work
in the write-up. But elevation_controls.py showed the same style of stratification
was confounded twice over: the lowest band is one third ocean, and every high cell
is also a western cell. The sp claim has not been re-tested under those controls,
so it is being cited on the strength of a measurement that its siblings failed.

Same three strata, both compressors, physical units this time because Pa is what
the original claim was stated in:

  uncontrolled   all cells, as originally measured
  land only      removes the sea/mountain comparison
  land + west    removes the longitude gradient as well

If the 3.9x survives inside western land, terrain is real for sp specifically even
though it was not for the group. If it collapses the way the group did, the claim
comes out of the paper.

    python scripts/analysis/sp_terrain_control.py
    -> results/sp_terrain_control.{csv,json}
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

STATS = BASE / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
ORIG = BASE / "data/processed/fourcastnet20_conus"
OUT_CSV = BASE / "results/sp_terrain_control.csv"
OUT_JSON = BASE / "results/sp_terrain_control.json"
YEAR = 2018
BATCH = 16
TAG = "unet_d2_c64_lr3e-4_clip1"
WEST_OF = -100.0
BANDS = [(0, 200), (200, 600), (600, 1200), (1200, 2000), (2000, 9999)]
CONFIGS = [("CRA5", "memmap_cra5", f"checkpoints/cra5/{TAG}/best.pt"),
           ("MPCA+PCA e995", "memmap_e995", f"checkpoints/e995/{TAG}/best.pt")]
CHANNEL = "sp"


def main():
    st = np.load(STATS)
    names = [str(v) for v in st["channel_names"]]
    i = names.index(CHANNEL)
    std, mean = float(st["std"][i]), float(st["mean"][i])
    lon = np.asarray(zarr.open_group(str(ORIG / f"{YEAR}.zarr"), mode="r")["lon"])
    t2_i = names.index("t2m")
    t2_std, t2_mean = float(st["std"][t2_i]), float(st["mean"][t2_i])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = {}
    elev = diurnal = None
    for label, memdir, ck in CONFIGS:
        mem, ckpt = BASE / "data" / memdir, BASE / ck
        if not (mem.exists() and ckpt.exists()):
            print(f"skip {label}")
            continue
        net = UNet2(20, 64, kernel_size=3, depth=2,
                    bottleneck_channels=128).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net.eval()
        x = np.load(mem / f"input_{YEAR}.npy", mmap_mode="r")
        y = np.load(mem / f"target_{YEAR}.npy", mmap_mode="r")
        n_time, _, n_lat, n_lon = x.shape
        sq_b = np.zeros((n_lat, n_lon))
        sq_a = np.zeros((n_lat, n_lon))
        sp_sum = np.zeros((n_lat, n_lon))
        t2_phase = np.zeros((4, n_lat, n_lon))
        t2_cnt = np.zeros(4)
        with torch.no_grad():
            for t0 in range(0, n_time, BATCH):
                t1 = min(t0 + BATCH, n_time)
                xb = torch.from_numpy(np.asarray(x[t0:t1],
                                                 dtype=np.float32)).to(device)
                yb = np.asarray(y[t0:t1], dtype=np.float64)
                pred = net(xb).float().cpu().numpy().astype(np.float64)
                # physical Pa: the memmaps store errors divided by the channel std
                sq_b += ((yb[:, i] * std) ** 2).sum(axis=0)
                sq_a += (((pred[:, i] - yb[:, i]) * std) ** 2).sum(axis=0)
                sp_sum += (np.asarray(x[t0:t1, i], dtype=np.float64) * std + mean
                           + yb[:, i] * std).sum(axis=0)
                if elev is None:
                    tt = (np.asarray(x[t0:t1, t2_i], dtype=np.float64) * t2_std
                          + t2_mean + yb[:, t2_i] * t2_std)
                    for k in range(t1 - t0):
                        t2_phase[(t0 + k) % 4] += tt[k]
                        t2_cnt[(t0 + k) % 4] += 1
        out[label] = {"before": np.sqrt(sq_b / n_time),
                      "after": np.sqrt(sq_a / n_time)}
        if elev is None:
            elev = np.clip(44330.0 * (1 - (sp_sum / n_time / 101325.0)
                                      ** (1 / 5.255)), 0, None)
            diurnal = (t2_phase / t2_cnt[:, None, None]).std(axis=0)
        print(f"{label}: 全域 sp RMSE {np.sqrt((out[label]['before']**2).mean()):.1f} Pa"
              f" → {np.sqrt((out[label]['after']**2).mean()):.1f} Pa")

    land = diurnal > 1.0
    west = np.broadcast_to(lon[None, :] < WEST_OF, elev.shape)
    strata = [("未控制（全部格點）", np.ones_like(land)),
              ("只看陸地", land),
              ("陸地 + 西部", land & west)]

    rows = []
    for sname, base_mask in strata:
        print(f"\n=== {sname}")
        hdr = f"{'海拔':<14s}{'格數':>7s}"
        for label in out:
            hdr += f"{label + ' 前':>16s}{label + ' 後':>16s}"
        print(hdr)
        first = {}
        for lo, hi in BANDS:
            m = base_mask & (elev >= lo) & (elev < hi)
            if m.sum() < 30:
                print(f"{f'{lo}-{hi}m':<14s}{int(m.sum()):>7,}   格數不足，跳過")
                continue
            line = f"{f'{lo}-{hi}m':<14s}{int(m.sum()):>7,}"
            row = {"stratum": sname, "band": f"{lo}-{hi}", "cells": int(m.sum())}
            for label in out:
                b = float(np.sqrt((out[label]["before"][m] ** 2).mean()))
                a = float(np.sqrt((out[label]["after"][m] ** 2).mean()))
                first.setdefault(label, b)
                row[f"{label}_before_Pa"] = round(b, 1)
                row[f"{label}_after_Pa"] = round(a, 1)
                row[f"{label}_vs_lowest"] = round(b / first[label], 2)
                line += f"{b:>13.0f} Pa{a:>13.0f} Pa"
            rows.append(row)
            print(line)
        print("  高山/最低帶倍數：" + "   ".join(
            f"{label} {rows[-1][f'{label}_vs_lowest']:.2f}x" for label in out))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"year": YEAR, "channel": CHANNEL, "west_of": WEST_OF,
               "rows": rows}, open(OUT_JSON, "w"), indent=2)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}\n      {OUT_JSON}")


if __name__ == "__main__":
    main()
