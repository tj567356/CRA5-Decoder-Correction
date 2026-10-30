"""Does CRA5's error track terrain, and is that the part the CNN removes?

why_sp.py established the relationship for surface pressure alone (+0.570 with
elevation, 3.9x error over mountains versus plains). One channel is not enough to
claim it: DeepMind's compressor already feeds elevation and a land/sea mask to its
encoder but never checks whether that changes anything -- that unexamined gap is
what this is meant to fill, and filling it with a single channel does not fill it.

So: every channel, error stratified by elevation, before AND after the correction
network. The third column is the one that matters. If the network's gain is
concentrated over high ground, the terrain-correlated part of the error is
demonstrably the removable part, which is a stronger claim than either the
correlation or the correction alone.

Elevation is derived from the time-mean surface pressure through the barometric
formula, the same proxy why_sp.py used. It is not a survey; what matters is that
the ranking of grid cells is right, and hydrostatic balance guarantees that.

    python scripts/analysis/error_vs_elevation.py
    -> results/error_vs_elevation.csv, .json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(os.path.expanduser("~/fourcastnet_us_mpca"))
sys.path.insert(0, str(BASE / "scripts/cnn"))
from model import UNet2

CKPT = BASE / "checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/best.pt"
MEM = BASE / "data/memmap_cra5"
STATS = BASE / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
OUT_CSV = BASE / "results/error_vs_elevation.csv"
OUT_JSON = BASE / "results/error_vs_elevation.json"
YEAR = 2018
BATCH = 16
BANDS = [(0, 200), (200, 600), (600, 1200), (1200, 2000), (2000, 9999)]


def load_weights(model, path, device):
    """The CRA5 checkpoint predates the UNet2 refactor on some trees."""
    sd = torch.load(path, map_location=device)
    try:
        model.load_state_dict(sd)
        return
    except RuntimeError:
        pass
    depth, out = 2, {}
    for k, v in sd.items():
        head = k.split(".", 1)[0]
        rest = k[len(head):]
        new = head
        for pre, lst in (("enc", "encs"), ("down", "downs")):
            if head.startswith(pre) and head[len(pre):].isdigit():
                new = f"{lst}.{int(head[len(pre):]) - 1}"
        for pre, lst in (("up", "ups"), ("dec", "decs")):
            if head.startswith(pre) and head[len(pre):].isdigit():
                new = f"{lst}.{depth - int(head[len(pre):])}"
        out[new + rest] = v
    model.load_state_dict(out)


def main():
    st = np.load(STATS)
    names = [str(x) for x in st["channel_names"]]
    std = st["std"].astype(np.float64)
    mean = st["mean"].astype(np.float64)

    x = np.load(MEM / f"input_{YEAR}.npy", mmap_mode="r")     # (recon-mean)/std
    y = np.load(MEM / f"target_{YEAR}.npy", mmap_mode="r")    # (orig-recon)/std
    n_time, n_ch, n_lat, n_lon = x.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet2(20, 64, kernel_size=3, depth=2, bottleneck_channels=128).to(device)
    load_weights(model, CKPT, device)
    model.eval()

    # accumulate squared error per grid cell, before and after correction
    sq_before = np.zeros((n_ch, n_lat, n_lon), dtype=np.float64)
    sq_after = np.zeros((n_ch, n_lat, n_lon), dtype=np.float64)
    sp_sum = np.zeros((n_lat, n_lon), dtype=np.float64)
    sp_i = names.index("sp")

    with torch.no_grad():
        for t0 in range(0, n_time, BATCH):
            t1 = min(t0 + BATCH, n_time)
            xb = torch.from_numpy(np.asarray(x[t0:t1], dtype=np.float32)).to(device)
            yb = np.asarray(y[t0:t1], dtype=np.float64)
            pred = model(xb).float().cpu().numpy().astype(np.float64)
            # error in normalised units: before = -target, after = pred - target
            sq_before += (yb ** 2).sum(axis=0)
            sq_after += ((pred - yb) ** 2).sum(axis=0)
            recon = np.asarray(x[t0:t1, sp_i], dtype=np.float64) * std[sp_i] + mean[sp_i]
            sp_sum += (recon + yb[:, sp_i] * std[sp_i]).sum(axis=0)   # truth, not recon
    rmse_before = np.sqrt(sq_before / n_time)      # normalised, per grid cell
    rmse_after = np.sqrt(sq_after / n_time)

    # sanity: whole-field nRMSE should reproduce the published test numbers
    whole_b = np.sqrt((rmse_before ** 2).mean(axis=(1, 2)))
    whole_a = np.sqrt((rmse_after ** 2).mean(axis=(1, 2)))
    print(f"check: 20-channel mean nRMSE  before {whole_b.mean():.5f}"
          f"  after {whole_a.mean():.5f}"
          f"  ({100 * (whole_a.mean() - whole_b.mean()) / whole_b.mean():+.1f}%)")
    res = BASE / "checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/test_results.json"
    if res.exists():
        pub = json.load(open(res)).get("test_nrmse")
        if pub:
            print(f"       published test_nrmse = {pub:.5f}"
                  f"  (delta {abs(pub - whole_a.mean()):.2e})")

    # elevation from time-mean surface pressure, barometric formula
    sp_mean = sp_sum / n_time
    elev = 44330.0 * (1.0 - (sp_mean / 101325.0) ** (1.0 / 5.255))
    elev = np.clip(elev, 0, None)
    print(f"elevation proxy: {elev.min():.0f}–{elev.max():.0f} m, "
          f"median {np.median(elev):.0f} m")
    counts = [int(((elev >= lo) & (elev < hi)).sum()) for lo, hi in BANDS]
    print("cells per band:", dict(zip([f"{lo}-{hi}" for lo, hi in BANDS], counts)))
    print()

    rows = []
    labels = [f"{lo}-{hi}m" if hi < 9999 else f"{lo}m+" for lo, hi in BANDS]
    hdr = (f"{'chan':<7s}{'corr':>7s}{'gain%':>8s}  " +
           "".join(f"{s:<9s}" for s in labels))
    print("每格 nRMSE 按海拔分層（上：CNN 前，下：CNN 後）")
    print(hdr)
    print("-" * len(hdr))
    for i, v in enumerate(names):
        b, a = rmse_before[i], rmse_after[i]
        corr = float(np.corrcoef(b.ravel(), elev.ravel())[0, 1])
        gain = 100 * (a.mean() - b.mean()) / b.mean()
        band_b, band_a = [], []
        for lo, hi in BANDS:
            m = (elev >= lo) & (elev < hi)
            band_b.append(float(np.sqrt((b[m] ** 2).mean())) if m.any() else None)
            band_a.append(float(np.sqrt((a[m] ** 2).mean())) if m.any() else None)
        ratio = (band_b[-1] / band_b[0]) if band_b[0] else None
        rows.append({"channel": v, "corr_error_elevation": round(corr, 4),
                     "cnn_gain_pct": round(gain, 1),
                     "mountain_over_plain": round(ratio, 2) if ratio else None,
                     "before": [round(z, 5) for z in band_b],
                     "after": [round(z, 5) for z in band_a],
                     "band_gain_pct": [round(100 * (aa - bb) / bb, 1)
                                       for bb, aa in zip(band_b, band_a)]})
        print(f"{v:<7s}{corr:>7.3f}{gain:>8.1f}  " +
              "".join(f"{z:<9.4f}" for z in band_b))
        print(f"{'':<7s}{'':>7s}{'':>8s}  " +
              "".join(f"{z:<9.4f}" for z in band_a))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"year": YEAR, "bands": [list(b) for b in BANDS],
               "cells_per_band": counts,
               "elevation_proxy": "barometric formula on time-mean sp",
               "channels": rows}, open(OUT_JSON, "w"), indent=2)
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "corr_error_elevation", "cnn_gain_pct",
                    "mountain_over_plain"]
                   + [f"before_{lo}_{hi}" for lo, hi in BANDS]
                   + [f"after_{lo}_{hi}" for lo, hi in BANDS]
                   + [f"gain_{lo}_{hi}" for lo, hi in BANDS])
        for r in rows:
            w.writerow([r["channel"], r["corr_error_elevation"], r["cnn_gain_pct"],
                        r["mountain_over_plain"]]
                       + r["before"] + r["after"] + r["band_gain_pct"])
    print(f"\nSaved {OUT_CSV}\n      {OUT_JSON}")


if __name__ == "__main__":
    main()
