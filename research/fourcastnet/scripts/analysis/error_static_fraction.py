"""S1: how much of CRA5's error is static, and does that predict what a CNN removes?

The terrain claim so far rests on error binned by elevation, which is confounded:
in CONUS every high cell is also a western cell, so "error grows with elevation"
and "error grows toward the jet" fit the same numbers. This measurement is not
confounded, because it asks a different question -- not WHERE the error is, but
whether it is the SAME error every frame.

Split each channel's error field into a time-mean part and a time-varying part:

    static fraction = mean(ē²) / mean(e²)      ē = error averaged over all frames

A static error is by definition predictable from position alone, so it can be
removed without spending a single extra bit. A moving error cannot. That makes
the two explanations give opposite predictions: terrain does not move, so it
should show up as static and be removable; the jet moves, so it should show up as
dynamic and survive. One number per channel decides it.

The sharp version of the claim is the correlation ACROSS channels: if static
fraction predicts CNN gain over all 20, the mechanism is quantified rather than
narrated. Computed on 2017 first, then confirmed on 2018 -- the relationship is
established on the validation year so that nobody can say it was found on test.

    python scripts/analysis/error_static_fraction.py
    -> results/error_static_fraction.{csv,json}
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(os.path.expanduser("~/fourcastnet_us_mpca"))
sys.path.insert(0, str(BASE / "scripts/cnn"))
from model import UNet2  # noqa: E402

CKPT = BASE / "checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/best.pt"
MEM = BASE / "data/memmap_cra5"
STATS = BASE / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
OUT_CSV = BASE / "results/error_static_fraction.csv"
OUT_JSON = BASE / "results/error_static_fraction.json"
YEARS = [2017, 2018]           # 2017 establishes it, 2018 confirms it
BATCH = 16
# derived from the 37-level q integral rather than compressed as a channel, so its
# improvement is partly self-repair; kept in the table, dropped from the fit.
DERIVED = "tcwv"


def moments(year, net, device, names):
    """Streaming sums of the error field, before and after the network."""
    x = np.load(MEM / f"input_{year}.npy", mmap_mode="r")
    y = np.load(MEM / f"target_{year}.npy", mmap_mode="r")
    n_time, n_ch, n_lat, n_lon = x.shape
    s_b = np.zeros((n_ch, n_lat, n_lon))      # sum of error
    s_a = np.zeros((n_ch, n_lat, n_lon))
    q_b = np.zeros((n_ch, n_lat, n_lon))      # sum of squared error
    q_a = np.zeros((n_ch, n_lat, n_lon))
    with torch.no_grad():
        for t0 in range(0, n_time, BATCH):
            t1 = min(t0 + BATCH, n_time)
            xb = torch.from_numpy(np.asarray(x[t0:t1], dtype=np.float32)).to(device)
            yb = np.asarray(y[t0:t1], dtype=np.float64)
            pred = net(xb).float().cpu().numpy().astype(np.float64)
            eb = -yb                 # recon - truth, in normalised units
            ea = pred - yb           # corrected - truth
            s_b += eb.sum(axis=0); q_b += (eb ** 2).sum(axis=0)
            s_a += ea.sum(axis=0); q_a += (ea ** 2).sum(axis=0)
    out = {}
    for i, v in enumerate(names):
        mb, ma = s_b[i] / n_time, s_a[i] / n_time          # time-mean error field
        pb, pa = q_b[i].mean() / n_time, q_a[i].mean() / n_time   # total power
        sb, sa = (mb ** 2).mean(), (ma ** 2).mean()        # static power
        out[v] = {"static_frac_before": float(sb / pb),
                  "static_frac_after": float(sa / pa),
                  "nrmse_before": float(np.sqrt(pb)),
                  "nrmse_after": float(np.sqrt(pa)),
                  "gain_pct": float(100 * (1 - np.sqrt(pa / pb))),
                  # of the error power removed, how much was the static part
                  "removed_static_share": float((sb - sa) / (pb - pa))
                  if pb > pa else float("nan")}
    return out


def main():
    st = np.load(STATS)
    names = [str(v) for v in st["channel_names"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet2(20, 64, kernel_size=3, depth=2, bottleneck_channels=128).to(device)
    net.load_state_dict(torch.load(CKPT, map_location=device))
    net.eval()

    res = {}
    for yr in YEARS:
        res[yr] = moments(yr, net, device, names)
        print(f"{yr} done")

    fit = [v for v in names if v != DERIVED]
    report = []
    print(f"\n{'chan':<7s}" + "".join(f"{y}靜態%  {y}修掉%  " for y in YEARS)
          + "靜態占修掉%")
    for v in names:
        row = {"channel": v}
        for yr in YEARS:
            r = res[yr][v]
            row[f"static_{yr}"] = round(100 * r["static_frac_before"], 2)
            row[f"static_after_{yr}"] = round(100 * r["static_frac_after"], 2)
            row[f"gain_{yr}"] = round(r["gain_pct"], 2)
            row[f"removed_static_share_{yr}"] = round(
                100 * r["removed_static_share"], 1)
        report.append(row)
        mark = "  (排除於擬合)" if v == DERIVED else ""
        print(f"{v:<7s}" + "".join(
            f"{row[f'static_{y}']:8.2f}{row[f'gain_{y}']:9.2f}  " for y in YEARS)
            + f"{row[f'removed_static_share_{YEARS[-1]}']:9.1f}" + mark)

    print()
    corr = {}
    for yr in YEARS:
        a = np.array([100 * res[yr][v]["static_frac_before"] for v in fit])
        b = np.array([res[yr][v]["gain_pct"] for v in fit])
        r = float(np.corrcoef(a, b)[0, 1])
        # rank correlation too: the relationship need not be linear to be real
        ra, rb = a.argsort().argsort(), b.argsort().argsort()
        rho = float(np.corrcoef(ra, rb)[0, 1])
        corr[yr] = {"pearson": r, "spearman": rho, "n": len(fit)}
        print(f"{yr}  靜態比例 vs CNN 修掉比例   r = {r:+.4f}   "
              f"rho = {rho:+.4f}   (n={len(fit)}, 排除 {DERIVED})")

    sb = np.mean([res[YEARS[-1]][v]["static_frac_before"] for v in fit])
    sa = np.mean([res[YEARS[-1]][v]["static_frac_after"] for v in fit])
    print(f"\n{YEARS[-1]} 平均靜態比例   CNN 前 {100*sb:.2f}%  →  CNN 後 {100*sa:.2f}%")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"years": YEARS, "excluded_from_fit": DERIVED,
               "correlation": {str(k): v for k, v in corr.items()},
               "channels": report}, open(OUT_JSON, "w"), indent=2)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    print(f"\nwrote {OUT_CSV}\n      {OUT_JSON}")


if __name__ == "__main__":
    main()
