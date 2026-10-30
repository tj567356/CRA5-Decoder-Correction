"""Verification 2b: is CRA5's residual predictable from the previous timestep?

This decides whether a multi-frame CNN input is worth building. The case for it was
that CRA5 is documented as reconstructing each state "without conditioning on
adjacent timesteps", so temporal redundancy is entirely unexploited -- and the
largest surviving residual after post-processing is r500 (0.1605), a field of
moving moisture plumes that a single frame cannot place.

The case against is what the PCA scores just showed: at 6-hourly sampling the
median AR(1) coefficient of the stage-2 coefficients is 0.137, i.e. the small
scales have already decorrelated, and temporal prediction bought only 2%. If the
CRA5 residual behaves the same way, multi-frame is a dead end and should not be
built.

Measured per channel, 2018:

  lag-1 autocorrelation of the residual field, pooled over grid points
  the same for lag 2 and 4 (12 h, 24 h) to see the decay
  and, as the ceiling on what a temporal model could remove:
  the variance ratio of (e_t - a*e_{t-1}) to e_t at the optimal a

A high lag-1 value means a previous-frame-conditioned model has something to work
with. Near zero means the residual is temporally white and multi-frame cannot help.
"""
import json
import os
from pathlib import Path

import numpy as np
import zarr

BASE = Path(os.path.expanduser("~/fourcastnet_us_mpca"))
V = ['u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z1000', 'u850',
     'v850', 'z850', 'u500', 'v500', 'z500', 't500', 'z50', 'r500', 'r850', 'tcwv']
YEAR = 2018
STEP = 1          # consecutive frames: 6 h apart
POST_GAIN = {'u10': -36.5, 'v10': -34.0, 't2m': -31.4, 'sp': -74.9, 'msl': -14.1,
             't850': -12.3, 'u1000': -40.5, 'v1000': -40.7, 'z1000': -15.5,
             'u850': -20.1, 'v850': -19.6, 'z850': -3.4, 'u500': -0.0, 'v500': -0.0,
             'z500': -0.4, 't500': -0.1, 'z50': -0.9, 'r500': -7.4, 'r850': -15.1,
             'tcwv': -59.6}

os.chdir(BASE)
truth = zarr.open_group(f"data/processed/fourcastnet20_conus/{YEAR}.zarr",
                        mode="r")["fields"]
cra5 = np.load(f"cra5_conus/{YEAR}.npy", mmap_mode="r")

print(f"{YEAR}: {truth.shape[0]} frames, 6-hourly\n")
print(f"{'chan':7s} {'lag1':>7s} {'lag2':>7s} {'lag4':>7s} "
      f"{'殘差變異數可去除':>16s} {'CNN已修':>9s}")

rows = []
for i, v in enumerate(V):
    # one channel at a time keeps this to ~135 MB per array
    t = np.asarray(truth[:, i], dtype=np.float64)
    c = np.asarray(cra5[:, i], dtype=np.float64)
    e = (c - t).reshape(len(t), -1)
    e -= e.mean(axis=0, keepdims=True)      # remove the static bias map first:
    # otherwise a constant offset shows up as near-perfect autocorrelation and
    # says nothing about whether the *time-varying* part is predictable -- and the
    # static part is exactly what the single-frame CNN already removes.
    var = float((e ** 2).mean())

    def ac(lag):
        a, b = e[lag:], e[:-lag]
        return float((a * b).mean() / var)

    r1, r2, r4 = ac(1), ac(2), ac(4)
    # best linear one-step predictor: residual variance ratio 1 - r1^2
    removable = 100 * (r1 ** 2)
    rows.append({"chan": v, "lag1": r1, "lag2": r2, "lag4": r4,
                 "removable_var_pct": removable, "cnn_gain_pct": POST_GAIN[v]})
    print(f"{v:7s} {r1:7.3f} {r2:7.3f} {r4:7.3f} {removable:15.1f}% "
          f"{POST_GAIN[v]:8.1f}%")
    del t, c, e

a = np.array([[r["lag1"], r["cnn_gain_pct"]] for r in rows])
print(f"\nlag-1 中位數 {np.median(a[:,0]):.3f}   最大 {a[:,0].max():.3f} "
      f"({rows[int(np.argmax(a[:,0]))]['chan']})")
print(f"correlation( lag-1 , CNN 已修掉的量 ) = {np.corrcoef(a[:,0], -a[:,1])[0,1]:+.3f}")
print("\n對照：stage-2 PCA 係數的 AR(1) 中位數是 0.137（6 小時間隔）")

json.dump(rows, open("results/cra5_temporal.json", "w"), indent=2, default=float)
print("wrote results/cra5_temporal.json")
