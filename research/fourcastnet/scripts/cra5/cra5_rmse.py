"""CRA5 baseline: RMSE of the CRA5-decoded CONUS fields against ERA5 truth.

Mirrors the metric layout of decoded/true_rmse_results.json so the CRA5 point
can be placed alongside the e90..e995 PCA curve.

Outputs results/cra5_rmse.json with, per year and per variable:
    rmse   -- physical units (K, Pa, m^2 s^-2, %, kg m^-2, m s^-1)
    bias   -- mean(cra5 - era5), signed
    nrmse  -- rmse / train-period channel std   (the comparable number)

Variable names follow the ERA5 zarr convention (t850, not t_850).
"""
import os, json, time
import numpy as np
import zarr

os.chdir(os.path.expanduser("~/fourcastnet_us_mpca"))

YEARS = [2014, 2015, 2016, 2017, 2018]
CHUNK = 64
OUT = "results/cra5_rmse.json"

stats = np.load("mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz")
print("stats keys:", list(stats.keys()))
std = stats["std"].astype(np.float64)
mean = stats["mean"].astype(np.float64)

results = {}
for year in YEARS:
    a = np.load(f"cra5_conus/{year}.npy", mmap_mode="r")
    z = zarr.open(f"data/processed/fourcastnet20_conus/{year}.zarr", mode="r")
    f = z["fields"]
    names = list(z.attrs["channel_names"])          # canonical naming
    assert a.shape == f.shape, f"{year}: {a.shape} vs {f.shape}"
    T, C, H, W = a.shape
    assert len(std) == C, f"stats has {len(std)} channels, data has {C}"

    sse = np.zeros(C, dtype=np.float64)             # sum of squared error
    serr = np.zeros(C, dtype=np.float64)            # sum of signed error
    n = 0
    t0 = time.time()
    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        d = a[s:e].astype(np.float64) - np.asarray(f[s:e], dtype=np.float64)
        sse += (d ** 2).sum(axis=(0, 2, 3))
        serr += d.sum(axis=(0, 2, 3))
        n += (e - s) * H * W
        if s % (CHUNK * 8) == 0:
            el = time.time() - t0
            print(f"  [{year}] {e}/{T}  {el:.0f}s elapsed", flush=True)
        del d

    rmse = np.sqrt(sse / n)
    bias = serr / n
    nrmse = rmse / std

    results[str(year)] = {
        "per_variable": {
            names[i]: {"rmse": float(rmse[i]),
                       "bias": float(bias[i]),
                       "nrmse": float(nrmse[i])}
            for i in range(C)
        },
        # raw unweighted mean of per-channel RMSE -- same convention as
        # true_rmse_results.json "overall"; unit-mixed, kept only for continuity
        "overall_rmse_rawmix": float(rmse.mean()),
        # the comparable headline: mean over channels of rmse/std
        "overall_nrmse": float(nrmse.mean()),
    }
    print(f"  [{year}] done in {time.time()-t0:.0f}s  "
          f"nRMSE={nrmse.mean():.4f}  rawmix={rmse.mean():.2f}", flush=True)

os.makedirs("results", exist_ok=True)
with open(OUT, "w") as fh:
    json.dump({"source": "CRA5 VAEformer decode, CONUS 25-49N 130-70W, 97x241",
               "truth": "data/processed/fourcastnet20_conus/{year}.zarr",
               "norm_std": "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz",
               "note": "tcwv is derived from the 37-level q integral, not a native "
                       "CRA5 channel; its error mixes compression and integration error",
               "years": results}, fh, indent=2)

print(f"\nwrote {OUT}\n")
names = list(zarr.open(f"data/processed/fourcastnet20_conus/2018.zarr",
                       mode="r").attrs["channel_names"])
print(f"{'chan':7s} {'RMSE 2017':>12s} {'RMSE 2018':>12s} {'nRMSE 17':>9s} {'nRMSE 18':>9s} {'bias 18':>11s}")
for nm in names:
    a17 = results["2017"]["per_variable"][nm]
    a18 = results["2018"]["per_variable"][nm]
    print(f"{nm:7s} {a17['rmse']:12.4f} {a18['rmse']:12.4f} "
          f"{a17['nrmse']:9.4f} {a18['nrmse']:9.4f} {a18['bias']:11.4f}")
print(f"\nmean nRMSE : 2017 {results['2017']['overall_nrmse']:.4f}   "
      f"2018 {results['2018']['overall_nrmse']:.4f}")
print(f"raw-mix RMSE (comparable to summary.csv): "
      f"2017 {results['2017']['overall_rmse_rawmix']:.2f}   "
      f"2018 {results['2018']['overall_rmse_rawmix']:.2f}")
