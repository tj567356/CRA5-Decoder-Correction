import os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cra5.api import cra5_api
import sdpa_patch; sdpa_patch.apply()

ROOT = os.path.expanduser("~/fourcastnet_us_mpca/cra5")
TS = "2014-01-01T00:00:00"

api = cra5_api(local_root=ROOT)
names = api.channels_to_vname
name_to_idx = {v: k for k, v in names.items()}

# FourCastNet (Pathak et al. 2022) 20-channel set, in the paper's order.
FCN20 = ["u10", "v10", "t2m", "sp", "msl",
         "t_850", "u_1000", "v_1000", "z_1000",
         "u_850", "v_850", "z_850",
         "u_500", "v_500", "z_500", "t_500",
         "z_50", "r_500", "r_850", "tcwv"]

print("=== FourCastNet 20 channels -> CRA5 index ===")
missing = []
for i, n in enumerate(FCN20, 1):
    idx = name_to_idx.get(n)
    if idx is None:
        missing.append(n)
        print(f"{i:2d}. {n:10s} -> NOT IN CRA5")
    else:
        print(f"{i:2d}. {n:10s} -> ch {idx}")
print("missing:", missing)

print()
print("=== all 9 single-level channels CRA5 does have ===")
print([names[i] for i in range(259, 268)])

# --- grid orientation check -------------------------------------------------
out = api.decode_from_bin(time_stamp=TS, return_format="de_normalized")
x = out["x_hat"].cpu().float().numpy()
t2m = x[name_to_idx["t2m"]]
print()
print("t2m shape:", t2m.shape)

def at(lat, lon):
    """assume ERA5 layout: lat 90..-90 (row 0 = North Pole), lon 0..359.75"""
    r = int(round((90.0 - lat) / 0.25))
    c = int(round((lon % 360) / 0.25))
    return t2m[r, c]

print("--- assuming lat[0]=90N, lon[0]=0E (Jan 1, 00Z) ---")
for label, lat, lon, expect in [
    ("North Pole   ", 89.0,   0.0, "~245 K (Arctic winter)"),
    ("South Pole   ", -89.0,  0.0, "~250 K (Antarctic summer)"),
    ("Sahara       ", 22.0,  15.0, "~285 K (night, desert)"),
    ("Taiwan       ", 24.0, 121.0, "~290 K"),
    ("Siberia      ", 65.0, 100.0, "~235 K (very cold)"),
    ("Amazon       ", -3.0, 300.0, "~297 K"),
]:
    print(f"{label} ({lat:6.1f},{lon:6.1f}) = {at(lat,lon):7.2f} K   expect {expect}")

# --- CONUS crop -------------------------------------------------------------
LAT_N, LAT_S = 49.0, 25.0
LON_W, LON_E = -130.0, -70.0
r0 = int(round((90.0 - LAT_N) / 0.25))
r1 = int(round((90.0 - LAT_S) / 0.25))
c0 = int(round((LON_W % 360) / 0.25))
c1 = int(round((LON_E % 360) / 0.25))
print()
print(f"=== CONUS crop: rows {r0}..{r1} ({r1-r0+1}), cols {c0}..{c1} ({c1-c0+1}) ===")
crop = t2m[r0:r1+1, c0:c1+1]
print("crop shape:", crop.shape)
print("crop t2m min/mean/max: %.1f %.1f %.1f K" % (crop.min(), crop.mean(), crop.max()))
print("corners (NW, NE, SW, SE): %.1f %.1f %.1f %.1f" % (
    crop[0,0], crop[0,-1], crop[-1,0], crop[-1,-1]))

nbytes = 20 * crop.shape[0] * crop.shape[1] * 4
print()
print("per-frame (20ch, float32): %.2f MB" % (nbytes/1e6))
print("all 7300 frames: %.1f GB" % (nbytes*7300/1e9))
