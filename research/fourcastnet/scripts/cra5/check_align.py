import json, os, numpy as np

os.chdir(os.path.expanduser("~/fourcastnet_us_mpca"))

for y in [2014, 2015, 2016, 2017, 2018]:
    ts = json.load(open(f"cra5_conus/{y}_timestamps.json"))
    a = np.load(f"cra5_conus/{y}.npy", mmap_mode="r")
    b = np.load(f"data/memmap/target_{y}.npy", mmap_mode="r")
    c = np.load(f"data/memmap/input_{y}.npy", mmap_mode="r")
    print(f"{y}  cra5 {a.shape} {a.dtype}  n_ts={len(ts)}  {ts[0]} .. {ts[-1]}")
    print(f"      target {b.shape} {b.dtype}   input {c.shape} {c.dtype}")

print("\n--- per-channel stats, 2018 frame 0 (cra5 vs era5 target) ---")
meta = json.load(open("cra5_conus/meta.json"))
ch = meta["channels"]
a = np.load("cra5_conus/2018.npy", mmap_mode="r")[0].astype(np.float32)
b = np.load("data/memmap/target_2018.npy", mmap_mode="r")[0].astype(np.float32)
print(f"{'chan':8s} {'cra5 mean':>12s} {'era5 mean':>12s} {'cra5 std':>10s} {'era5 std':>10s} {'rmse':>10s}")
for i, n in enumerate(ch):
    r = float(np.sqrt(((a[i] - b[i]) ** 2).mean()))
    print(f"{n:8s} {a[i].mean():12.3f} {b[i].mean():12.3f} {a[i].std():10.3f} {b[i].std():10.3f} {r:10.3f}")

print("\n--- prepare_memmap.py head ---")
print("".join(open("scripts/cnn/prepare_memmap.py").readlines()[:40]))
