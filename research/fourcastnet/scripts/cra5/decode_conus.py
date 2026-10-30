"""Decode CRA5 .bin -> FourCastNet 20-channel CONUS arrays, one .npy per year.

Region : 49N..25N, 130W..70W  -> rows 164:261, cols 920:1161  (97 x 241)
Output : {OUT}/{year}.npy  float32 (T, 20, 97, 241)
         {OUT}/{year}_timestamps.json
         {OUT}/meta.json
"""
import os, sys, json, time, argparse, glob
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cra5.api import cra5_api
import sdpa_patch

ROOT = os.path.expanduser("~/fourcastnet_us_mpca/cra5")
OUT = os.path.expanduser("~/fourcastnet_us_mpca/cra5_conus")

R0, R1 = 164, 261          # rows  49.00N .. 25.00N  (python slice, end-exclusive)
C0, C1 = 920, 1161         # cols 130.00W .. 70.00W
H, W = R1 - R0, C1 - C0

G = 9.80665

# FourCastNet channel order. tcwv is not in CRA5 and is derived from q.
FCN20 = ["u10", "v10", "t2m", "sp", "msl",
         "t_850", "u_1000", "v_1000", "z_1000",
         "u_850", "v_850", "z_850",
         "u_500", "v_500", "z_500", "t_500",
         "z_50", "r_500", "r_850", "tcwv"]


def build_index(api):
    name_to_idx = {v: k for k, v in api.channels_to_vname.items()}
    direct = {}
    for n in FCN20:
        if n != "tcwv":
            direct[n] = name_to_idx[n]
    # q occupies one contiguous block of 37 levels
    q_idx = [name_to_idx[f"q_{int(l)}" if float(l).is_integer() else f"q_{l}"]
             for l in api.cfg.pressure_level]
    return name_to_idx, direct, q_idx


def make_tcwv(x, q_idx, sp_idx, p_pa_asc, q_order_asc):
    """TCWV = (1/g) * integral of q dp, clipped at the surface.

    x        : (268, H, W) de-normalized tensor on GPU
    p_pa_asc : (37,) pressure in Pa, ascending (top of atmosphere first)
    Levels below ground are handled by clamping p to sp, which drives their
    layer thickness to zero instead of letting them add spurious moisture.
    """
    q = x[q_order_asc]                       # (37, H, W) ascending in pressure
    sp = x[sp_idx]                           # (H, W) Pa
    p = p_pa_asc[:, None, None].expand_as(q)
    p_eff = torch.clamp(p, max=sp[None, :, :])
    dp = p_eff[1:] - p_eff[:-1]              # zero where fully below ground
    return (0.5 * (q[1:] + q[:-1]) * dp).sum(0) / G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", default=["2014", "2015", "2016", "2017", "2018"])
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only first N frames per year")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    api = cra5_api(local_root=ROOT)
    print(sdpa_patch.apply(), flush=True)

    name_to_idx, direct, q_idx = build_index(api)
    levels = [float(l) for l in api.cfg.pressure_level]          # 1000 -> 1 hPa

    order = np.argsort(levels)                                    # ascending pressure
    p_pa_asc = torch.tensor([levels[i] * 100.0 for i in order],
                            dtype=torch.float32, device=api.device)
    q_order_asc = [q_idx[i] for i in order]
    sp_idx = name_to_idx["sp"]

    take = [direct[n] for n in FCN20 if n != "tcwv"]              # 19 channels, FCN order

    if args.validate_only:
        ts = "2014-01-01T00:00:00"
        out = api.decode_from_bin(time_stamp=ts, return_format="de_normalized")
        x = out["x_hat"]
        tcwv = make_tcwv(x, q_idx, sp_idx, p_pa_asc, q_order_asc)
        t = tcwv.cpu().numpy()
        print(f"\n=== tcwv validation on {ts} ===")
        print("global  min/mean/max: %.2f %.2f %.2f kg/m2  (ERA5 global mean ~25)"
              % (t.min(), t.mean(), t.max()))
        c = t[R0:R1, C0:C1]
        print("CONUS   min/mean/max: %.2f %.2f %.2f kg/m2  (Jan CONUS ~5-30)"
              % (c.min(), c.mean(), c.max()))
        trop = t[int((90 - 5) / 0.25):int((90 + 5) / 0.25)]
        print("tropics mean: %.2f kg/m2  (expect ~45-50)" % trop.mean())
        print("negatives:", int((t < 0).sum()))
        return

    meta = {
        "channels": FCN20,
        "cra5_index": {n: direct.get(n, "derived from q") for n in FCN20},
        "region": {"lat_north": 49.0, "lat_south": 25.0,
                   "lon_west": -130.0, "lon_east": -70.0,
                   "rows": [R0, R1], "cols": [C0, C1]},
        "lat": (90.0 - 0.25 * np.arange(R0, R1)).tolist(),
        "lon": (0.25 * np.arange(C0, C1) - 360.0).tolist(),
        "shape": ["time", 20, H, W],
        "dtype": "float32",
        "tcwv": "derived: (1/g) * integral q dp, pressure clipped at sp",
        "note": "decode_from_bin default return_format is misspelled upstream; "
                "'de_normalized' passed explicitly",
    }
    with open(f"{OUT}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    for year in args.years:
        files = sorted(glob.glob(f"{ROOT}/CRA5/{year}/*.bin"))
        if args.limit:
            files = files[:args.limit]
        T = len(files)
        stamps = [os.path.basename(f)[:-4] for f in files]
        print(f"\n=== {year}: {T} frames -> {OUT}/{year}.npy ===", flush=True)

        arr = np.lib.format.open_memmap(f"{OUT}/{year}.npy", mode="w+",
                                        dtype=np.float32, shape=(T, 20, H, W))
        t_start = time.time()
        for i, (path, ts) in enumerate(zip(files, stamps)):
            out = api.decode_from_bin(custom_path=path, return_format="de_normalized")
            x = out["x_hat"]
            sel = x[take][:, R0:R1, C0:C1]
            tcwv = make_tcwv(x, q_idx, sp_idx, p_pa_asc, q_order_asc)[R0:R1, C0:C1]
            arr[i, :19] = sel.cpu().numpy()
            arr[i, 19] = tcwv.cpu().numpy()

            if i % 100 == 0 or i == T - 1:
                el = time.time() - t_start
                rate = (i + 1) / el
                print(f"  [{year}] {i+1}/{T}  {rate:.2f} fps  "
                      f"eta {(T-i-1)/rate/60:.1f} min", flush=True)

        arr.flush()
        del arr
        with open(f"{OUT}/{year}_timestamps.json", "w") as f:
            json.dump(stamps, f)
        print(f"  [{year}] done in {(time.time()-t_start)/60:.1f} min", flush=True)

    print("\nALL DONE")


if __name__ == "__main__":
    main()
