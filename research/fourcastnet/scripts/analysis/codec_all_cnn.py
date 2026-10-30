"""Run the CNN for real on every quantised energy level, not just e995.

codec_all.py measures bytes honestly but fills its post-CNN column with the
unquantised champion-U-Net numbers, because quant_cnn_eval.py measured the
quantisation penalty at e995 (+0.07%) and it was assumed to carry to e90..e99.
Four of the five points on that curve are therefore extrapolated, and a curve with
four assumed points should not be drawn as if it were measured. This measures them.

No retraining -- the per-energy champion weights are loaded as-is and run on
reconstructions decoded from quantised scores. That is also the deployment story:
the coder changes how coefficients are stored, not what the model expects.

Self-validating: for each energy a budget-0 (unquantised) control row is computed
first and must reproduce the published post-CNN value. If it does not, the wrong
checkpoint was picked and the run aborts rather than writing a plausible wrong
number. The two candidate checkpoint paths are tried in order for that reason --
these runs predate the test_nrmse field, so there is nothing to read it from.

    python scripts/analysis/codec_all_cnn.py
    -> results/pca_codec_all_cnn.json
"""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import zarr

BASE = Path(os.path.expanduser("~/fourcastnet_us_mpca"))
sys.path.insert(0, str(BASE / "scripts/cnn"))
from model import UNet2  # noqa: E402

MPCA_DIR = BASE / "mpca_train_basis_bundle/models/mpca_spatial_99energy"
ORIG = BASE / "data/processed/fourcastnet20_conus"
STATS = BASE / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
OUT = BASE / "results/pca_codec_all_cnn.json"
RAW = 467540
YEAR = 2018
BUDGET = 0.01                     # the 1% core-space budget codec_all.py codes at
TOL = 0.005                       # control row must land within 0.5% of published
POST_CNN = {"e90": 0.27882, "e95": 0.21772, "e97": 0.18828,
            "e99": 0.15257, "e995": 0.13964}

stats = np.load(STATS)
names = [str(x) for x in stats["channel_names"]]
mean_c = stats["mean"].astype(np.float32)[:, None, None]
std_c = stats["std"].astype(np.float32)[:, None, None]
std_ref = stats["std"].astype(np.float64)
dev = "cuda" if torch.cuda.is_available() else "cpu"

T = np.asarray(zarr.open_group(str(ORIG / f"{YEAR}.zarr"), mode="r")["fields"][:],
               dtype=np.float64)

mp = {}
for v in names:
    d = np.load(MPCA_DIR / f"{v}.npz")
    mp[v] = (float(d["mean"]), float(d["std"]),
             d["lat_components"].astype(np.float32),
             d["lon_components"].astype(np.float32))


def nrmse(R):
    return float((np.sqrt(((R.astype(np.float64) - T) ** 2).mean(axis=(0, 2, 3)))
                  / std_ref).mean())


def load_net(energy):
    """Pick whichever of the two layouts holds the champion weights."""
    for p in (BASE / f"checkpoints/{energy}/unet_d2_c64_lr3e-4_clip1/best.pt",
              BASE / f"checkpoints/{energy}/best.pt"):
        if not p.exists():
            continue
        net = UNet2(20, 64, 3, 2, 128).to(dev).eval()
        try:
            net.load_state_dict(torch.load(p, map_location=dev))
        except RuntimeError:
            continue
        yield p, net


def run(energy):
    d = BASE / "second_stage_pca" / energy
    pca = joblib.load(d / "pca_model.joblib")
    meta = json.load(open(d / "metadata.json"))
    assert meta["variable_order"] == names
    shapes = {v: tuple(s) for v, s in meta["variable_core_shapes"].items()}
    cum = [0]
    for v in names:
        cum.append(cum[-1] + int(np.prod(shapes[v])))
    ev = pca.explained_variance_.astype(np.float64)
    n = len(ev)
    trunc = ev.sum() / pca.explained_variance_ratio_.sum() - ev.sum()
    S = np.load(d / f"scores_{YEAR}.npy").astype(np.float64)

    def decode(scores):
        Z = pca.inverse_transform(scores).astype(np.float32)
        out = np.empty((Z.shape[0], 20, T.shape[2], T.shape[3]), dtype=np.float32)
        for i, v in enumerate(names):
            lr, lo = shapes[v]
            cores = Z[:, cum[i]:cum[i + 1]].reshape(-1, lr, lo)
            m, s, L, R = mp[v]
            out[:, i] = np.einsum("ha,tab,bw->thw", L, cores, R.T,
                                  optimize=True) * s + m
        return out

    @torch.no_grad()
    def refine(net, recon, bs=32):
        out = np.empty_like(recon)
        for s0 in range(0, len(recon), bs):
            x = recon[s0:s0 + bs]
            xn = torch.from_numpy((x - mean_c) / std_c).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(dev == "cuda")):
                delta = net(xn)
            out[s0:s0 + bs] = x + delta.float().cpu().numpy() * std_c
        return out

    R0 = decode(S)                              # unquantised control
    for path, net in load_net(energy):
        post0 = nrmse(refine(net, R0))
        rel = post0 / POST_CNN[energy] - 1
        print(f"  control {path.parent.name}/best.pt -> {post0:.5f} "
              f"(published {POST_CNN[energy]:.5f}, {100*rel:+.2f}%)")
        if abs(rel) <= TOL:
            break
    else:
        raise SystemExit(f"{energy}: no checkpoint reproduces the published value")

    D = np.sqrt(12.0 * BUDGET * trunc / n)
    Rq = decode(np.rint(S / D) * D)             # quantised, same step as codec_all
    pre, post = nrmse(Rq), nrmse(refine(net, Rq))
    print(f"  quantised D={D:.5f}  CNN前 {pre:.5f}  CNN後 {post:.5f}  "
          f"vs 未量化 {100*(post/post0-1):+.3f}%")
    return {"energy": energy, "n_components": n, "quant_step": float(D),
            "checkpoint": str(path.relative_to(BASE)),
            "post_cnn_unquantised": post0, "post_cnn_published": POST_CNN[energy],
            "pre_cnn": pre, "post_cnn": post,
            "quantisation_penalty_pct": 100 * (post / post0 - 1)}


rows = []
for e in ("e995", "e99", "e97", "e95", "e90"):
    print(f"=== {e}")
    rows.append(run(e))

print(f"\n{'energy':7s}{'n':>7s}{'CNN前':>10s}{'CNN後':>10s}{'量化代價':>10s}")
for r in rows:
    print(f"{r['energy']:7s}{r['n_components']:7d}{r['pre_cnn']:10.5f}"
          f"{r['post_cnn']:10.5f}{r['quantisation_penalty_pct']:+9.3f}%")
json.dump(rows, open(OUT, "w"), indent=2, default=float)
print(f"wrote {OUT}")
