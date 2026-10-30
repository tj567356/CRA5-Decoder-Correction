"""Verification 2a: does the trained CNN still work on quantised input?

The rate side is measured (5.43 bits/coefficient) and the pre-CNN distortion cost
of quantisation is measured (+0.05% nRMSE). The post-CNN number at that operating
point is not -- it was inferred by assuming a 0.05% input perturbation does not
matter to the network. That assumption is cheap to check and expensive to get
wrong, because the post-CNN nRMSE is the number the paper reports.

No retraining. The e995 champion weights are loaded as-is and evaluated on
reconstructions decoded from quantised scores. That is also the honest deployment
story: the coder is a change to how coefficients are stored, not to the model, so
the model should not need to know.

Reports the full quantiser sweep so the rate-distortion curve has post-CNN points,
not just pre-CNN ones.
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

PCA_DIR = BASE / "second_stage_pca" / "e995"
MPCA_DIR = BASE / "mpca_train_basis_bundle/models/mpca_spatial_99energy"
CKPT = BASE / "checkpoints/e995/unet_d2_c64_lr3e-4_clip1/best.pt"
ORIG = BASE / "data/processed/fourcastnet20_conus"
STATS = BASE / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
RAW = 467540
YEAR = 2018
TRAIN = [2014, 2015, 2016]
BUDGETS = [0.0, 0.001, 0.01, 0.05, 0.10, 0.25]

stats = np.load(STATS)
names = [str(x) for x in stats["channel_names"]]
mean_c = stats["mean"].astype(np.float32)[:, None, None]
std_c = stats["std"].astype(np.float32)[:, None, None]
std_ref = stats["std"].astype(np.float64)

pca = joblib.load(PCA_DIR / "pca_model.joblib")
meta = json.load(open(PCA_DIR / "metadata.json"))
vorder = meta["variable_order"]
core_shapes = {v: tuple(s) for v, s in meta["variable_core_shapes"].items()}
n_comp = meta["n_components"]
assert vorder == names

mp = {}
for v in vorder:
    d = np.load(MPCA_DIR / f"{v}.npz")
    mp[v] = (float(d["mean"]), float(d["std"]),
             d["lat_components"].astype(np.float32),
             d["lon_components"].astype(np.float32))
cum = [0]
for v in vorder:
    cum.append(cum[-1] + int(np.prod(core_shapes[v])))

ev = pca.explained_variance_.astype(np.float64)
trunc = ev.sum() / pca.explained_variance_ratio_.sum() - ev.sum()

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = UNet2(in_channels=20, mid_channels=64, kernel_size=3, depth=2,
            bottleneck_channels=128).to(dev).eval()
net.load_state_dict(torch.load(CKPT, map_location=dev))
print(f"loaded {CKPT.relative_to(BASE)}  on {dev}")
print(f"params {sum(p.numel() for p in net.parameters()):,}\n")

T = np.asarray(zarr.open_group(str(ORIG / f"{YEAR}.zarr"), mode="r")["fields"][:],
               dtype=np.float64)
S_te = np.load(PCA_DIR / f"scores_{YEAR}.npy").astype(np.float64)
S_tr = np.concatenate([np.load(PCA_DIR / f"scores_{y}.npy") for y in TRAIN]).astype(np.float64)


def h_cols(q):
    tot = 0.0
    for j in range(q.shape[1]):
        _, c = np.unique(q[:, j], return_counts=True)
        p = c / c.sum()
        tot += float(-(p * np.log2(p)).sum())
    return tot


def decode(scores):
    Z = pca.inverse_transform(scores).astype(np.float32)
    out = np.empty((Z.shape[0], 20, T.shape[2], T.shape[3]), dtype=np.float32)
    for i, v in enumerate(vorder):
        lat_r, lon_r = core_shapes[v]
        cores = Z[:, cum[i]:cum[i + 1]].reshape(-1, lat_r, lon_r)
        m, s, L, R = mp[v]
        out[:, i] = np.einsum("ha,tab,bw->thw", L, cores, R.T, optimize=True) * s + m
    return out


@torch.no_grad()
def refine(recon, bs=32):
    """recon -> recon + std * net((recon-mean)/std), the training-time convention."""
    out = np.empty_like(recon)
    for s in range(0, len(recon), bs):
        x = recon[s:s + bs]
        xn = torch.from_numpy((x - mean_c) / std_c).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(dev == "cuda")):
            d = net(xn)
        out[s:s + bs] = x + (d.float().cpu().numpy() * std_c)
    return out


def nrmse(R):
    return float((np.sqrt(((R.astype(np.float64) - T) ** 2).mean(axis=(0, 2, 3)))
                  / std_ref).mean())


rows = []
print(f"{'加的誤差':>9s} {'bits/係數':>9s} {'壓縮率':>9s} "
      f"{'CNN前':>8s} {'CNN後':>8s} {'CNN增益':>8s} {'vs 無量化':>9s}")
base_post = None
for frac in BUDGETS:
    if frac == 0.0:
        sc, bits = S_te, n_comp * 32.0
    else:
        D = np.sqrt(12.0 * frac * trunc / n_comp)
        sc = np.rint(S_te / D) * D
        bits = h_cols(np.rint(S_tr / D).astype(np.int64))

    R = decode(sc)
    pre = nrmse(R)
    post = nrmse(refine(R))
    if base_post is None:
        base_post = post
    ratio = RAW * 32.0 / bits
    rows.append({"budget": frac, "bits_per_coeff": bits / n_comp, "ratio": ratio,
                 "pre_cnn": pre, "post_cnn": post,
                 "cnn_gain_pct": 100 * (post - pre) / pre,
                 "post_vs_unquantised_pct": 100 * (post / base_post - 1)})
    print(f"{100*frac:8.1f}% {bits/n_comp:9.2f} {ratio:8.0f}x {pre:8.5f} "
          f"{post:8.5f} {100*(post-pre)/pre:7.1f}% {100*(post/base_post-1):+8.2f}%")

print(f"\n控制組：未量化 CNN 後 = {rows[0]['post_cnn']:.5f}"
      f"   已知 e995+UNet 值 = 0.13964"
      f"   差 {100*(rows[0]['post_cnn']/0.13964-1):+.2f}%")
json.dump(rows, open(BASE / "results/quant_cnn_eval.json", "w"), indent=2, default=float)
print("wrote results/quant_cnn_eval.json")
