"""The single-stage pipeline with a real entropy coder instead of LZMA.

quantise_cores.py already established the important half: drop the global PCA
entirely, quantise the stage-1 MPCA cores, and the post-CNN error is 0.0818 -- better
than CRA5's own 0.0875 -- at 205x. That 205x came from LZMA over the quantised
integers, which works out to 7.14 bits per coefficient.

LZMA is a general-purpose matcher. It knows nothing about the fact that coefficient
j has a characteristic scale, and that scale varies over orders of magnitude within a
variable, so most tail coefficients are nearly always the same integer. Modelling
that explicitly is what the range coder in scripts/analysis/pca_codec.py does, and on
the stage-2 scores it reached 5.58 bits where the zeroth-order entropy was 5.43.

This measures the same thing on the stage-1 cores, over a sweep of quantiser steps,
and reports:

  bound     summed per-coefficient zeroth-order entropy, fitted on 2014-2016
  coded     bytes the range coder actually emits for 2018, round-trip verified
  pre/post  field-space nRMSE before and after the trained single-stage CNN

Quantisation is a single global step, not per-variable min/max. The lat and lon bases
are orthonormal, so squared error is additive across coefficients and one shared step
is what reverse water-filling prescribes; per-variable rescaling spends precision on
whichever variable happens to have the widest range.
"""
import json
import os
import sys
from pathlib import Path

import constriction
import numpy as np
import torch
import zarr

NEW20 = Path("/home/tong/new_20")
MAIN = Path("/home/tong/fourcastnet_us_mpca")
sys.path.insert(0, str(MAIN / "scripts/cnn"))
from model import UNet2  # noqa: E402

CORES = MAIN                                                # 99% cores live in the main repo root
MODEL = MAIN / "mpca_train_basis_bundle/models/mpca_spatial_99energy"
CKPT = NEW20 / "checkpoints/s99_stage1only/unet_d2_c64_lr3e-4_clip1/best.pt"
ORIG = MAIN / "data/processed/fourcastnet20_conus"
STATS = MAIN / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
RAW = 467540
YEAR = 2018
TRAIN = [2014, 2015, 2016]
STEPS = [0.08, 0.16, 0.32, 0.64, 1.28, 2.56]

st = np.load(STATS)
names = [str(x) for x in st["channel_names"]]
mean_c = st["mean"].astype(np.float32)[:, None, None]
std_c = st["std"].astype(np.float32)[:, None, None]
std_ref = st["std"].astype(np.float64)

mp, budget = {}, 0
for v in names:
    m = np.load(MODEL / f"{v}.npz", allow_pickle=True)
    L = np.asarray(m["lat_components"], dtype=np.float32)
    R = np.asarray(m["lon_components"], dtype=np.float32)
    mp[v] = (L, R, float(m["mean"]), float(m["std"]))
    budget += L.shape[1] * R.shape[1]
print(f"stage-1 basis {MODEL.name}:  {budget:,} 係數/frame   float32 壓縮率 {RAW/budget:.1f}x")


def load_flat(year):
    """All variables' cores for one year, flattened to (frames, budget)."""
    g = zarr.open_group(str(CORES / f"{year}.zarr"), mode="r")
    parts = []
    for v in names:
        a = np.asarray(g[v][:], dtype=np.float64)
        parts.append(a.reshape(a.shape[0], -1))
    return np.concatenate(parts, axis=1)


F_tr = np.concatenate([load_flat(y) for y in TRAIN])
F_te = load_flat(YEAR)
assert F_te.shape[1] == budget
# per-coefficient scale, from the training years only -- this is the decoder-side
# model, 10,193 floats = 40 KB once, amortised over 7,304 frames = 5.6 bytes/frame
sd = F_tr.std(axis=0)
print(f"係數 std：最大 {sd.max():.4f}  中位 {np.median(sd):.5f}  最小 {sd.min():.2e}")

T = np.asarray(zarr.open_group(str(ORIG / f"{YEAR}.zarr"), mode="r")["fields"][:], dtype=np.float64)
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = UNet2(20, 64, 3, 2, 128).to(dev).eval()
net.load_state_dict(torch.load(CKPT, map_location=dev))
print(f"CNN {CKPT.parent.parent.name} 載入於 {dev}\n")

offs = [0]
for v in names:
    L, R, _, _ = mp[v]
    offs.append(offs[-1] + L.shape[1] * R.shape[1])


def decode(flat):
    out = np.empty((len(flat), 20, T.shape[2], T.shape[3]), dtype=np.float32)
    for i, v in enumerate(names):
        L, R, m, s = mp[v]
        c = flat[:, offs[i]:offs[i + 1]].reshape(-1, L.shape[1], R.shape[1]).astype(np.float32)
        out[:, i] = np.einsum("ha,tab,wb->thw", L, c, R, optimize=True) * s + m
    return out


@torch.no_grad()
def refine(x0, bs=24):
    o = np.empty_like(x0)
    for s in range(0, len(x0), bs):
        x = x0[s:s + bs]
        xn = torch.from_numpy((x - mean_c) / std_c).to(dev)
        with torch.autocast("cuda", torch.float16, enabled=(dev == "cuda")):
            d = net(xn)
        o[s:s + bs] = x + d.float().cpu().numpy() * std_c
    return o


def nr(R):
    return float((np.sqrt(((R.astype(np.float64) - T) ** 2).mean(axis=(0, 2, 3))) / std_ref).mean())


def h_cols(q):
    tot = 0.0
    for j in range(q.shape[1]):
        _, c = np.unique(q[:, j], return_counts=True)
        p = c / c.sum()
        tot += float(-(p * np.log2(p)).sum())
    return tot


rows = []
print(f"{'步長':>7s} {'熵下界':>9s} {'實測':>9s} {'bytes/f':>9s} {'壓縮率':>9s} "
      f"{'CNN前':>8s} {'CNN後':>8s}")
# float32 reference row
R0 = decode(F_te)
pre0, post0 = nr(R0), nr(refine(R0))
print(f"{'none':>7s} {'-':>9s} {'32.00':>9s} {budget*4:9.0f} {RAW*4/(budget*4):8.1f}x "
      f"{pre0:8.5f} {post0:8.5f}")
rows.append({"step": None, "bits_per_coeff": 32.0, "bytes_per_frame": budget * 4.0,
             "ratio": RAW * 4 / (budget * 4), "pre_cnn": pre0, "post_cnn": post0})

for D in STEPS:
    sigma = np.maximum(sd / D, 0.05)
    means = np.zeros(budget)
    q_tr = np.rint(F_tr / D).astype(np.int32)
    q_te = np.rint(F_te / D).astype(np.int32)
    bound = h_cols(q_tr) / budget

    LIM = int(max(np.abs(q_tr).max(), np.abs(q_te).max()) * 1.2 + 8)
    model = constriction.stream.model.QuantizedGaussian(-LIM, LIM)
    sizes = np.empty(len(q_te), dtype=np.int64)
    ok = True
    for t in range(len(q_te)):
        e = constriction.stream.queue.RangeEncoder()
        e.encode(q_te[t], model, means, sigma)
        c = e.get_compressed()
        sizes[t] = c.nbytes
        if t in (0, len(q_te) // 2, len(q_te) - 1):
            ok &= bool(np.array_equal(
                constriction.stream.queue.RangeDecoder(c).decode(model, means, sigma), q_te[t]))
    assert ok, f"round trip failed at D={D}"

    bpf = float(sizes.mean()) + budget * 4 / 7304          # sigma table, amortised
    bpc, ratio = bpf * 8 / budget, RAW * 4 / bpf
    R = decode(q_te * D)
    pre, post = nr(R), nr(refine(R))
    rows.append({"step": D, "entropy_bound_bits": bound, "bits_per_coeff": bpc,
                 "bytes_per_frame": bpf, "ratio": ratio, "pre_cnn": pre, "post_cnn": post,
                 "roundtrip_exact": ok})
    print(f"{D:7.3f} {bound:9.3f} {bpc:9.3f} {bpf:9.1f} {ratio:8.0f}x {pre:8.5f} {post:8.5f}")

print(f"\n控制組：float32 的 CNN 後 = {post0:.5f}   既有 s99_stage1only 值 = 0.08191"
      f"   差 {100*(post0/0.08191-1):+.2f}%")
print("既有 12bit + LZMA：205.2x @ 0.08178，等效 7.14 bits/係數")
print("對照 CRA5：471x @ 0.0875（原始）、471x @ 0.0632（+CNN）")
json.dump(rows, open(NEW20 / "results/stage1_codec.json", "w"), indent=2, default=float)
print("wrote new_20/results/stage1_codec.json")
