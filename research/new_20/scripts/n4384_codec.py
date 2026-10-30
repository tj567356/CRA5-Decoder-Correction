"""99.9% stage-1 + full-rank stage-2 (n=4384), with a real entropy coder.

This is the pipeline's lowest-error configuration: post-CNN 0.12598, the best number
the linear route has produced. Its reported ratio is 106.6x (467540/4384), which --
like every other ratio in the project -- assumes float32 coefficients.

The quantiser step cannot be budgeted the way the truncated cases were. n=4384 is the
training sample count, so the stage-2 PCA explains essentially all training variance
and its core-space truncation MSE is ~0; the fraction-of-truncation-MSE formula divides
by zero. So the step is swept directly and the cost is read off where it matters, in
field-space nRMSE after the CNN.

Both the pre- and post-CNN numbers are measured with the trained n4384 weights, no
retraining.
"""
import json, os, sys
from pathlib import Path
import constriction, joblib, numpy as np, torch, zarr

NEW20 = Path("/home/tong/new_20")
MAIN = Path("/home/tong/fourcastnet_us_mpca")
sys.path.insert(0, str(MAIN / "scripts/cnn"))
from model import UNet2

PCA_DIR = NEW20 / "second_stage_pca/n4384"
MPCA_DIR = NEW20 / "mpca_train_basis_bundle/models/mpca_spatial_999energy"
CKPT = NEW20 / "checkpoints/n4384/unet_d2_c64_lr3e-4_clip1/best.pt"
ORIG = MAIN / "data/processed/fourcastnet20_conus"
STATS = MAIN / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
RAW, YEAR = 467540, 2018
STEPS = [None, 0.10, 0.25, 0.50, 1.00, 2.00]

st = np.load(STATS)
names = [str(x) for x in st["channel_names"]]
mean_c = st["mean"].astype(np.float32)[:, None, None]
std_c = st["std"].astype(np.float32)[:, None, None]
std_ref = st["std"].astype(np.float64)

pca = joblib.load(PCA_DIR / "pca_model.joblib")
meta = json.load(open(PCA_DIR / "metadata.json"))
vorder, n = meta["variable_order"], meta["n_components"]
shapes = {v: tuple(s) for v, s in meta["variable_core_shapes"].items()}
assert vorder == names
ev = pca.explained_variance_.astype(np.float64)
print(f"n4384: stage-1 dim {meta['total_input_dim']}  n_components {n}")
print(f"  explained_variance_ratio 總和 {pca.explained_variance_ratio_.sum():.6f}"
      f"   → core 空間截斷誤差 ≈ 0，所以步長只能直接掃")

mp = {}
for v in vorder:
    d = np.load(MPCA_DIR / f"{v}.npz")
    mp[v] = (float(d["mean"]), float(d["std"]),
             d["lat_components"].astype(np.float32), d["lon_components"].astype(np.float32))
cum = [0]
for v in vorder:
    cum.append(cum[-1] + int(np.prod(shapes[v])))

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = UNet2(20, 64, 3, 2, 128).to(dev).eval()
net.load_state_dict(torch.load(CKPT, map_location=dev))

T = np.asarray(zarr.open_group(str(ORIG / f"{YEAR}.zarr"), mode="r")["fields"][:], dtype=np.float64)
S = np.load(PCA_DIR / f"scores_{YEAR}.npy").astype(np.float64)
print(f"  truth {T.shape}  scores {S.shape}\n")


def decode(sc):
    Z = pca.inverse_transform(sc).astype(np.float32)
    out = np.empty((len(Z), 20, T.shape[2], T.shape[3]), dtype=np.float32)
    for i, v in enumerate(vorder):
        lr, wr = shapes[v]
        c = Z[:, cum[i]:cum[i + 1]].reshape(-1, lr, wr)
        m, s, L, R = mp[v]
        out[:, i] = np.einsum("ha,tab,bw->thw", L, c, R.T, optimize=True) * s + m
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


rows, base = [], None
print(f"{'步長':>7s} {'bytes/frame':>12s} {'bits/係數':>10s} {'壓縮率':>9s} "
      f"{'CNN前':>8s} {'CNN後':>8s} {'vs 未量化':>9s}")
for D in STEPS:
    if D is None:
        sc, bpf, ratio, bpc = S, n * 4.0, RAW * 4 / (n * 4.0), 32.0
    else:
        sigma = np.maximum(np.sqrt(ev) / D, 0.05)
        means = np.zeros(n)
        q = np.rint(S / D).astype(np.int32)
        LIM = int(np.abs(q).max() * 1.5 + 8)
        model = constriction.stream.model.QuantizedGaussian(-LIM, LIM)
        sizes = np.empty(len(q), dtype=np.int64)
        ok = True
        for t in range(len(q)):
            e = constriction.stream.queue.RangeEncoder()
            e.encode(q[t], model, means, sigma)
            c = e.get_compressed()
            sizes[t] = c.nbytes
            if t in (0, len(q) // 2, len(q) - 1):
                ok &= bool(np.array_equal(
                    constriction.stream.queue.RangeDecoder(c).decode(model, means, sigma), q[t]))
        assert ok, f"round trip failed at D={D}"
        bpf = float(sizes.mean()) + n * 4 / 7304
        bpc, ratio, sc = bpf * 8 / n, RAW * 4 / bpf, q * D

    R = decode(sc)
    pre, post = nr(R), nr(refine(R))
    if base is None:
        base = post
    rows.append({"step": D, "bytes_per_frame": bpf, "bits_per_coeff": bpc,
                 "ratio": ratio, "pre_cnn": pre, "post_cnn": post,
                 "post_vs_unquantised_pct": 100 * (post / base - 1)})
    print(f"{('none' if D is None else f'{D:.2f}'):>7s} {bpf:12.1f} {bpc:10.3f} "
          f"{ratio:8.0f}x {pre:8.5f} {post:8.5f} {100*(post/base-1):+8.2f}%")

print(f"\n控制組：未量化 CNN 後 = {rows[0]['post_cnn']:.5f}   已知 n4384 值 = 0.12598"
      f"   差 {100*(rows[0]['post_cnn']/0.12598-1):+.2f}%")
print("對照：CRA5 + CNN = 471x @ 0.0632")
json.dump(rows, open(NEW20 / "results/n4384_codec.json", "w"), indent=2, default=float)
print("wrote new_20/results/n4384_codec.json")
