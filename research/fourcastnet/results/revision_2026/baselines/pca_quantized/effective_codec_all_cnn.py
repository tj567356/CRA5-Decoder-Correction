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

    path = BASE / f"checkpoints/{energy}/unet_d2_c64_lr3e-4_clip1/best.pt"
    net = UNet2(20, 64, 3, 2, 128).to(dev).eval()
    net.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    record = next(r for r in json.load(open(BASE / "results/pca_codec_all.json")) if r["energy"] == energy)
    D = record["quant_step"]
    Rq = decode(np.rint(S / D) * D)
    Pq = refine(net, Rq)
    def channel_nr(x):
        return np.sqrt(((x.astype(np.float64)-T)**2).mean(axis=(0,2,3))) / std_ref
    b, a = channel_nr(Rq), channel_nr(Pq)
    near = ["u10","v10","t2m","sp","msl","u1000","v1000","z1000","u850","v850","z850","t850","r850"]
    upper = ["u500","v500","z500","t500","r500","z50"]
    groups = {"all20":list(range(20)), "direct19":[i for i,v in enumerate(names) if v != "tcwv"],
              "near":[names.index(v) for v in near], "upper":[names.index(v) for v in upper]}
    import hashlib
    result = {"energy":energy, "year":2018, "frames":1460,
              "quant_step":D, "ratio_measured":record["ratio_measured"],
              "checkpoint":str(path), "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "precision":"float32 physical reconstruction; float16 autocast correction; float64 errors",
              "no_training_or_selection":True,
              "groups":{g:{"before_nrmse":float(b[ii].mean()), "after_nrmse":float(a[ii].mean()),
                           "G_pct":float(100*(1-a[ii].sum()/b[ii].sum())),
                           "Gbar_pct":float(100*np.mean(1-a[ii]/b[ii]))} for g,ii in groups.items()},
              "channels":{v:{"before_nrmse":float(b[i]), "after_nrmse":float(a[i])} for i,v in enumerate(names)}}
    (OUTPUT / f"{energy}.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result["groups"],indent=2),flush=True)
    return result

rows = [run(e) for e in ("e995", "e99", "e97")]
(OUTPUT / "summary.json").write_text(json.dumps(rows,indent=2))
