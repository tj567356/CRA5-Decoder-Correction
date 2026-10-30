"""Per-frame rate spread for both codecs, so the paper can say what was matched.

rate_matched.py bisects on the MEAN coded size, which is the right target -- both
CRA5 and our range coder emit variable-length frames, and an archive's cost is
the sum over frames. But the write-up currently says "both systems at 471.6x",
which reads as a per-frame guarantee neither system offers. CRA5's own frames
span 459-481x. This measures our side the same way so the claim can be stated as
what it is: the 2018 mean rates are aligned.

Reuses rate_matched.py's configuration and its solved quantiser step rather than
re-bisecting, so the numbers describe the operating point the paper reports.

    python scripts/rate_spread.py -> results/rate_spread.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import constriction
import joblib
import zarr

NEW = Path("/home/tong/new_20")
OLD = Path("/home/tong/fourcastnet_us_mpca")
sys.path.insert(0, str(NEW / "scripts"))

RAW = 467540
RAW_GLOBAL = 721 * 1440 * 268 * 4
YEAR = 2018
VAL = 2017
TRAIN = [2014, 2015, 2016]
ARCHIVE_FRAMES = 7304

names = [str(v) for v in np.load(
    OLD / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
)["channel_names"]]

CONFIGS = [
    dict(tag="single-stage 99%", kind="cores", src=OLD,
         basis=OLD / "mpca_train_basis_bundle/models/mpca_spatial_99energy"),
    dict(tag="two-stage n4384", kind="scores",
         src=NEW / "second_stage_pca/n4384",
         basis=NEW / "mpca_train_basis_bundle/models/mpca_spatial_999energy"),
]


def cores_flat(src, year):
    g = zarr.open_group(str(src / f"{year}.zarr"), mode="r")
    return np.concatenate([np.asarray(g[v][:], dtype=np.float64)
                           .reshape(g[v].shape[0], -1) for v in names], axis=1)


def spread(cfg, step):
    if cfg["kind"] == "cores":
        C_tr = np.concatenate([cores_flat(cfg["src"], y) for y in TRAIN])
        C_va = cores_flat(cfg["src"], VAL)
        C_te = cores_flat(cfg["src"], YEAR)
    else:
        C_tr = np.concatenate([np.load(cfg["src"] / f"scores_{y}.npy")
                               for y in TRAIN]).astype(np.float64)
        C_va = np.load(cfg["src"] / f"scores_{VAL}.npy").astype(np.float64)
        C_te = np.load(cfg["src"] / f"scores_{YEAR}.npy").astype(np.float64)
    n = C_te.shape[1]
    sd = C_tr.std(axis=0)
    sigma = np.maximum(sd / step, 0.05)
    means = np.zeros(n)
    q = np.rint(C_te / step).astype(np.int32)
    q_tr_max = np.abs(np.rint(C_tr / step)).max()
    q_val_max = np.abs(np.rint(C_va / step)).max()
    LIM = int(max(q_tr_max, q_val_max) * 1.2 + 8)
    model = constriction.stream.model.QuantizedGaussian(-LIM, LIM)
    model_bytes = n * 4 / ARCHIVE_FRAMES          # sigma table, amortised

    sizes = np.empty(len(q))
    for t in range(len(q)):
        e = constriction.stream.queue.RangeEncoder()
        e.encode(q[t], model, means, sigma)
        sizes[t] = e.get_compressed().nbytes + model_bytes
    return sizes


def report(label, sizes, raw):
    r = raw / sizes
    print(f"{label}")
    print(f"  bytes/frame  mean {sizes.mean():,.0f}  std {sizes.std():,.0f}"
          f" ({100*sizes.std()/sizes.mean():.2f}%)"
          f"  min {sizes.min():,.0f}  max {sizes.max():,.0f}")
    print(f"  ratio  mean-rate {raw/sizes.mean():.2f}x"
          f"  per-frame {r.min():.1f}–{r.max():.1f}x")
    return {"label": label, "n_frames": int(len(sizes)),
            "bytes_mean": float(sizes.mean()), "bytes_std": float(sizes.std()),
            "bytes_cv_pct": float(100 * sizes.std() / sizes.mean()),
            "bytes_min": float(sizes.min()), "bytes_max": float(sizes.max()),
            "ratio_of_mean": float(raw / sizes.mean()),
            "ratio_min": float(r.min()), "ratio_max": float(r.max())}


out = []
man = [l.split("\t") for l in open(OLD / "cra5/manifest.tsv") if "\t" in l]
cra = np.array([float(b) for f, b in man if f.startswith("2018/")])
out.append(report("CRA5", cra, RAW_GLOBAL))

steps = {r["n"]: r["step"] for r in
         json.load(open(NEW / "results/rate_matched_val.json"))["rows"]}
for cfg in CONFIGS:
    n = 10193 if cfg["kind"] == "cores" else 4384
    out.append(report(f"ours: {cfg['tag']}", spread(cfg, steps[n]), RAW * 4))

json.dump(out, open(NEW / "results/rate_spread.json", "w"), indent=2)
print(f"\nwrote {NEW}/results/rate_spread.json")
