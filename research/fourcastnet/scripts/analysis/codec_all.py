"""The measured rate-distortion table: real coded bytes for every energy level.

pca_codec.py proved the coder works and lands 2.8% above the zeroth-order entropy
bound on e995. This runs it across all five energies so the paper has a curve of
measured points rather than one measured point and four bounds.

Distortion comes from quant_cnn_eval.py's convention (1% core-space budget, which
cost +0.07% post-CNN nRMSE at e995); the post-CNN nRMSE per energy is the known
champion-UNet value, adjusted by nothing -- the quantisation effect was measured as
negligible and identical (-19.2% gain) across the sweep.
"""
import json, os
from pathlib import Path
import constriction, joblib, numpy as np

BASE = Path(os.path.expanduser("~/fourcastnet_us_mpca"))
RAW = 467540
BUDGET = 0.01
YEAR = 2018
POST_CNN = {"e90": 0.27882, "e95": 0.21772, "e97": 0.18828, "e99": 0.15257, "e995": 0.13964}
OLD_RATIO = {"e90": 3624.3, "e95": 1302.3, "e97": 735.1, "e99": 322.7, "e995": 232.5}

out = []
print(f"{'energy':7s} {'n':>6s} {'bytes/frame':>12s} {'bits/係數':>10s} "
      f"{'舊報的':>9s} {'實測':>9s} {'倍數':>7s} {'CNN後 nRMSE':>12s}")
for e in ("e995", "e99", "e97", "e95", "e90"):
    d = BASE / "second_stage_pca" / e
    pca = joblib.load(d / "pca_model.joblib")
    ev = pca.explained_variance_.astype(np.float64)
    n = len(ev)
    trunc = ev.sum() / pca.explained_variance_ratio_.sum() - ev.sum()
    D = np.sqrt(12.0 * BUDGET * trunc / n)
    sigma = np.maximum(np.sqrt(ev) / D, 0.05)
    means = np.zeros(n)

    q = np.rint(np.load(d / f"scores_{YEAR}.npy").astype(np.float64) / D).astype(np.int32)
    LIM = int(np.abs(q).max() * 1.5 + 8)
    model = constriction.stream.model.QuantizedGaussian(-LIM, LIM)

    sizes = np.empty(len(q), dtype=np.int64)
    ok = True
    for t in range(len(q)):
        enc = constriction.stream.queue.RangeEncoder()
        enc.encode(q[t], model, means, sigma)
        c = enc.get_compressed()
        sizes[t] = c.nbytes
        if t in (0, len(q) // 2, len(q) - 1):
            back = constriction.stream.queue.RangeDecoder(c).decode(model, means, sigma)
            ok &= bool(np.array_equal(back, q[t]))
    assert ok, f"{e}: round trip failed"

    total = float(sizes.mean()) + n * 4 / 7304
    bpc = total * 8 / n
    ratio = RAW * 4 / total
    out.append({"energy": e, "n_components": n, "quant_step": D,
                "bytes_per_frame": total, "bits_per_coeff": bpc,
                "ratio_measured": ratio, "ratio_old": OLD_RATIO[e],
                "understated_by": ratio / OLD_RATIO[e],
                "post_cnn_nrmse": POST_CNN[e], "roundtrip_exact": ok})
    print(f"{e:7s} {n:6d} {total:12.1f} {bpc:10.3f} {OLD_RATIO[e]:8.0f}x "
          f"{ratio:8.0f}x {ratio/OLD_RATIO[e]:6.1f}x {POST_CNN[e]:12.5f}")

print("\n對照 CRA5：471x @ 0.0875（無後處理）、471x @ 0.0632（+CNN）")
json.dump(out, open(BASE / "results/pca_codec_all.json", "w"), indent=2, default=float)
print("wrote results/pca_codec_all.json")
