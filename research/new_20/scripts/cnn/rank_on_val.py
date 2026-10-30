"""Rank every trained run on the 2017 validation split instead of the 2018 test split.

Why this exists: within a run, the checkpoint is chosen on 2017 (train_cnn.py
saves at min val_loss), which is correct. But the choice of WHICH RUN is the
champion was made by comparing 2018 numbers -- arch_tiers.csv and
cnn_arch_summary.csv both sort on test. Picking the maximum of ~20 candidates on
the reporting split makes that maximum optimistic: even with identical true
skill, one of them wins on luck, and that winner's 2018 score is the one quoted.

It matters here more than usual. The champion U-Net was selected from six U-Net
variants; resblock_c293 is a single untuned configuration. So part of the "1.0%
at equal capacity" gap is selection advantage, not architecture.

No retraining. best.pt already holds the 2017-selected weights for every run;
this only replays the 2017 split through them.

Output is deliberately a SEPARATE file. cnn_arch_summary.csv and
cnn_arch_comparison.png keep reporting test, so nothing that already cites them
silently changes meaning underneath.

    results/val_ranking_2017.csv

Usage:
    .venv/bin/python scripts/cnn/rank_on_val.py
    .venv/bin/python scripts/cnn/rank_on_val.py --energy e99   # just the arch sweep
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import ResidualCNN, ResBlockCNN, MultiScaleCNN, UNet2, FNO2d
from dataset import build_dataloaders, VARIABLE_ORDER
from train_cnn import evaluate

BASE = Path("/home/tong/new_20")
CKPT = BASE / "checkpoints"
STATS = BASE / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz"
ORIG = BASE / "data/processed/fourcastnet20_conus"
OUT = BASE / "results/val_ranking_2017.csv"


def basis_of(test):
    src = str((test.get("memmap_manifest") or {}).get("decoded_dir", "")) \
        or str(test.get("args", {}).get("cache_dir") or "")
    return "greedy_equal" if "greedy" in src else "99energy"


def cache_for(energy, basis):
    """Where this run's memmap actually lives now.

    Old runs recorded cache_dir=data/memmap, which was renamed to
    data/memmap_e99 once the per-energy split existed, so the stored path
    cannot be trusted and is reconstructed from (energy, basis).
    """
    if basis == "greedy_equal":
        return BASE / "data/memmap_greedy_equal_n2011"
    return BASE / f"data/memmap_{energy}"


def remap_legacy_unet(sd, depth):
    """Translate pre-refactor UNet2 checkpoints onto the current module names.

    UNet2 was rewritten from hard-coded depth-2 attributes to depth-configurable
    ModuleLists. The architecture is unchanged; only the names moved:

        enc1 -> encs.0     down1 -> downs.0
        enc2 -> encs.1     down2 -> downs.1
        up2  -> ups.0      dec2  -> decs.0     (decoder counts down from the
        up1  -> ups.1      dec1  -> decs.1      bottleneck, so it reverses)

    Every checkpoint trained before the refactor -- including the champion --
    cannot be loaded by the current code without this. That is a reproducibility
    hole in its own right, quite apart from this script.
    """
    out = {}
    for k, v in sd.items():
        head = k.split(".", 1)[0]
        rest = k[len(head):]
        new = head
        for pre, lst in (("enc", "encs"), ("down", "downs")):
            if head.startswith(pre) and head[len(pre):].isdigit():
                new = f"{lst}.{int(head[len(pre):]) - 1}"
        for pre, lst in (("up", "ups"), ("dec", "decs")):
            if head.startswith(pre) and head[len(pre):].isdigit():
                new = f"{lst}.{depth - int(head[len(pre):])}"
        out[new + rest] = v
    return out


def load_weights(model, path, a, device):
    """Load best.pt, falling back to the legacy UNet key layout."""
    sd = torch.load(path, map_location=device)
    try:
        model.load_state_dict(sd)
        return "direct"
    except RuntimeError:
        if a.get("arch") != "unet":
            raise
        model.load_state_dict(remap_legacy_unet(sd, a.get("depth", 2)))
        return "legacy-remap"


def build_model(a):
    arch = a.get("arch", "conv")
    k = a.get("kernel_size", 3)
    mid = a.get("mid_channels", 64)
    if arch == "resblock":
        bd = [int(d) for d in a["block_dilations"].split(",")] if a.get("block_dilations") else None
        return ResBlockCNN(20, mid, kernel_size=k, block_dilations=bd)
    if arch == "multiscale":
        return MultiScaleCNN(20, mid, kernel_size=k,
                             branch_channels=a.get("branch_channels") or mid // 2)
    if arch == "unet":
        return UNet2(20, mid, kernel_size=k, depth=a.get("depth", 2),
                     bottleneck_channels=a.get("bottleneck_channels"))
    if arch == "fno":
        m1, m2 = [int(m) for m in a["fno_modes"].split(",")]
        return FNO2d(20, width=a["fno_width"], modes1=m1, modes2=m2,
                     n_layers=a.get("fno_layers", 4),
                     project_channels=a.get("fno_project", 128),
                     use_grid=not a.get("fno_no_grid", False))
    dil = [int(d) for d in a["dilations"].split(",")] if a.get("dilations") else None
    return ResidualCNN(20, mid, n_layers=4, kernel_size=k, dilations=dil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", default=None, help="restrict to one energy label")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = np.load(STATS)
    std_np = stats["std"].astype(np.float64)
    criterion = nn.MSELoss()

    runs = []
    for p in sorted(CKPT.glob("*/*/test_results.json")) + sorted(CKPT.glob("*/test_results.json")):
        rel = p.relative_to(CKPT).parts
        energy = rel[0]
        tag = rel[1] if len(rel) > 2 else "baseline"
        if args.energy and energy != args.energy:
            continue
        if not (p.parent / "best.pt").exists():
            print(f"  no best.pt, skipping {energy}/{tag}")
            continue
        runs.append((energy, tag, p))

    print(f"{len(runs)} runs to evaluate on 2017\n")

    loader_cache = {}
    rows = []
    failures = []
    for i, (energy, tag, p) in enumerate(runs, 1):
        test = json.load(open(p))
        a = test.get("args", {})
        basis = basis_of(test)
        cdir = cache_for(energy, basis)
        if not cdir.exists():
            print(f"  [{i}/{len(runs)}] SKIP {energy}/{tag}: no {cdir.name}")
            continue

        key = str(cdir)
        if key not in loader_cache:
            loaders, mean, std = build_dataloaders(
                decoded_dir=str(BASE / "decoded" / energy), orig_dir=str(ORIG),
                stats_path=str(STATS), batch_size=args.batch_size, num_workers=4,
                cache_dir=str(cdir), seed=0)
            loader_cache[key] = (loaders["val"], mean, std)
        val_loader, mean, std = loader_cache[key]

        try:
            model = build_model(a).to(device)
            how = load_weights(model, p.parent / "best.pt", a, device)
        except Exception as e:
            print(f"  [{i}/{len(runs)}] FAILED {energy}/{tag}: {type(e).__name__}: "
                  f"{str(e)[:90]}")
            failures.append(f"{energy}/{tag}")
            continue
        n_params = sum(x.numel() for x in model.parameters())
        if n_params != test.get("n_params", n_params):
            print(f"  [{i}/{len(runs)}] WARN {tag}: rebuilt {n_params:,} != "
                  f"recorded {test.get('n_params'):,}")

        _, pooled, per_ch = evaluate(model, val_loader, device, criterion, mean, std,
                                     amp=False)
        nr = per_ch / std_np
        val_nrmse = float(nr.mean())

        # Two cross-checks. val_nrmse only exists for runs trained after 8/13,
        # but the pooled val RMSE was logged from the start, so the second check
        # covers every run -- including the legacy-remapped U-Nets, where a
        # mis-mapped key would show up as a wildly wrong number rather than
        # silently producing a plausible one.
        logged = None
        pooled_logged = None
        hp = p.parent / "history.json"
        if hp.exists():
            h = json.load(open(hp))
            if h:
                best = min(h, key=lambda r: r["val_loss"])
                logged = best.get("val_nrmse")
                pooled_logged = best.get("val_rmse")
        agree = "" if logged is None else f"{abs(logged - val_nrmse):.2e}"
        if pooled_logged is not None:
            rel = abs(pooled - pooled_logged) / max(pooled_logged, 1e-9)
            if rel > 1e-3:
                print(f"  [{i}/{len(runs)}] MISMATCH {tag}: pooled val {pooled:.3f} "
                      f"vs logged {pooled_logged:.3f} ({rel:.1%}) -- weights or "
                      f"model reconstruction is wrong, EXCLUDING")
                failures.append(f"{energy}/{tag} (pooled mismatch)")
                del model
                torch.cuda.empty_cache()
                continue

        row = {
            "energy": energy, "basis": basis, "arch": a.get("arch", "conv"),
            "tag": tag, "n_params": n_params,
            "val_nrmse_2017": round(val_nrmse, 5),
            "test_nrmse_2018": round(float(test["test_nrmse"]), 5)
            if "test_nrmse" in test else
            round(float(np.mean(np.array([test["test_per_variable_rmse"][v]
                                          for v in VARIABLE_ORDER]) / std_np)), 5),
            "logged_val_nrmse": None if logged is None else round(logged, 5),
            "logged_vs_recomputed": agree,
            "weights_loaded": how,
            "pooled_val_check": "" if pooled_logged is None
            else f"{abs(pooled - pooled_logged) / max(pooled_logged, 1e-9):.1e}",
        }
        for v, x in zip(VARIABLE_ORDER, nr):
            row[f"val_nrmse_{v}"] = round(float(x), 5)
        rows.append(row)
        print(f"  [{i}/{len(runs)}] {energy}/{tag:<38s} val {val_nrmse:.5f}"
              f"  test {row['test_nrmse_2018']:.5f}"
              f"{'  (logged Δ ' + agree + ')' if agree else ''}", flush=True)

        del model
        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("nothing evaluated")

    by_val = sorted(rows, key=lambda r: r["val_nrmse_2017"])
    by_test = sorted(rows, key=lambda r: r["test_nrmse_2018"])
    vrank = {r["tag"] + r["energy"]: i for i, r in enumerate(by_val, 1)}
    trank = {r["tag"] + r["energy"]: i for i, r in enumerate(by_test, 1)}
    for r in rows:
        k = r["tag"] + r["energy"]
        r["rank_by_val"] = vrank[k]
        r["rank_by_test"] = trank[k]
        r["rank_shift"] = trank[k] - vrank[k]

    rows.sort(key=lambda r: r["rank_by_val"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    head = ["rank_by_val", "rank_by_test", "rank_shift", "energy", "basis", "arch",
            "tag", "n_params", "val_nrmse_2017", "test_nrmse_2018",
            "logged_val_nrmse", "logged_vs_recomputed", "weights_loaded",
            "pooled_val_check"]
    head += [f"val_nrmse_{v}" for v in VARIABLE_ORDER]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT}")

    e99 = [r for r in rows if r["energy"] == "e99" and r["basis"] == "99energy"]
    if e99:
        print("\n=== e99, ranked on 2017 validation (2018 shown, not used to rank) ===")
        hdr = (f"{'#val':>4s} {'#test':>5s} {'move':>5s}  {'arch':<10s} {'tag':<38s} "
               f"{'params':>10s} {'val 2017':>9s} {'test 2018':>10s}")
        print(hdr)
        print("-" * len(hdr))
        for r in e99:
            e = [x for x in e99]
            rv = sorted(e, key=lambda x: x["val_nrmse_2017"]).index(r) + 1
            rt = sorted(e, key=lambda x: x["test_nrmse_2018"]).index(r) + 1
            mv = rt - rv
            print(f"{rv:>4d} {rt:>5d} {mv:>+5d}  {r['arch']:<10s} {r['tag']:<38s} "
                  f"{r['n_params']:>10,} {r['val_nrmse_2017']:>9.5f} "
                  f"{r['test_nrmse_2018']:>10.5f}")
        print("\nmove = how many places the run shifts when ranked on val instead "
              "of test. Nonzero means test-ranking was reporting luck as skill.")


if __name__ == "__main__":
    main()
