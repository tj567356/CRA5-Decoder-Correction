"""Train residual CNN baseline.

Usage:
    python train_cnn.py --energy e99
    python train_cnn.py --energy e95 --batch_size 16 --lr 5e-4
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import memmap_manifest
from model import ResidualCNN, ResBlockCNN, MultiScaleCNN, UNet2, FNO2d
from dataset import build_dataloaders, VARIABLE_ORDER, TRAIN_YEARS, VAL_YEARS, TEST_YEARS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def evaluate(model, loader, device, criterion, mean, std, amp=False):
    """Evaluate on a data split.

    Computes RMSE in physical units by explicitly reconstructing:
        final = recon + pred * std
    and comparing against the original fields.

    Args:
        mean, std: (C,) numpy arrays — per-channel train stats.
        amp: if True, run the forward pass under fp16 autocast (reconstruction
             is always done in float32 to avoid fp16 overflow on large fields
             like msl ~ 1e5).
    Returns:
        avg_loss, overall_rmse, per_ch_rmse
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    n_samples = 0
    sum_sq = np.zeros(len(mean), dtype=np.float64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                pred_norm = model(x)                     # (B,C,H,W) normalised residual
                loss = criterion(pred_norm, y)
            total_loss += loss.item()
            n_batches += 1

            # ── explicit physical-space reconstruction (always float32) ──
            pred_norm = pred_norm.float()
            std_t = torch.from_numpy(std).to(device)     # (C,)
            mean_t = torch.from_numpy(mean).to(device)   # (C,)
            recon_phys = x * std_t[None, :, None, None] + mean_t[None, :, None, None]
            pred_resid_phys = pred_norm * std_t[None, :, None, None]
            final = recon_phys + pred_resid_phys          # CNN-corrected fields
            orig_phys = recon_phys + y * std_t[None, :, None, None]
            diff = final - orig_phys
            b, c, h, w = diff.shape
            sum_sq += (diff.float().cpu().numpy() ** 2).sum(axis=(0, 2, 3))
            n_samples += b

    avg_loss = total_loss / max(n_batches, 1)
    per_ch_mse = sum_sq / max(n_samples * h * w, 1)
    per_ch_rmse = np.sqrt(per_ch_mse)
    overall_rmse = float(np.sqrt(per_ch_mse.mean()))
    return avg_loss, overall_rmse, per_ch_rmse


def run_lr_finder(model, loader, device, args, ckpt_dir):
    """1-epoch LR range test: exponentially ramp LR per batch, log loss.

    LR goes start_lr -> end_lr over one pass of the training split (2014-2016).
    Suggests a conservative LR in the fast-descent region (steepest slope
    point / 3) rather than the lowest-loss point.
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_steps = max(len(loader), 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.start_lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    criterion = nn.MSELoss()

    log_lr0 = np.log10(args.start_lr)
    log_lr1 = np.log10(args.end_lr)

    records = []
    loss_ema = None
    t0 = time.time()
    for step, (x, y) in enumerate(loader):
        frac = step / max(n_steps - 1, 1)
        lr = 10 ** (log_lr0 + frac * (log_lr1 - log_lr0))
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            out = model(x)
            loss = criterion(out, y)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        v = float(loss.detach().float().cpu().item())
        loss_ema = v if loss_ema is None else 0.95 * loss_ema + 0.05 * v
        records.append({"step": step, "lr": lr, "loss": v, "loss_ema": loss_ema})

        if (step + 1) % 50 == 0 or step == n_steps - 1:
            print(f"  step {step + 1}/{n_steps}  lr={lr:.3e}  loss={v:.5f}  "
                  f"ema={loss_ema:.5f}  ({time.time() - t0:.0f}s)")

    lrs = np.array([r["lr"] for r in records])
    ema = np.array([r["loss_ema"] for r in records])

    skip = max(int(len(ema) * 0.1), 2)
    finite = np.isfinite(ema)
    usable = np.arange(len(ema))[finite]
    usable = usable[usable >= skip]
    if len(usable) == 0:
        usable = np.arange(len(ema))[finite]

    min_idx = int(usable[np.argmin(ema[usable])])
    min_lr = float(lrs[min_idx])

    pre = usable[usable <= min_idx]
    if len(pre) > 3:
        x = np.log10(lrs[pre])
        y = ema[pre]
        slope = np.gradient(y, x)
        steep_idx = int(pre[np.argmin(slope)])
        steep_lr = float(lrs[steep_idx])
    else:
        steep_lr = min_lr

    suggested_lr = min_lr / 3.0
    suggested_lr = float(np.clip(suggested_lr, args.start_lr, args.end_lr))

    tag = args.tag or f"k{args.kernel_size}_c{args.mid_channels}"
    base = f"lr_finder_{tag}"
    csv_path = f"{ckpt_dir}/{base}.csv"
    png_path = f"{ckpt_dir}/{base}.png"

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "lr", "loss", "loss_ema"])
        w.writeheader()
        w.writerows(records)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(lrs, ema, label="loss (EMA 0.95)", color="tab:blue")
    ax.semilogx(lrs, [r["loss"] for r in records], alpha=0.35,
                label="raw loss", color="tab:gray")
    ax.axvline(steep_lr, color="tab:green", ls="--",
               label=f"steepest {steep_lr:.2e}")
    ax.axvline(suggested_lr, color="tab:red", ls="--",
               label=f"suggested {suggested_lr:.2e}")
    ax.axvline(min_lr, color="tab:orange", ls=":", alpha=0.7,
               label=f"min-loss {min_lr:.2e}")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log)")
    ax.set_ylabel("train loss")
    ax.set_title(f"LR finder — {tag} ({args.arch})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"\n=== LR FINDER ({tag}) ===")
    print(f"  steepest-descent LR : {steep_lr:.3e}")
    print(f"  min-loss LR         : {min_lr:.3e}")
    print(f"  suggested LR        : {suggested_lr:.3e}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")


def main():
    parser = argparse.ArgumentParser()
    # Not a closed choice list: with --cache_dir the data comes from the cache and
    # this only labels the run (checkpoints/<energy>/<tag>). Forcing an allocation
    # experiment to call itself "e995" is how a run ends up mislabelled.
    parser.add_argument("--energy", type=str, default="e99",
                        help="run label; also decoded/<energy> when --cache_dir is unset")
    parser.add_argument("--kernel_size", type=int, default=3, choices=[3, 5, 7, 9])
    parser.add_argument("--mid_channels", type=int, default=64)
    parser.add_argument("--branch_channels", type=int, default=None,
                        help="multiscale branch width (default: mid_channels // 2, "
                             "which reproduces the original c64/b32 model)")
    parser.add_argument("--depth", type=int, default=2,
                        help="U-Net depth (number of down/up levels)")
    parser.add_argument("--bottleneck_channels", type=int, default=None,
                        help="override U-Net bottleneck width (default: "
                             "mid_channels * 2^depth)")
    parser.add_argument("--dilations", type=str, default=None,
                        help="comma-separated per-layer dilations, e.g. '1,2,4,1'")
    parser.add_argument("--fno_width", type=int, default=24,
                        help="FNO channel width after the lifting layer")
    parser.add_argument("--fno_modes", type=str, default="18,18",
                        help="FNO retained Fourier modes, 'm_lat,m_lon'. rfft2 on "
                             "97x241 gives 97x121 bins, so m_lat<=48, m_lon<=121")
    parser.add_argument("--fno_layers", type=int, default=4)
    parser.add_argument("--fno_project", type=int, default=128,
                        help="hidden width of the FNO projection head")
    parser.add_argument("--fno_no_grid", action="store_true",
                        help="drop the coordinate-grid input channels. The grid "
                             "hands the model absolute position, which conv/U-Net "
                             "never had -- this is the ablation for that confound")
    parser.add_argument("--arch", type=str, default="conv",
                        choices=["conv", "resblock", "multiscale", "unet", "fno"],
                        help="network architecture (default: conv)")
    parser.add_argument("--block_dilations", type=str, default=None,
                        help="comma-separated dilations for the two res-block convs, "
                             "e.g. '2,4'")
    parser.add_argument("--tag", type=str, default=None,
                        help="override checkpoint dir name under checkpoints/<energy>/")
    parser.add_argument("--decoded_dir", type=str,
                        default="/home/tong/new_20/decoded")
    parser.add_argument("--orig_dir", type=str,
                        default="/home/tong/new_20/data/processed/fourcastnet20_conus")
    parser.add_argument("--stats_path", type=str,
                        default="/home/tong/new_20/mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=None,
                        help="max grad norm for clipping (e.g. 1.0) to stabilise "
                             "training of large/deep models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--preload", action="store_true",
                        help="preload all years into RAM as float16 (fast, needs ~14 GB); "
                             "default is lazy reading from disk (low RAM, ~2 GB)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="dir with prepared float16 memmap .npy files (see "
                             "prepare_memmap.py). Fastest low-RAM mode.")
    parser.add_argument("--allow_unlabelled_cache", action="store_true",
                        help="proceed when --cache_dir has no manifest.json (warns "
                             "instead of failing); backfill one with memmap_manifest.py")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for reproducible training (default: 0)")
    parser.add_argument("--amp", action="store_true",
                        help="train with mixed precision (fp16 autocast + GradScaler)")
    parser.add_argument("--lr_finder", action="store_true",
                        help="run a 1-epoch LR range test instead of training; "
                             "saves results/<tag>_lr_finder.csv/.png and exits")
    parser.add_argument("--start_lr", type=float, default=1e-6)
    parser.add_argument("--end_lr", type=float, default=3e-3)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    decoded_dir = f"{args.decoded_dir}/{args.energy}"
    dilations = [int(d) for d in args.dilations.split(",")] if args.dilations else None
    dil_suffix = "_dil" + "".join(map(str, dilations)) if dilations else ""
    subdir = args.tag or f"k{args.kernel_size}_c{args.mid_channels}{dil_suffix}"
    ckpt_dir = f"/home/tong/new_20/checkpoints/{args.energy}/{subdir}"
    os.makedirs(ckpt_dir, exist_ok=True)

    cache_manifest = None
    if args.cache_dir:
        cache_manifest = memmap_manifest.verify(
            args.cache_dir, TRAIN_YEARS + VAL_YEARS + TEST_YEARS,
            allow_missing=args.allow_unlabelled_cache)
        if cache_manifest:
            print(f"Cache: {args.cache_dir}\n"
                  f"       built from {cache_manifest['decoded_dir']}"
                  f" ({cache_manifest['created']})")

    loaders, mean, std = build_dataloaders(
        decoded_dir, args.orig_dir, args.stats_path,
        batch_size=args.batch_size, num_workers=args.num_workers,
        preload=args.preload, cache_dir=args.cache_dir, seed=args.seed,
    )
    print(f"Train: {len(loaders['train'].dataset)} samples, "
          f"Val: {len(loaders['val'].dataset)} samples, "
          f"Test: {len(loaders['test'].dataset)} samples")

    if args.arch == "resblock":
        block_dils = [int(d) for d in args.block_dilations.split(",")] if args.block_dilations else None
        model = ResBlockCNN(in_channels=20, mid_channels=args.mid_channels,
                            kernel_size=args.kernel_size,
                            block_dilations=block_dils).to(device)
    elif args.arch == "multiscale":
        branch_ch = args.branch_channels or args.mid_channels // 2
        model = MultiScaleCNN(in_channels=20, mid_channels=args.mid_channels,
                              kernel_size=args.kernel_size,
                              branch_channels=branch_ch).to(device)
    elif args.arch == "fno":
        m1, m2 = [int(m) for m in args.fno_modes.split(",")]
        model = FNO2d(in_channels=20, width=args.fno_width, modes1=m1, modes2=m2,
                      n_layers=args.fno_layers, project_channels=args.fno_project,
                      use_grid=not args.fno_no_grid).to(device)
    elif args.arch == "unet":
        model = UNet2(in_channels=20, mid_channels=args.mid_channels,
                      kernel_size=args.kernel_size, depth=args.depth,
                      bottleneck_channels=args.bottleneck_channels).to(device)
    else:
        model = ResidualCNN(in_channels=20, mid_channels=args.mid_channels, n_layers=4,
                            kernel_size=args.kernel_size, dilations=dilations).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}  amp={args.amp}")

    if args.lr_finder:
        run_lr_finder(model, loaders["train"], device, args, ckpt_dir)
        return

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batch = 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                out = model(x)
                loss = criterion(out, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n_batch += 1
        train_loss /= max(n_batch, 1)

        val_loss, val_rmse, val_per_ch = evaluate(model, loaders["val"], device, criterion, mean, std,
                                         amp=args.amp)
        dt = time.time() - t0

        tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{ckpt_dir}/best.pt")
            tag = " *saved*"
        else:
            epochs_no_improve += 1

        record = {
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_rmse": val_rmse,
            # val_rmse is sqrt(mean(rmse^2)) in physical units and is 84% `sp`;
            # val_nrmse is the metric we actually report.
            "val_nrmse": float(np.mean(val_per_ch / std)),
        }
        history.append(record)
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"val_rmse={val_rmse:.3f}  {dt:.1f}s{tag}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Test evaluation with best checkpoint ──
    model.load_state_dict(torch.load(f"{ckpt_dir}/best.pt", map_location=device, weights_only=True))
    test_loss, test_rmse, test_per_ch = evaluate(model, loaders["test"], device, criterion, mean, std)
    print(f"\n=== TEST ({args.energy}) ===")
    print(f"  Overall RMSE (physical): {test_rmse:.3f}")
    for i, v in enumerate(VARIABLE_ORDER):
        print(f"  {v:6s}: {test_per_ch[i]:.3f}")

    # ── Save results ──
    results = {
        "energy": args.energy,
        "test_overall_rmse": test_rmse,
        "test_per_variable_rmse": {v: float(test_per_ch[i]) for i, v in enumerate(VARIABLE_ORDER)},
        "test_nrmse": float(np.mean(test_per_ch / std)),
        "test_per_variable_nrmse": {v: float(test_per_ch[i] / std[i])
                                    for i, v in enumerate(VARIABLE_ORDER)},
        "best_val_loss": best_val_loss,
        "args": vars(args),
        "n_params": total_params,
        "cache_manifest": cache_manifest,
    }
    with open(f"{ckpt_dir}/test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{ckpt_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nResults saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
