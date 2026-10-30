"""
Decode second-stage PCA scores back to reconstructed physical fields.

Pipeline:
  scores (n_time, n_components) -> inverse PCA -> flat (n_time, D)
  flat -> unflatten per variable -> cores (n_time, lat_rank, lon_rank)
  cores -> inverse MPCA -> physical fields (n_time, n_lat, n_lon)

Fully vectorized — no per-timestep Python loop.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import zarr
import joblib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode PCA scores to physical fields")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--pca-dir", type=Path, required=True)
    p.add_argument("--years", type=int, nargs="+", default=[2014, 2015, 2016, 2017, 2018])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--chunk-time", type=int, default=64)
    p.add_argument("--mpca-dir", type=Path, default=None,
                   help="Stage-1 basis directory; defaults to the 99%%-energy one. "
                        "Must match the basis the cores were encoded with.")
    return p.parse_args()


def load_mPCA_models(model_dir: Path, variable_order: list[str]):
    models = {}
    for v in variable_order:
        data = np.load(model_dir / f"{v}.npz")
        models[v] = {
            "mean": float(data["mean"]),
            "std": float(data["std"]),
            "lat_components": data["lat_components"].astype(np.float32),
            "lon_components": data["lon_components"].astype(np.float32),
        }
    return models


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca = joblib.load(args.pca_dir / "pca_model.joblib")

    with (args.pca_dir / "metadata.json").open() as f:
        meta = json.load(f)
    variable_order = meta["variable_order"]
    core_shapes = {v: tuple(s) for v, s in meta["variable_core_shapes"].items()}
    total_input_dim = meta["total_input_dim"]
    n_components = meta["n_components"]
    print(f"[INFO] PCA: {total_input_dim} -> {n_components} components")

    model_dir = args.mpca_dir or (
        args.input_dir / "mpca_train_basis_bundle" / "models" / "mpca_spatial_99energy")
    print(f"[INFO] Stage-1 basis: {model_dir}")
    mpca_models = load_mPCA_models(model_dir, variable_order)
    for v in variable_order:
        got = (mpca_models[v]["lat_components"].shape[1], mpca_models[v]["lon_components"].shape[1])
        if got != core_shapes[v]:
            raise ValueError(
                f"{v}: basis rank {got} does not match the core shape {core_shapes[v]} "
                f"recorded in {args.pca_dir}/metadata.json -- wrong --mpca-dir?")
    print(f"[INFO] Loaded MPCA models for {len(mpca_models)} variables")

    dim_cumsum = [0]
    for v in variable_order:
        dim_cumsum.append(dim_cumsum[-1] + int(np.prod(core_shapes[v])))

    sample_model = mpca_models[variable_order[0]]
    n_lat = sample_model["lat_components"].shape[0]
    n_lon = sample_model["lon_components"].shape[0]
    n_channels = len(variable_order)

    for year in args.years:
        t0 = time.time()
        scores = np.load(args.pca_dir / f"scores_{year}.npy")
        n_time = scores.shape[0]
        print(f"[INFO] {year}: scores {scores.shape}")

        Z_recon = pca.inverse_transform(scores).astype(np.float32)

        out_path = output_dir / f"{year}.zarr"
        if out_path.exists():
            shutil.rmtree(out_path)
        g = zarr.open_group(str(out_path), mode="w")

        fields = g.create_array(
            "fields",
            shape=(n_time, n_channels, n_lat, n_lon),
            chunks=(min(args.chunk_time, n_time), n_channels, n_lat, n_lon),
            dtype="float32",
        )
        g.create_array("channel", data=np.arange(n_channels, dtype=np.int16))

        orig = zarr.open_group(str(args.input_dir / f"{year}.zarr"), mode="r")
        g.create_array("time", data=np.asarray(orig["time"][:], dtype=np.int64))
        g.attrs.update({
            "year": year,
            "layout": "time, channel, latitude, longitude",
            "n_components": n_components,
            "channel_names": variable_order,
        })

        for ch_idx, v in enumerate(variable_order):
            start = dim_cumsum[ch_idx]
            end = dim_cumsum[ch_idx + 1]
            lat_r, lon_r = core_shapes[v]

            cores = Z_recon[:, start:end].reshape(n_time, lat_r, lon_r)

            L = mpca_models[v]["lat_components"]  # (n_lat, lat_rank)
            R = mpca_models[v]["lon_components"]  # (n_lon, lon_rank)
            mean = mpca_models[v]["mean"]
            std = mpca_models[v]["std"]

            # cores: (n_time, lat_rank, lon_rank)
            # x_norm[t] = L @ cores[t] @ R.T
            # Use einsum: (lat_rank, lat) . (time, lat_rank, lon_rank) . (lon_rank, lon)
            # = (time, lat, lon)
            x_norm = np.einsum("ha,tab,bw->thw", L, cores, R.T, optimize=True)
            x = (x_norm * std + mean).astype(np.float32)
            fields[:, ch_idx, :, :] = x
            print(f"  {v}: done ({x.shape})")

        print(f"[INFO] {year}: saved {out_path} in {time.time()-t0:.1f}s")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
