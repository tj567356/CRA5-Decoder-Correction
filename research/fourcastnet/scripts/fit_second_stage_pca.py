"""
Second-stage PCA on first-stage MPCA cores.

Reads per-variable core arrays from zarr stores, flattens and concatenates
all variables per timestep into a single feature vector, fits PCA on
train years only, and transforms all years.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit second-stage PCA on MPCA cores")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--train-years", type=int, nargs="+", default=[2014, 2015, 2016])
    p.add_argument("--val-years", type=int, nargs="+", default=[2017])
    p.add_argument("--test-years", type=int, nargs="+", default=[2018])
    p.add_argument("--energy", type=float, default=0.99, help="Cumulative explained variance")
    p.add_argument("--n-components", type=int, default=None,
                   help="Fix the component count instead of picking it by energy")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def load_year_flat(input_dir: Path, year: int, variable_order: list[str]):
    """Load one year, return (n_time, D) float32 matrix, flattened per timestep."""
    zarr_path = input_dir / f"{year}.zarr"
    g = zarr.open_group(str(zarr_path), mode="r")
    channel_names = list(g.attrs["channel_names"])
    # Assert order matches
    assert channel_names == variable_order, (
        f"Channel order mismatch in {year}: {channel_names} vs {variable_order}"
    )
    n_time = g["time"].shape[0]
    chunks = []
    shapes = []
    for v in variable_order:
        arr = g[v]
        shapes.append(arr.shape[1:])  # (lat_rank, lon_rank)
        chunks.append(arr[:])  # read entire (time, lat_rank, lon_rank)
    # chunks: list of (n_time, lat_rank, lon_rank)
    flat_parts = [c.reshape(n_time, -1) for c in chunks]  # each (n_time, lat_rank*lon_rank)
    Z = np.concatenate(flat_parts, axis=1).astype(np.float32)
    return Z, shapes


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_years = sorted(set(args.train_years + args.val_years + args.test_years))
    print(f"[INFO] All years: {all_years}")

    # Determine variable order from first year
    first_zarr = zarr.open_group(str(args.input_dir / f"{all_years[0]}.zarr"), mode="r")
    variable_order = list(first_zarr.attrs["channel_names"])
    print(f"[INFO] Variable order ({len(variable_order)} variables): {variable_order}")

    # Collect core shapes
    core_shapes = {}
    for v in variable_order:
        core_shapes[v] = list(first_zarr[v].shape[1:])
    print("[INFO] Core shapes per variable:")
    for v in variable_order:
        print(f"  {v}: {core_shapes[v]}")

    total_input_dim = sum(int(np.prod(first_zarr[v].shape[1:])) for v in variable_order)
    print(f"[INFO] Total input feature dimension: {total_input_dim}")

    # Load all years
    year_data = {}
    for yr in all_years:
        t0 = time.time()
        Z, shapes = load_year_flat(args.input_dir, yr, variable_order)
        year_data[yr] = Z
        print(f"[INFO] {yr}: shape {Z.shape}, loaded in {time.time()-t0:.1f}s")

    # Fit PCA on train years only
    Z_train = np.concatenate([year_data[y] for y in args.train_years], axis=0)
    print(f"[INFO] Train matrix: {Z_train.shape}")

    max_components = min(Z_train.shape[0], Z_train.shape[1])

    if args.n_components:
        # Fixed component count: the comparison axis when two different stage-1
        # allocations must land on the same end-to-end compression ratio. Skips
        # the full-rank fit entirely -- explained_variance_ratio_ is still taken
        # against the total variance, so the energy report stays meaningful.
        n_comp = min(args.n_components, max_components)
        print(f"[INFO] Fixed n_components = {n_comp}")
    else:
        # Fit with high n_components first, then truncate by energy
        print(f"[INFO] Fitting PCA with up to {max_components} components...")
        t0 = time.time()
        pca_full = PCA(n_components=max_components, random_state=42)
        pca_full.fit(Z_train)
        fit_time = time.time() - t0
        print(f"[INFO] PCA fit done in {fit_time:.1f}s")

        # Find number of components for desired energy
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cumvar, args.energy)) + 1
        # Clamp
        n_comp = min(n_comp, max_components)
        actual_energy = float(cumvar[n_comp - 1])
        print(f"[INFO] Energy {args.energy*100:.1f}% -> {n_comp} components (actual: {actual_energy*100:.2f}%)")
    print(f"[INFO] Compression ratio: {total_input_dim} -> {n_comp} = {total_input_dim/n_comp:.1f}x")

    # Refit with exact component count for efficiency
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(Z_train)
    actual_energy_final = float(np.cumsum(pca.explained_variance_ratio_)[-1])

    # Save model
    model_path = output_dir / "pca_model.joblib"
    joblib.dump(pca, model_path)
    print(f"[INFO] Saved model: {model_path}")

    # Transform and save scores for each year
    for yr in all_years:
        scores = pca.transform(year_data[yr]).astype(np.float32)
        scores_path = output_dir / f"scores_{yr}.npy"
        np.save(scores_path, scores)
        print(f"[INFO] {yr}: scores shape {scores.shape} -> {scores_path}")

    # Save metadata
    cumvar_selected = np.cumsum(pca.explained_variance_ratio_).tolist()
    metadata = {
        "train_years": args.train_years,
        "val_years": args.val_years,
        "test_years": args.test_years,
        "variable_order": variable_order,
        "variable_core_shapes": core_shapes,
        "total_input_dim": total_input_dim,
        "n_components": n_comp,
        "explained_variance_ratio_cumulative": cumvar_selected,
        "actual_energy": actual_energy_final,
        "compression_ratio": total_input_dim / n_comp,
        "energy_threshold": args.energy,
        "pca_model_path": "pca_model.joblib",
        "score_files": {str(yr): f"scores_{yr}.npy" for yr in all_years},
        "year_shapes": {str(yr): list(year_data[yr].shape) for yr in all_years},
    }
    meta_path = output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved metadata: {meta_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SECOND-STAGE PCA SUMMARY")
    print("=" * 60)
    for yr in all_years:
        s = np.load(output_dir / f"scores_{yr}.npy")
        print(f"  {yr}: scores shape = {s.shape}")
    print(f"  Components: {n_comp}")
    print(f"  Cumulative explained variance: {actual_energy_final*100:.2f}%")
    print(f"  Compression: {total_input_dim} -> {n_comp} ({total_input_dim/n_comp:.1f}x)")
    print("=" * 60)


if __name__ == "__main__":
    main()
