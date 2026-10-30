import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "data" / "processed" / "fourcastnet20_conus"
DEFAULT_MODEL_DIR = ROOT / "models" / "mpca_spatial_99energy"
DEFAULT_OUTPUT_DIR = ROOT / "cores" / "mpca_spatial_99energy" / "train"
LOG_DIR = ROOT / "logs"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "encode_mpca_cores.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_summary(model_dir: Path) -> dict:
    summary_path = model_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def model_path(model_dir: Path, item: dict) -> Path:
    path = Path(item["model_path"])
    if path.is_absolute():
        return path
    return model_dir / path.name


def load_models(model_dir: Path, summary: dict, channels: list[str] | None) -> list[dict]:
    selected = []
    requested = set(channels or [])
    available = {item["name"] for item in summary["channels"]}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Unknown channels: {missing}")

    for item in summary["channels"]:
        if requested and item["name"] not in requested:
            continue
        data = np.load(model_path(model_dir, item))
        selected.append(
            {
                "name": str(data["channel_name"].item()),
                "index": int(data["channel_index"].item()),
                "mean": float(data["mean"].item()),
                "std": float(data["std"].item()),
                "lat_components": data["lat_components"].astype(np.float32),
                "lon_components": data["lon_components"].astype(np.float32),
                "lat_rank": int(data["lat_rank"].item()),
                "lon_rank": int(data["lon_rank"].item()),
                "lat_retained_energy": float(data["lat_retained_energy"].item()),
                "lon_retained_energy": float(data["lon_retained_energy"].item()),
            }
        )
    return selected


def open_input(input_dir: Path, year: int) -> zarr.Group:
    path = input_dir / f"{year}.zarr"
    if not path.exists():
        raise FileNotFoundError(path)
    group = zarr.open_group(str(path), mode="r")
    if "fields" not in group:
        raise KeyError(f"Missing fields array in {path}")
    return group


def create_output(path: Path, input_group: zarr.Group, models: list[dict], year: int, summary: dict,
                  overwrite: bool, chunk_time: int) -> zarr.Group:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}. Use --overwrite to rebuild.")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = input_group["fields"]
    n_time = fields.shape[0]
    group = zarr.open_group(str(path), mode="w")
    group.create_array("time", data=np.asarray(input_group["time"][:], dtype=np.int64), chunks=(min(1024, n_time),))
    group.attrs.update(
        {
            "year": year,
            "layout": "Each channel array is time, mpca_lat_component, mpca_lon_component",
            "source_layout": "time, channel, latitude, longitude",
            "source_normalization": "train_2014_2016 channel mean/std before projection",
            "model_method": summary["method"],
            "energy_threshold": summary["energy_threshold"],
            "channel_names": [m["name"] for m in models],
        }
    )

    for model in models:
        array = group.create_array(
            model["name"],
            shape=(n_time, model["lat_rank"], model["lon_rank"]),
            chunks=(min(chunk_time, n_time), model["lat_rank"], model["lon_rank"]),
            dtype="float32",
        )
        array.attrs.update(
            {
                "channel_index": model["index"],
                "lat_rank": model["lat_rank"],
                "lon_rank": model["lon_rank"],
                "lat_retained_energy": model["lat_retained_energy"],
                "lon_retained_energy": model["lon_retained_energy"],
            }
        )
    return group


def encode_year(input_dir: Path, output_dir: Path, year: int, models: list[dict], summary: dict,
                overwrite: bool, chunk_time: int) -> None:
    input_group = open_input(input_dir, year)
    fields = input_group["fields"]
    source_channels = list(input_group.attrs.get("channel_names", []))
    for model in models:
        if source_channels[model["index"]] != model["name"]:
            raise ValueError(f"Channel order mismatch for {model['name']} in {year}")

    output_path = output_dir / f"{year}.zarr"
    output_group = create_output(output_path, input_group, models, year, summary, overwrite, chunk_time)
    n_time = fields.shape[0]
    logging.info("Encoding %s to %s", input_dir / f"{year}.zarr", output_path)

    for start in range(0, n_time, chunk_time):
        end = min(start + chunk_time, n_time)
        data = np.asarray(fields[start:end, :, :, :], dtype=np.float32)
        for model in models:
            x = (data[:, model["index"], :, :] - model["mean"]) / model["std"]
            x = x.astype(np.float32, copy=False)
            core = np.einsum(
                "ha,thw,wb->tab",
                model["lat_components"],
                x,
                model["lon_components"],
                optimize=True,
            )
            output_group[model["name"]][start:end, :, :] = core.astype(np.float32, copy=False)
    logging.info("Saved %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode annual tensors into per-channel spatial MPCA cores.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--years", type=int, nargs="+", default=[2014, 2015, 2016])
    parser.add_argument("--channels", nargs="+", default=None)
    parser.add_argument("--chunk-time", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    summary = load_summary(args.model_dir)
    models = load_models(args.model_dir, summary, args.channels)
    for year in args.years:
        encode_year(args.input_dir, args.output_dir, year, models, summary, args.overwrite, args.chunk_time)


if __name__ == "__main__":
    main()
