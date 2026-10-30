import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "variables.json"
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed" / "fourcastnet20_conus"
LOG_DIR = ROOT / "logs"

SINGLE_SHORT_NAMES = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "surface_pressure": "sp",
    "mean_sea_level_pressure": "msl",
    "total_column_water_vapour": "tcwv",
}

PRESSURE_SHORT_NAMES = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
    "relative_humidity": "r",
}


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "preprocess_year_tensor.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def current_raw_paths(year: int, month: int) -> tuple[Path, Path]:
    return (
        RAW_DIR / "single_levels" / str(year) / f"{year}_{month:02d}_single.nc",
        RAW_DIR / "pressure_levels" / str(year) / f"{year}_{month:02d}_pressure.nc",
    )


def alternate_raw_paths(year: int, month: int) -> tuple[Path, Path]:
    base = RAW_DIR / "fourcastnet20_conus" / str(year)
    return (
        base / f"{year}_{month:02d}_sfc.nc",
        base / f"{year}_{month:02d}_pl.nc",
    )


def month_paths(year: int, month: int) -> tuple[Path, Path]:
    single, pressure = current_raw_paths(year, month)
    if single.exists() and pressure.exists():
        return single, pressure
    single, pressure = alternate_raw_paths(year, month)
    if single.exists() and pressure.exists():
        return single, pressure
    return current_raw_paths(year, month)


def assert_complete_year(year: int) -> tuple[list[Path], list[Path]]:
    single_paths = []
    pressure_paths = []
    missing = []
    for month in range(1, 13):
        single, pressure = month_paths(year, month)
        single_paths.append(single)
        pressure_paths.append(pressure)
        for path in (single, pressure):
            if not path.exists() or path.stat().st_size == 0:
                missing.append(path)
    if missing:
        raise FileNotFoundError("Missing raw files:\n" + "\n".join(str(p) for p in missing))
    return single_paths, pressure_paths


def infer_year_grid(single_paths: list[Path], pressure_paths: list[Path]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    n_time = 0
    times = []
    lat_ref = None
    lon_ref = None
    for single_path, pressure_path in zip(single_paths, pressure_paths):
        with xr.open_dataset(single_path) as ds_single, xr.open_dataset(pressure_path) as ds_pressure:
            time_single = ds_single["valid_time"].values
            time_pressure = ds_pressure["valid_time"].values
            if not np.array_equal(time_single, time_pressure):
                raise ValueError(f"Time mismatch: {single_path} vs {pressure_path}")

            lat = ds_single["latitude"].values
            lon = ds_single["longitude"].values
            if lat_ref is None:
                lat_ref = lat
                lon_ref = lon
            elif not np.array_equal(lat_ref, lat) or not np.array_equal(lon_ref, lon):
                raise ValueError(f"Grid mismatch in {single_path}")

            if not np.array_equal(lat_ref, ds_pressure["latitude"].values):
                raise ValueError(f"Pressure latitude mismatch in {pressure_path}")
            if not np.array_equal(lon_ref, ds_pressure["longitude"].values):
                raise ValueError(f"Pressure longitude mismatch in {pressure_path}")

            n_time += time_single.shape[0]
            times.append(time_single)

    return n_time, lat_ref.astype(np.float32), lon_ref.astype(np.float32), np.concatenate(times)


def read_channel(ds_single: xr.Dataset, ds_pressure: xr.Dataset, channel: dict) -> np.ndarray:
    if channel["source"] == "single":
        data = ds_single[SINGLE_SHORT_NAMES[channel["variable"]]]
    elif channel["source"] == "pressure":
        data = ds_pressure[PRESSURE_SHORT_NAMES[channel["variable"]]].sel(pressure_level=float(channel["level"]))
    else:
        raise ValueError(channel["source"])
    return np.asarray(data.values, dtype=np.float32)


def create_store(path: Path, shape: tuple[int, int, int, int], lat: np.ndarray, lon: np.ndarray,
                 time_values: np.ndarray, channel_names: list[str], config: dict, year: int) -> zarr.Group:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    group = zarr.open_group(str(path), mode="w")
    group.create_array("fields", shape=shape, chunks=(32, shape[1], shape[2], shape[3]), dtype="float32")
    group.create_array("lat", data=lat.astype(np.float32), chunks=lat.shape)
    group.create_array("lon", data=lon.astype(np.float32), chunks=lon.shape)
    group.create_array("time", data=time_values.astype("datetime64[ns]").astype("int64"), chunks=(min(1024, shape[0]),))
    group.create_array("channel", data=np.arange(len(channel_names), dtype=np.int16), chunks=(len(channel_names),))
    group.attrs.update(
        {
            "year": year,
            "layout": "time, channel, latitude, longitude",
            "field_array": "fields",
            "time_units": "datetime64[ns] stored as int64 nanoseconds since Unix epoch",
            "channel_names": channel_names,
            "area": config["area"],
            "times": config["times"],
            "units_note": "Raw FourCastNet ERA5 units are preserved: temperature K, geopotential m^2 s^-2, pressure Pa.",
        }
    )
    return group


def preprocess_year(year: int, overwrite: bool) -> None:
    config = load_config()
    channels = config["fourcastnet_channels"]
    channel_names = [c["name"] for c in channels]
    out_path = OUT_DIR / f"{year}.zarr"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_path}. Use --overwrite to rebuild.")

    single_paths, pressure_paths = assert_complete_year(year)
    n_time, lat, lon, time_values = infer_year_grid(single_paths, pressure_paths)
    shape = (n_time, len(channels), lat.size, lon.size)
    logging.info("Creating %s with shape=%s", out_path, shape)
    group = create_store(out_path, shape, lat, lon, time_values, channel_names, config, year)
    fields = group["fields"]

    start = 0
    for month, (single_path, pressure_path) in enumerate(zip(single_paths, pressure_paths), start=1):
        logging.info("Writing %04d-%02d", year, month)
        with xr.open_dataset(single_path) as ds_single, xr.open_dataset(pressure_path) as ds_pressure:
            month_n = ds_single.sizes["valid_time"]
            end = start + month_n
            for c_idx, channel in enumerate(channels):
                arr = read_channel(ds_single, ds_pressure, channel)
                expected_shape = (month_n, lat.size, lon.size)
                if arr.shape != expected_shape:
                    raise ValueError(f"Unexpected shape for {channel['name']}: {arr.shape}, expected {expected_shape}")
                fields[start:end, c_idx, :, :] = arr
            start = end

    if start != n_time:
        raise RuntimeError(f"Wrote {start} time steps; expected {n_time}")
    logging.info("Saved %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert monthly ERA5 raw files to annual FourCastNet20 Zarr tensor.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    preprocess_year(args.year, args.overwrite)


if __name__ == "__main__":
    main()
