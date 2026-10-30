import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import zarr
import os
import warnings
warnings.filterwarnings("ignore")


VARIABLE_ORDER = [
    "u10", "v10", "t2m", "sp", "msl", "t850",
    "u1000", "v1000", "z1000", "u850", "v850", "z850",
    "u500", "v500", "z500", "t500", "z50", "r500", "r850", "tcwv",
]

TRAIN_YEARS = [2014, 2015, 2016]
VAL_YEARS = [2017]
TEST_YEARS = [2018]


class _ChunkCache:
    """Tiny LRU cache for decompressed zarr chunks (one per worker process)."""

    __slots__ = ("maxsize", "_data", "_order")

    def __init__(self, maxsize=2):
        self.maxsize = maxsize
        self._data = {}
        self._order = []

    def get(self, key):
        return self._data.get(key)

    def put(self, key, value):
        if key in self._data:
            self._order.remove(key)
        self._data[key] = value
        self._order.append(key)
        if len(self._order) > self.maxsize:
            old = self._order.pop(0)
            self._data.pop(old, None)


class FieldDataset(Dataset):
    """Reads fields from zarr.

    Modes (resolution order):
      preload=True  -> loads the whole year into RAM as float16 (fast, ~2.7 GB/year).
      cache_dir     -> memory-maps prepared float16 .npy files (see prepare_memmap.py);
                       data stays on disk, near-RAM speed via OS page cache, low RSS.
      default       -> lazy zarr, one chunk decompressed and cached per worker
                       (lowest RSS but slowest).
    """

    def __init__(self, recon_zarr_path, orig_zarr_path, mean, std, preload=False, cache_dir=None):
        mean_b = np.asarray(mean[:, None, None], dtype=np.float32)
        std_b = np.asarray(std[:, None, None], dtype=np.float32)
        if preload:
            z_r = zarr.open(recon_zarr_path, mode="r")
            z_o = zarr.open(orig_zarr_path, mode="r")
            recon = np.asarray(z_r["fields"], dtype=np.float32)
            orig = np.asarray(z_o["fields"], dtype=np.float32)
            self.input = ((recon - mean_b) / std_b).astype(np.float16)
            self.target = ((orig - recon) / std_b).astype(np.float16)
            del recon, orig, z_r, z_o
            self._mode = "ram"
        elif cache_dir:
            year = os.path.basename(recon_zarr_path).replace(".zarr", "")
            in_path = os.path.join(cache_dir, f"input_{year}.npy")
            tg_path = os.path.join(cache_dir, f"target_{year}.npy")
            if not (os.path.exists(in_path) and os.path.exists(tg_path)):
                raise FileNotFoundError(
                    f"memmap data missing for {year}: expected {in_path} and {tg_path}. "
                    "Run prepare_memmap.py first."
                )
            self.input = np.load(in_path, mmap_mode="r")
            self.target = np.load(tg_path, mmap_mode="r")
            self._mode = "memmap"
        else:
            z_r = zarr.open(recon_zarr_path, mode="r")
            z_o = zarr.open(orig_zarr_path, mode="r")
            self.z_r = z_r["fields"]
            self.z_o = z_o["fields"]
            self.chunk_r = int(self.z_r.chunks[0])
            self.chunk_o = int(self.z_o.chunks[0])
            self.mean_b = mean_b
            self.std_b = std_b
            self._cache_r = _ChunkCache()
            self._cache_o = _ChunkCache()
            self._mode = "lazy"

    def __len__(self):
        if self._mode == "ram":
            return len(self.input)
        return int(self.z_r.shape[0]) if self._mode == "lazy" else self.input.shape[0]

    def _lazy_field(self, arr, chunk, cache, idx):
        ci, off = divmod(idx, chunk)
        buf = cache.get(ci)
        if buf is None:
            s = arr[ci * chunk:(ci + 1) * chunk]
            buf = np.asarray(s, dtype=np.float32)
            cache.put(ci, buf)
        return buf[off]

    def __getitem__(self, idx):
        if self._mode == "ram":
            return (torch.from_numpy(self.input[idx].copy()).float(),
                    torch.from_numpy(self.target[idx].copy()).float())
        if self._mode == "memmap":
            return (torch.from_numpy(np.asarray(self.input[idx], dtype=np.float32)),
                    torch.from_numpy(np.asarray(self.target[idx], dtype=np.float32)))
        recon = self._lazy_field(self.z_r, self.chunk_r, self._cache_r, idx)
        orig = self._lazy_field(self.z_o, self.chunk_o, self._cache_o, idx)
        x = (recon - self.mean_b) / self.std_b
        y = (orig - recon) / self.std_b
        return torch.from_numpy(x), torch.from_numpy(y)


def build_dataloaders(decoded_dir, orig_dir, stats_path, batch_size=8, num_workers=4,
                      preload=False, cache_dir=None, seed=0):
    stats = np.load(stats_path)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)

    generator = torch.Generator()
    generator.manual_seed(seed)

    splits = {
        "train": TRAIN_YEARS,
        "val": VAL_YEARS,
        "test": TEST_YEARS,
    }
    loaders = {}
    for split, years in splits.items():
        datasets = []
        for y in years:
            recon_path = f"{decoded_dir}/{y}.zarr"
            orig_path = f"{orig_dir}/{y}.zarr"
            datasets.append(FieldDataset(recon_path, orig_path, mean, std,
                                         preload=preload, cache_dir=cache_dir))
        concat = ConcatDataset(datasets)
        loaders[split] = torch.utils.data.DataLoader(
            concat,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            generator=generator,
        )
    return loaders, mean, std
