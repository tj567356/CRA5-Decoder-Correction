"""Provenance for prepared memmap caches.

A memmap directory is just ten unlabelled .npy files, and `--cache_dir` bypasses
`--energy` entirely in train_cnn.py -- so pointing a run at the wrong cache trains
on the wrong data without any error. The fix is to make the data self-describing:
prepare_memmap.py drops a manifest.json here, train_cnn.py reads it, checks it
still matches the files on disk, prints where the data came from, and copies the
whole thing into test_results.json.

Note the invariant this checks: the cache must match *the decoded directory it was
built from*, recorded at build time. It deliberately does NOT check the cache
against `--energy`. That would be wrong -- a greedy-allocation run legitimately
carries `--energy e995` (the flag's choices are closed) while its data comes from
decoded/greedy_equal_n2011 -- and a check that fires on correct runs just teaches
you to pass --skip-check forever.

Backfill for directories built before manifests existed:

    python memmap_manifest.py --backfill ../../data/memmap_e99
    python memmap_manifest.py --backfill-all

Backfill does not trust the directory name: it de-normalises frame 0 and matches
it against every decoded/* candidate, so the recorded provenance is the one the
bytes actually agree with.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MANIFEST = "manifest.json"


def _fingerprint(arr):
    """sha1 of frame 0 -- exact, and one frame is cheap to re-read."""
    return hashlib.sha1(np.ascontiguousarray(arr[0]).tobytes()).hexdigest()


def describe(cache_dir, years):
    """Shape + frame-0 hash for each year's input/target pair."""
    cache_dir = Path(cache_dir)
    out = {}
    for y in years:
        entry = {}
        for kind in ("input", "target"):
            p = cache_dir / f"{kind}_{y}.npy"
            a = np.load(p, mmap_mode="r")
            entry[kind] = {"shape": list(a.shape), "dtype": str(a.dtype),
                           "bytes": p.stat().st_size, "sha1_frame0": _fingerprint(a)}
        out[str(y)] = entry
    return out


def write(cache_dir, decoded_dir, orig_dir, stats_path, years, extra=None):
    cache_dir = Path(cache_dir)
    m = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "decoded_dir": str(Path(decoded_dir).resolve()),
         "orig_dir": str(Path(orig_dir).resolve()),
         "stats_path": str(Path(stats_path).resolve()),
         "input_is": "(recon - mean) / std",
         "target_is": "(orig - recon) / std",
         "years": describe(cache_dir, years)}
    if extra:
        m.update(extra)
    (cache_dir / MANIFEST).write_text(json.dumps(m, indent=2))
    return m


def verify(cache_dir, years, allow_missing=False):
    """Load the manifest and confirm the files still match it. Returns the manifest."""
    path = Path(cache_dir) / MANIFEST
    if not path.exists():
        msg = (f"{path} is missing -- this cache has no recorded provenance, so there is "
               f"no way to tell which decoded fields it was built from. Run "
               f"`python memmap_manifest.py --backfill {cache_dir}` first, "
               f"or pass --allow_unlabelled_cache to skip this check.")
        if allow_missing:
            print(f"[WARN] {msg}")
            return None
        raise FileNotFoundError(msg)

    m = json.loads(path.read_text())
    now = describe(cache_dir, years)
    for y, entry in now.items():
        was = m["years"].get(y)
        if was is None:
            raise ValueError(f"{path}: no entry for {y}, but {cache_dir}/input_{y}.npy exists")
        for kind in ("input", "target"):
            for key in ("shape", "sha1_frame0"):
                if was[kind][key] != entry[kind][key]:
                    raise ValueError(
                        f"{cache_dir}/{kind}_{y}.npy no longer matches {MANIFEST} "
                        f"({key}: manifest {was[kind][key]} vs file {entry[kind][key]}). "
                        f"The cache was rebuilt or overwritten -- re-run prepare_memmap.py.")
    return m


def identify(cache_dir, year=2018, base=None):
    """Which decoded/* directory do these bytes actually come from?

    Undoes the normalisation on frame 0 and compares against every candidate. The
    true one lands orders of magnitude below the rest, so the answer is unambiguous.
    """
    import zarr
    base = Path(base or os.path.expanduser("~/fourcastnet_us_mpca"))
    stats = np.load(base / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz")
    mean = stats["mean"].astype(np.float32)[:, None, None]
    std = stats["std"].astype(np.float32)[:, None, None]

    a = np.load(Path(cache_dir) / f"input_{year}.npy", mmap_mode="r")
    recon = a[0].astype(np.float32) * std + mean

    scores = {}
    for cand in sorted((base / "decoded").iterdir()):
        f = cand / f"{year}.zarr"
        if not f.exists():
            continue
        ref = np.asarray(zarr.open_group(str(f), mode="r")["fields"][0], dtype=np.float32)
        denom = np.abs(ref).mean() or 1.0
        scores[cand] = float(np.abs(recon - ref).mean() / denom)
    if not scores:
        raise RuntimeError(f"no decoded/*/{year}.zarr candidates found under {base}")
    best = min(scores, key=scores.get)
    return best, scores


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=Path)
    ap.add_argument("--backfill-all", action="store_true")
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--years", type=int, nargs="+", default=[2014, 2015, 2016, 2017, 2018])
    args = ap.parse_args()

    base = Path(os.path.expanduser("~/fourcastnet_us_mpca"))
    targets = ([args.backfill] if args.backfill else
               sorted(d for d in (base / "data").iterdir()
                      if d.is_dir() and d.name.startswith("memmap_")))

    for d in targets:
        best, scores = identify(d, args.year, base)
        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        runner = ranked[1][1] if len(ranked) > 1 else float("inf")
        print(f"\n{d.name}")
        for cand, s in ranked[:3]:
            print(f"   {cand.name:24s} {s:.3e}" + ("   <-- matched" if cand == best else ""))
        # The true source round-trips through float16 only (~4e-6 relative); every
        # other energy level sits at 1e-4 or worse. Demand both an absolute pass and
        # a clear gap, rather than guessing from a narrow win.
        if ranked[0][1] > 1e-5 or runner < 10 * ranked[0][1]:
            raise RuntimeError(f"{d}: match is not decisive ({ranked[0][1]:.2e} vs "
                               f"{runner:.2e}) -- refusing to guess the provenance")
        if best.name != d.name.replace("memmap_", ""):
            print(f"   [!] directory name says '{d.name.replace('memmap_', '')}' "
                  f"but the bytes say '{best.name}'")
        write(d, best, base / "data/processed/fourcastnet20_conus",
              base / "mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz",
              args.years,
              extra={"backfilled": True,
                     "identified_by": f"content match on {args.year} frame 0",
                     "match_score": ranked[0][1], "runner_up": ranked[1][0].name,
                     "runner_up_score": runner})
        print(f"   wrote {d}/{MANIFEST}")


if __name__ == "__main__":
    main()
