# Full-year 2022 data preparation

All outputs are independent of the original 2014–2018 caches and results. No model training is performed.

## Sources and exactness checks

* CRA5: official `taohan10200/CRA5-Dataset` Hugging Face dataset, repository revision pinned in `data/cra5/2022_data_manifest.json`. Exactly 1,460 six-hourly timestamps are selected. Every file is verified against its publisher LFS SHA256, including resumed files.
* ERA5 native fields: Google's public ARCO ERA5 `ar/full_37-1h-0p25deg-chunk-1.zarr-v3`. HTTP ranges retrieve only the existing Blosc blocks containing the requested pressure levels and latitude rows. Every request after the header is pinned to the Google object generation. The source values are never regridded. Range hashes, source generation, units and metadata are preserved in frame receipts.
* Native relative humidity: NCAR GDEX `d633000/e5.oper.an.pl/YYYYMM/e5.oper.an.pl.128_157_r.ll025sc.YYYYMMDD00_YYYYMMDD23.nc`. Official THREDDS OPeNDAP server-side subsets preserve native ECMWF RH in percent. Both levels and all four daily timestamps were verified bitwise against a separate HTTPS HDF5 range extraction. Every day's returned time, pressure-level, latitude and longitude arrays are independently validated and saved. Unit/calendar schemas were initially checked per day; following transient 503/504 metadata-server responses, subsequent execution verifies and caches the schema once per calendar month for the same ERA5 parameter family. Each receipt records the actual metadata-source URL, SHA256 and monthly scope; it does not imply a fresh per-day metadata request. The earlier HDF5 range reader is retained as a fallback; its completed outputs receive metadata-only OPeNDAP verification. WeatherBench's derived RH is excluded.
* Native ERA5 total column water vapour is truth. CRA5 TCWV uses the original pressure-clipped specific-humidity integral.

The extraction grid is checked against latitude 49 to 25 degrees descending, longitude -130 to -70 degrees ascending, both at 0.25 degrees: 97 × 241. Geopotential remains in m²/s². Original ERA5 truth is compared with a new-source 2018-01-01 00 UTC frame in `2022_source2018_validation.json`: native NCAR r500/r850 are bitwise identical; differences in the other 18 variables are small packing differences, with maximum temperature difference 0.000733 K, wind difference 0.000813 m/s, surface-pressure difference 0.453125 Pa and geopotential difference 0.140625 m²/s². Partial Blosc extraction was separately verified bitwise against full-chunk decompression for all 18 fields in that frame.

## Execution on the existing Ubuntu host

Scripts live under `/home/tong/fourcastnet_us_mpca/revision_2026/scripts/`; data under `revision_2026/data/`; logs under `revision_2026/logs/`.

1. `download_2022_cra5.py --out revision_2026/data/cra5` downloads and validates all bitstreams.
2. `run_2022_era5.sh` starts eight disjoint timestamp shards. `era5_2022/{timestamp}.npy` deliberately has NaN only in r500/r850 pending separate RH assembly. These are incomplete intermediate data, not evaluation-ready 20-channel truth. `aggregate_2022_truth.py --wait` checks all 1,460 data/receipt hashes and writes the global manifest while explicitly retaining missing RH.
3. `run_2022_native_rh.sh` starts eight disjoint OPeNDAP day shards. Bounded worker wrappers resume stopped shards after a 60-second backoff; each HTTP request also has bounded retries. `native_rh_2022/YYYYMMDD_rh.npz` contains four timestamps, two channels (`r500`, `r850`), and 97 × 241 grids. Metadata cache files retain the precise source response under `native_rh_metadata_2022/`.
4. `run_2022_decode.sh --wait-download` uses the original `.venv_cra5`, original `CRA5_src` PYTHONPATH, original CUDA allocator setting, original decoder helpers, and original codec weight SHA256 `812a6d09ff039e4ab468915942e7fd00f11e7c58b839cd1b60fe1d0586a40dad`. It waits for complete checksum-verified bitstream provenance before loading the model, then saves atomic decoded frames and receipts.
5. The separately maintained `assemble_2022_cache.py` must verify every receipt and all 1,460 exact timestamp pairs before combining the intermediate truth and native RH, applying frozen statistics, or permitting evaluation.

Use `status_2022_data.py` for a lightweight presence inventory. Presence alone does not establish hash or scientific validity; final assembly must revalidate hashes and full-year alignment.

Every job supports resume. Restart only stopped shards to avoid concurrent writes to the same timestamp. Existing successful per-frame data are hash-checked. Atomic `.part` files do not count as completed data. No partial evaluation may be reported as a full-year experiment.

Completed preparation: all 1,460 CRA5 bitstreams passed publisher SHA256 checks, with zero errors (`2022_cra5_download_manifest.json`); all 1,460 ERA5 native-18 frames passed aggregate data/receipt hashes, timestamp, shape, finiteness and source-generation proof checks (`2022_truth_preparation_manifest.json`); and all 1,460 original-codec decoded frames finished (`2022_decoded_preparation_manifest.json`). Independent validation confirmed all 365 native-RH days / 1,460 frames, including every data/receipt hash, exact daily timestamps and coordinates, units, fill-value exclusion and the actual daily/monthly metadata origins (`2022_native_rh_validation.json`). The intermediate native-18 files retain explicit NaN RH slots; their combination with validated native RH belongs to the final strict assembly stage.

The first four exact 2022 truth/decoded frame pairs also passed the data/codec smoke checks (`2022_four_frame_smoke.json`). That smoke script runs no U-Net inference and is not full-year evaluation. Frozen normalization, U-Net evaluation and result exports are handled by the separate final pipeline.

A single contiguous-pressure optimization probe timed out and was not adopted. Production preserved the exact sparse pressure-level query and eight-worker limit. Transient 503/504 metadata responses prompted monthly schema caching and bounded process-level resume wrappers; successful per-day data were retained and all final source checks passed.

## Codec training-period evidence

The published CRA5 methods describe full-resolution training during 1979–2021 and testing during 2022–2023: [Han et al., Communications Earth & Environment](https://www.nature.com/articles/s43247-025-02903-z), “ERA5 dataset.” The local original source revision is `2b9e06d8ca31f7b27c5039c9b5fc3334a43b4976`; its model zoo maps the 268-channel decoder to the official pretrained-weight URL. The exact local weight hash is recorded above and in every decoded frame receipt. This ties the execution to the original local released artifact; an independently signed training manifest for that weight is not supplied by the upstream release.
