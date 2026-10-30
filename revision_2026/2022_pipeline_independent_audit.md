# Independent 2022 pipeline audit — 2026-10-26

Scope: source inspection of `assemble_2022_cache.py`, `evaluate_frozen.py`,
`prepare_2022_era5.py`, `prepare_2022_native_rh.py`, `decode_2022_cra5.py`,
the original decoder/cache preparation, and saved actual 2018 source-validation
receipts. This report includes the initial implementation review and the later
**executed full-year acceptance audit**, which passed after actual 2022 assembly,
frozen evaluation and bootstrap completed. See the executed acceptance section
at the end; the initial review is preserved to trace concerns and corrections.

## Channel, grid and time mapping

The canonical truth order is
`u10,v10,t2m,sp,msl,t850,u1000,v1000,z1000,u850,v850,z850,u500,v500,z500,t500,z50,r500,r850,tcwv`.
The original CRA5 helper uses corresponding pressure-level aliases with an
underscore, such as `t_850`. The assembler explicitly strips underscores from
decoder receipts before checking the full 20-channel sequence against training
statistics. Decoder selection follows that original helper's named index map;
the 19 direct channels fill slots 0–18 and derived `tcwv` fills slot 19.

Both ARCO extraction and native-RH extraction assert the actual coordinate slices
49°N through 25°N, descending at 0.25°, and 230°E through 290°E (converted to
−130° through −70°). The 97×241 crop is rows 164:261 and columns 920:1161, exactly
the original decoder crop. Pressure levels are found by numeric equality rather
than fixed positions: 1000, 850, 500 and 50 hPa where requested. No longitude
rolling, interpolation, pressure interpolation, or geopotential/height conversion
is applied to the direct model fields.

For 2022, canonical timestamps are constructed as exactly 1460 six-hour frames
from 2022-01-01T00:00:00 through 2022-12-31T18:00:00. ARCO chunk indices use the
metadata time origin; native-RH extraction checks requested hours against the
archive's hourly time coordinate. The newly hardened native-RH script also
asserts the actual `hours since 1900-01-01 00:00:00` units and Gregorian calendar.
Assembler receipt timestamps and native-RH timestamp strings must equal each
canonical timestamp. Decoder final timestamps must equal the complete ordered
canonical list, so missing, duplicated and reordered frames cannot pass assembly.

The four native-RH shards use starting days 0,1,2,3 with step 4 over days 0–364,
which partitions all 365 days without overlap or omission.

## ARCO Blosc partial extraction

The extractor checks float32 little-endian, C-order, no filters and expected
single-time chunk shapes before interpreting chunk offsets. It retrieves the
Blosc header/block table, then every compressed block intersecting the requested
latitude rows and pressure level. A compressed-buffer replica preserves original
block offsets. `blosc_getitem` receives original element indices and a count of
97×1440 values per selected level; longitude cropping happens after decompression.
There is no element/byte-index confusion: block selection multiplies element
indices by four, while `blosc_getitem` uses element indices. Each selected output
is checked finite.

HTTP range requests require 206 status and exact response length. ARCO object
generation is pinned after the header read for subsequent reads of that object.
Source metadata, compressed range hashes, header/table hash, object generation,
URL and units are saved in receipts. The supplied full-decompression validation
compares selected outputs with full decompression of the same object generation.

Actual saved `2022_source_validation/era5_partial_full_receipt.json` is for
2018-01-01T00:00:00 and reports `partial_equals_full: true` for every extracted
ARCO variable used for the 18 available channels. This validates the custom
range extraction at that timestamp; it is not a validation of every 2022 object.

## Native RH and HDF5 reads

The RH source is the NCAR GDEX native ECMWF ERA5 relative-humidity daily archive,
variable `R`. Relative humidity is read directly in percent at the native 500 and
850 hPa levels. It is not inferred from specific humidity and temperature.
The extracted shape is 4×2×97×241; hours are 00,06,12,18. No percent/fraction
conversion or clipping is performed. Producer checks cover finite values,
coordinates, level equality, time equality and `%` units.

The random-access wrapper exposes original HDF5 bytes to h5py and lets HDF5
interpret chunking/dimension storage; it does not manually guess compressed
chunk-to-pressure offsets. The wrapper checks the HDF5 v0 signature and size
format, reads the EOF address from that supported superblock layout, and requires
exact byte counts for every range. The first 2018 source comparison gives
bitwise equality for both native RH fields, providing direct evidence against
an unnoticed level, time, or crop offset for the validated frame.

Range receipts record the source URL, date, output hash and fetched byte-range
hashes. Unlike GCS generation-pinned reads, NCAR reads do not pin a source object
version/ETag. The archive is treated as stable during a daily read; this is a
remaining provenance limitation, not evidence of observed inconsistency.
Existing daily receipts do not persist the full source coordinate arrays/time
metadata. Those are checked by the producer and corroborated by the actual 2018
comparison, but an independent consumer has weaker coordinate trace than ARCO.

## Actual 2018 source-validation evidence and precision

`2022_source2018_validation.json` compares the first 2018 source frame with the
original processed 2018 ground truth across all 20 channels. RH500 and RH850 are
bitwise identical (RMSE and maximum absolute difference exactly zero). The other
18 channels agree to small archive-packing differences, not bitwise identity.
Examples in original physical units:

| Channel | RMSE | Maximum absolute difference |
|---|---:|---:|
| t2m (K) | 0.000416490 | 0.000732422 |
| sp (Pa) | 0.249442 | 0.453125 |
| z50 (m²/s²) | 0.0751970 | 0.140625 |
| tcwv (kg/m²) | 0.000347754 | 0.000606537 |
| r500, r850 (%) | 0 | 0 |

This is persuasive evidence that variable identity, units and indexing match the
original source. The 18-channel archive precision difference must be disclosed
when comparing 2018 and 2022; the source reanalysis values are not bitwise
identical across storage distributions. One midnight frame is a source-mapping
check, not a full-year source-equivalence proof or a substitute for 2022 testing.

## Decoder identity and derived tcwv

The decode wrapper invokes the original `.venv_cra5`, original `decode_conus`
helper and original SDPA patch. The codec weight is pinned to SHA-256
`812a6d09ff039e4ab468915942e7fd00f11e7c58b839cd1b60fe1d0586a40dad`.
Each source bitstream is checked against the pinned Hugging Face LFS hash before
new decoding. That manifest is generated from a repository commit and an ordered
selection of exactly the canonical timestamp filenames.

Decoded `tcwv` uses the unchanged original pressure-clipped trapezoidal integral
of specific humidity divided by g. Native ERA5 total-column water vapour remains
the truth channel; it is not replaced with an RH-derived or pressure-level
approximation. This preserves the original derived-channel convention. Hashing
the local CRA5 weights verifies file identity but does not independently establish
the upstream model training period; the separate upstream mapping requirement
still applies.

## Normalization, completion gates and evaluation

Assembly checks all required frame files/receipts before creating the new output
directory. It validates complete ordered decoder timestamps, 1460 verified source
bitstreams, per-frame receipt hashes/timestamps, decoded channel order, codec
weight and bitstream hashes. RH receipts must match their daily file hash/date
and percent units; each daily array's channel sequence, shape and each selected
timestamp are checked. The prepared ARCO frame's RH slots may intentionally be
NaN under `--skip-rh`; they are replaced with native RH before the complete-frame
finite check. Thus a nominally completed 18-channel ARCO acquisition alone cannot
be evaluated as complete 20-channel truth.

Truth, reconstruction, mean and std use float32 arithmetic, followed by float16
input/target storage. This matches the original `prepare_memmap.py` operation
order. Positive standard deviations, correct statistics channel order, finite
physical arrays and finite normalized output are checked. No statistics are fit
on 2022. The final complete manifest is written only after both caches flush,
timestamp CSV completion and input/target/CSV hashing.

For non-2018 evaluation, the newly hardened evaluator requires a complete aligned
1460-frame manifest with matching channel names, year, statistics SHA-256 and
input/target/timestamp-file hashes. It then uses frozen float32 inference and
float64 SSE accumulation over every frame/cell, retaining the same 2018 metric
definition and separate direct19/all20/near/upper/tcwv outputs. Checkpoint file
identity is hashed before and after inference. The intended invocation must use
the original CRA5 corrector, not another CLI-specified checkpoint. The original
2018 scientific results do not need rerunning: the added guards affect only the
non-2018 input-acceptance branch.

## Resume hazards found and disposition

1. **Decoder all-skipped resume bug — corrected during audit.** Previously the
   final `status.update` referenced an uninitialized variable if every frame was
   already present. A status dictionary is now initialized before the loop.
2. **Decoder weak skip provenance — corrected during audit.** Resume now verifies
   timestamp, codec weight hash, bitstream hash and original channel list in
   addition to output hash. Malformed receipts fall back to regeneration.
3. **Native-RH interruption risk — corrected during audit.** Daily NPZ and receipt
   are now written to temporary files and renamed atomically. Resume reads catch
   malformed JSON and verify the relevant receipt before accepting cached data.
4. **Cache incomplete/mismatched acceptance — corrected during audit.** Assembly
   and non-2018 evaluation gained the full manifest/provenance checks above.
5. **Interrupted assembly intentionally does not resume in place.** A fresh
   directory is required; a partial directory without its final complete manifest
   cannot pass the evaluator. This trades convenience for preserving reviewable
   artifacts. Producers and assembly must remain sequenced: no assembly while
   truth/RH/decoder producers are still writing their relevant outputs.

The producer fixes were communicated to and implemented by the data agent; the
assembler/evaluator fixes were implemented by the root agent. This audit did not
edit those shared producer/consumer scripts or modify original experiment files.
The data agent reports that already-running RH workers retain their earlier
loaded Python source; future starts/resumes use the hardened code. Assembly must
therefore independently reject finite RH fill sentinels (the producer's broad
`abs(RH)<1000` guard) in addition to shape/date/hash/finite checks. The producer
fixes alone do not retroactively validate daily files already written by those
workers. This final consumer guard was communicated to the root agent.

Initial review disposition: the pipeline was consistent with the original
20-channel protocol while full-year readiness still required execution.
**That execution and independent acceptance are now complete**, as follows.

## Executed full-year acceptance — passed

The independent script `scripts/verify_2022_acceptance.py` was actually executed
against completed evidence. Its output is `2022_acceptance_verification.json`.
The acceptance checks covered:

* Exactly **1460** CSV rows with indices 0–1459 and the exact ordered six-hour
  sequence 2022-01-01T00:00:00 through 2022-12-31T18:00:00. Missing timestamps:
  **0**; duplicates: **0**. The CSV hash matches the completed cache manifest.
* `2022_cache_manifest.json`, the published result manifest and the archived
  manifest are identical. Completion, year/frame count and channel identity pass.
* All **1460 truth receipts**, **1460 decoded receipts** and **365 native-RH
  receipts** in `provenance_2022.tar.gz` match the timestamp CSV and final producer
  manifests. Truth/RH receipt byte hashes match final validation records. Every
  decoder receipt matches the expected codec weight, channel order and corresponding
  pinned Hugging Face LFS bitstream hash. Every RH daily receipt has the exact four
  timestamps, 500/850 hPa levels, 97×241 coordinates, percent units and Gregorian
  hourly time metadata.
* Final native-RH validation reports all daily data/receipt hashes checked, all
  fields finite and fill sentinels excluded. Source-attribute provenance retains
  **130 days with same-day metadata** and **235 days using the same native ECMWF
  parameter 157 file family/calendar month**. The latter is not represented as
  a same-day source-attribute fetch. Every daily output's spatial/time coordinate
  arrays remain checked.
* Executed 2022 and 2018 summaries contain the **identical original U-Net and
  training-statistics hashes**, with 1,646,548 parameters. No checkpoint was
  trained or selected on 2022.
* Actual saved temporal SSE arrays are 1460×20 with 97×241 valid cells per entry,
  finite nonnegative errors, and spatial-map/temporal-sum agreement. All **20
  per-channel** errors and **five group** metrics were independently recomputed
  from actual SSE and agree with CSV/full-precision JSON. Direct19 additionally
  equals the 13-near/6-upper channel-count-weighted nRMSE aggregate.
* Bootstrap point estimates match the same recomputed group metrics and lie
  inside their saved intervals. The **negative v500 gain is preserved**:
  −0.00819981923527724%, rather than clipped into a positive result.

| Actual 2022 quantity | Before nRMSE | After nRMSE | Gain (%) |
|---|---:|---:|---:|
| Direct19 | 0.08624524222992214 | 0.0634785885620568 | 26.397576352294838 |
| All20 | 0.08700187670230002 | 0.062361176349367296 | 28.322033141017645 |
| Near | 0.09843221923916347 | 0.06615310500271751 | 32.793240349500294 |
| Upper | 0.05984012537656593 | 0.05768380294062528 | 3.603472456601986 |

The direct19 2022 temporal block-bootstrap 95% interval is
25.904632121972778–26.92780835767633%. Its conditional interpretation follows
`analysis_independent_audit.md`.

This independent pass inspected the complete local receipt chain and actual
sufficient statistics. Large original frame/cache byte hashes were verified by
the executed producer, assembler and evaluator as recorded in complete manifests;
those large files were not downloaded and rehashed again locally. This is an
independent evidence-chain/statistics audit, not a redundant second decoder run.

`data_manifest.json` now has the verified 2022 entry with source paths, cache
hashes and acceptance evidence. The five 2014–2018 records were preserved exactly
as JSON objects. The result satisfies full-year rather than partial-year or
smoke-test acceptance.
