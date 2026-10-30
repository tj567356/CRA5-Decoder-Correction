# Frozen baseline replay, 2026-10-26

All five systems use the full 2018 test year (1460 frames), the existing
2014–2016 channel statistics, and distinct previously trained U-Net weights.
No training, fine tuning, checkpoint search, or rate search was performed.
`revised_table1.csv` and `.json` preserve full precision; all group errors are
arithmetic means of per-channel RMSE divided by the training channel standard
deviation. `G_pct` is 100(1−sum(after)/sum(before)), rather than a mean of
per-channel percentage gains. Near and upper contain 13 and 6 direct channels.
The derived `tcwv` is excluded from direct19 and from both groups.

## Checkpoints and training provenance

* CRA5: `checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/best.pt` under
  `/home/tong/fourcastnet_us_mpca`.
* MPCA-only: `/home/tong/new_20/checkpoints/s99_stage1only/unet_d2_c64_lr3e-4_clip1/best.pt`.
  Its existing train log records a cache built from
  `/home/tong/new_20/decoded/s99_stage1only` on 2026-08-21, with 4384 training,
  1460 validation and 1460 test examples; 1,646,548 parameters; AMP enabled;
  maximum 200 epochs. `reconstruct_stage1_only.py` documents and implements the
  stage-1 projection/reconstruction without second-stage PCA. Thus this corrector
  was trained on the stage-1-only reconstruction, and reused at the rate-matched
  quantized deployment rate. It was not retrained for Table 1 or this revision.
* e995, e99, e97: distinct paths
  `checkpoints/{energy}/unet_d2_c64_lr3e-4_clip1/best.pt` under the main project.
  These are separate per-compressor weights, not the CRA5 or MPCA-only weights.
  Their SHA-256 values are included in every result and in the revised table.
  An inspected e995 train log reports 4384/1460/1460 examples and maximum 100
  epochs; the original blanket 200-epoch description must not be applied to all
  historical baseline checkpoints.

## Replayed rate-matched MPCA-only

`baseline_mpca_frozen.py` derives its executable source from the original
`rate_matched_val.py`, removes the entire bisection, and fixes the previously
stored 2017-selected step at **0.7659035172375452**. It retains training/validation
coder-support construction. Frozen-step encoding is repeated for 2017 and 2018;
the step is never updated. The 2018 mean coded rate, pre/post nRMSE, and all
previously stored group gains reproduce the original result exactly to displayed
floating-point precision:

* 473.11514887104664×;
* 0.12728844380005333 before, 0.08687819856088126 after;
* near 0.1291303021728192 → 0.08206987614872399;
* upper 0.12288826830437623 → 0.09690387477239093.

The coder round-trip checks at first/middle/last frames succeed. All 1460 frames
contribute to rate and distortion. Original float32 physical reconstruction,
float16 correction autocast, and float64 distortion aggregation are preserved.

## Quantized MPCA+PCA correction

The historical `generality` analysis evaluates unquantized cached reconstruction
errors, whereas `pca_codec_all.json` measures the entropy-coded quantized rate.
These are not exactly the same reconstruction. The historical
`codec_all_cnn.py` separately evaluated quantized reconstructions, but did not
save the requested group baseline errors.

This revision first replays the unquantized caches and preserves those values in
`e995.json`, `e99.json`, and `e97.json`. It then evaluates **actual quantized
reconstructions** at exactly the previously stored steps, with the fixed known
checkpoint path, using `baseline_pca_quantized.py`. The original script's
test-based checkpoint reproduction/selection loop is removed. The resulting
`pca_quantized/*.json` supplies the revised table, pairing the measured rates with
the matching quantized reconstruction. No retained energy, quantizer, basis,
checkpoint, or hyperparameter is changed. Quantized gains remain near/upper
20.9163/15.5024%, 18.8942/14.3628%, and 14.3930/10.8899% for e995/e99/e97.

## Accounting and interpretation

CR values are the original measured per-frame entropy-coded rates, with the
original amortized coder sigma storage. They are not coefficient-count estimates.
Correction weights are shared deployment overhead, excluded from per-frame
bitstreams; the rate figure also does not charge full basis/model deployment
storage. CRA5 is a global 268-channel codec; MPCA is a CONUS 20-channel codec.
Mean-rate matching therefore does not make their encoding tasks identical.
The other three MPCA+PCA rates (1333.22×, 2000.48×, 5093.14×) are non-matched.

Different existing inference precision paths are recorded explicitly. The cache
replays use float32 inference from the original float16 caches; quantized MPCA
replays use the original float16 autocast and float32 physical fields with
float64 aggregation. Tiny differences from rounded historical tables are not
model improvements. The historical result files were never opened for writing.
Every new result, log, effective script and source hash is in an independent
`results/revision_2026/baselines` directory. Large per-time sufficient statistics
for the unquantized baseline replay remain on the original Ubuntu host.
