# Independent audit: parameter-matched bottleneck RF ablation

Audit date: 2026-10-29 (Asia/Taipei). Auditor: separate `rf_ablation_audit`
agent context. This report distinguishes executed local structural checks from
source review and from scientific results still pending at this audit.

## Evidence status

- Executed locally: Python syntax/AST parsing; Original UNet2 class comparison
  against Git baseline `13b9c866`; independent analytic parameter count; exact
  structural RF dependency propagation and assertions.
- Reviewed as source: new DilatedUNet2, training/selection/evaluation/cost script,
  gradient ERF script, and the prespecified protocol.
- Not independently executed by this auditor: GPU forward/shape preflight,
  training, frozen annual performance, gradient ERF, or timing measurements.
  The main execution lane retains actual GPU evidence. No expected improvement
  is inferred from parameter matching or dilation.
- At this review, training/evaluation/ERF results are **pending independent
  acceptance**. No performance number is certified here.

## Original preservation and intervention

The current `UNet2` class AST exactly matches the class in Git baseline
`13b9c866:research/fourcastnet/scripts/cnn/model.py`. Adding a sibling subclass
changes the module file hash but does not alter the Original UNet2 class.

`DilatedUNet2` calls the original constructor and inherits the original forward.
Only `bottleneck.0` and `bottleneck.2` Conv2d dilation/padding metadata change
from (1,1) to (2,2) and (3,3), respectively. It does not replace or resize weight
tensors, consume extra initialization randomness, add activations, or change
encoder/decoder/skip operations. Resetting the same seed before each constructor
therefore produces matching initial tensors/state keys. The execution script
asserts that equality, original class AST equality, parameter counts, only the
two intended convolution differences, and B=1/B=8 output shapes.

Independent analytic parameter count (kernel-3 weights plus biases):

| Block | Parameters |
|---|---:|
| enc0 | 48,512 |
| down0 | 73,856 |
| enc1 | 295,168 |
| down1 | 147,584 |
| bottleneck | 295,168 |
| up0 | 147,584 |
| dec0 | 442,624 |
| up1 | 73,792 |
| dec1 | 110,720 |
| head | 11,540 |
| **Total, either model** | **1,646,548** |

The source asserts the original checkpoint and training-statistics SHA256 before
execution and again after evaluation. New output directories are exclusive;
the original remote model source/checkpoint are read-only inputs.

## Exact theoretical receptive field: executed

`scripts/rf_theory.py` tracks reachable input-index sets through every spatial
operation, including the convolution in each down block, the convolution before
each resize, floor MaxPool2d, bilinear `align_corners=False` with exact rational
coordinates, skip unions, decoder convolutions and final kernel-3 head. Constant
zero padding is excluded from real-input support. MaxPool tracks every possible
winner, not only the winner realized on a particular input.

| Structural RF quantity | Original | Dilated |
|---|---:|---:|
| Actual 97x241 central-output bounding support | 72x72 | 96x96 |
| Central output (row,column) | (48,120) | (48,120) |
| Central input row bounds, inclusive | 12..83 | 0..95 |
| Central input column bounds, inclusive | 84..155 | 72..167 |
| Maximum per-axis span without domain clipping | 76 | 100 |
| Maximum axis spans on actual 97x241 domain | 76x76 | 97x100 |

A 513-point artificial axis (same odd-size floor-pool behavior) verifies the
unclipped maxima away from domain boundaries. Support changes with output phase
because of pooling and resizing; reporting only a single dilation multiplier
would be inaccurate. All support sets in the tested axes are contiguous.

These quantities are in input-grid pixels. They describe possible structural
dependencies, not physical kilometres, independent observations, or proof of
actual trained use. Dilated potential height is clipped by the 97-row domain.

## Training, selection and performance source review

The script uses original CRA5 residual caches, not MPCA reconstruction caches.
Training opens only 2014-2016 arrays; validation opens only 2017. The selected
checkpoint record is persisted before either test-year array is opened.

The reviewed recipe matches inspected original training code: seed0 for Python,
NumPy and Torch/CUDA; cuDNN deterministic=True/benchmark=False; AdamW lr3e-4,
weight decay1e-4 and default betas/eps; batch8; four workers; train shuffle;
pin_memory=True/drop_last=False; fp16 autocast plus GradScaler; gradient clip1;
max200 epochs; patience12. Validation selection uses the original mean of batch
normalized MSE values (equal weight on its partial final batch), not validation
nRMSE or physical-unit RMSE. No architecture or hyperparameter search occurs.

Frozen annual evaluation uses float32 forward passes and float64 time/channel
SSE, matching the established evaluation arithmetic. Both years must have1460
frames. The2022 cache manifest, input/target/timestamp hashes, channel order and
training-statistics hash are checked. Direct19 excludes tcwv; near13/upper6
memberships match the existing analysis. Group gains, near/upper ratio,
near-minus-upper percentage points and per-channel gain formulas are correct.
No test-based checkpoint reselection appears in the reviewed source.

Original and Dilated inference costs are measured in one run on the same fixed
GPU-resident2017 batch, warmup/timed counts and CUDA event method. Dilation does
not change convolution parameter shapes/FLOP counts but can change runtime.
Allocated GPU memory includes other still-resident tensors; it must not be
represented as a clean isolated-model memory requirement without accounting for
the allocation baseline. Historical Original training time is separate evidence.

## Gradient effective RF source review

`rf_erf.py` uses eight fixed, evenly spaced2017 validation frames, all20 output
channels and three fixed positions:480 scalar probes per model. Test-year field
arrays are not read; performance/selection JSON files are read only as completion
and frozen-checkpoint guards.

- Differentiates each individual correction-output scalar `f(x)`, not the final
  identity-plus-correction `x+f(x)`, and does not sum output positions together.
- `retain_graph=probe<60` is correct for3 positions times20 channels on each
  shared forward graph. The final probe frees the graph.
- Sums squared input gradients over20 standardized input channels, using float64
  before squaring, then normalizes each nonzero probe to unit mass. This avoids
  large-gradient channels automatically dominating averaged ERF maps.
- Records zero-gradient probes rather than replacing them with fictitious maps.
- Checks mass outside each probe's own structural support and measures Dilated
  mass outside the corresponding Original theoretical support.
- Computes50/90/95% energy radii correctly using Chebyshev distance in pixels
  from the output position; these are radii, not receptive-field diameters.
- Stores individual probe CSV and group/position-specific mean maps. Aggregate
  radius summaries combine positions, so interpretation must acknowledge that
  off-centre probes experience different boundary truncation.

No blocking algorithmic issue was found in this review. Minor defensive
improvements suggested to the main agent: explicitly reject non-finite gradient
energy, assert statistics hash/channel count/input shape in ERF, create its output
directory after prerequisite guards, and include per-position scalar summaries.
None of these suggestions licenses changing checkpoints or selecting probes from
test performance. ERF is local sensitivity for recorded weights/inputs, not a
proof of use of all theoretically reachable context.

## Source hashes at this review

- `research/fourcastnet/scripts/cnn/model.py`:
  `36faeb8702c8670c462d2c1d5825de44a8216d592e44a8503245bd4f67253ca3`
- `revision_2026/scripts/rf_ablation.py`:
  `cdb08120ee3ea2383858e358d2736dbc9727d71cbb433becd0eeb3bd63bc7bc7`
- `revision_2026/scripts/rf_erf.py`:
  `4a98a20d93ce52017536715131157275643541e4c385453ebda0d5569d25fad0`
- `revision_2026/scripts/rf_theory.py`:
  `6de8b6f531c394dd2b7e8a9f1f2cff2d3231102424e1358dfb08ac4a7e0fe95a`

Later defensive edits require updated execution hashes; these hashes identify
the exact source version reviewed, not all future revisions.

## Export and paired-model uncertainty audit

Subsequently reviewed `rf_postprocess.py` and `rf_paired_uncertainty.py` without
GPU execution or remote training-status access. Their source hashes at review:

- `rf_postprocess.py`:
  `a16ff5b93e3b48f82f81ce714ad5af425c7437d6eab91f416511c155fea4ce8d`
- `rf_paired_uncertainty.py`:
  `f1a759796c1e0a8cdd2ce852529c457fc3ddc740d73812e017381e1bbf1479ef`

The historical affine CSV fields are exactly `channel,group,before_nrmse,
after_nrmse,G_pct`; the new loader reads the correct keys, converts numerical
fields and reconstructs its20 channel dictionary. Executed local recomputation
of all four reported groups from that CSV agrees with the historical summary to
at most1.11e-14. All20 names are unique, and recomputed per-channel G values
agree. The historical affine checkpoint SHA is asserted before its new frozen
2022 evaluation; no retraining/reselection appears in the export script.

The paired-model bootstrap first requires exact equality of Original/Dilated
baseline per-time SSE, valid counts and channel order. It recomputes RMSE and
gain for both models from the same resampled timestamp sequence, then differences
the resulting gains. Near-minus-upper is differenced within each draw. Percentile
intervals on these paired differences are mathematically appropriate conditional
temporal intervals, not subtraction of separate marginal interval endpoints.
Units are percentage points. The published near/upper ratios are point estimates
from the comparison CSV; this script does not produce ratio confidence intervals.

Executed local numerical checks used the actual nested `statistic` function
extracted from its AST, existing2018 sufficient statistics, and synthetic new
arrays, without importing Torch or using GPU:

1. Identical Original/New arrays give exactly zero differences for both the
   full-year point estimate and a seeded7-day resampled index sequence.
2. Multiplying New near-channel SSE by0.81 produces a10% reduction in those
   channels' RMSE. The function returns the analytically expected near-gain
   difference6.7821667212 percentage points, upper difference0, and the same
   near-minus-upper difference; direct19 difference is5.2612712302 points.
3. The historical training-time parser reads187 original epochs with3394.3s
   summed rounded train-plus-validation times. It is labeled historical, not
   contemporaneous retraining cost.

No blocking error was found. Source review and synthetic tests do not substitute
for validating the forthcoming actual annual/ERF outputs.

## Required completed-artifact acceptance checklist

The following must be inspected after the main run finishes; this checklist is
not a claim that they already exist or pass:

- `structural_validation.json`: actual count1,646,548 for both, exact intended
  metadata differences, identical initialization and B=1/B=8 shapes.
- `provenance.json`, `training_data.json`, training log: CRA5 cache identity,
  unchanged original/stat hashes, recipe/environment/source hashes, completed
  status and timestamps.
- `history.json`, `selection.json`, `best.pt`: selected epoch minimizes recorded
  2017 criterion; patience/max-epoch logic; checkpoint hash agrees; selection
  persisted before test evaluation. No test-based reselection.
- `eval2018/` and `eval2022/`:1460 frames each, summary/per-channel CSV and
  sufficient statistics; recompute all channels/groups from SSE/count arrays.
  Original and Dilated before-SSE/count/channel order must match exactly.
- `eval2022/verified_data_manifest.json`: complete aligned cache with correct
  original statistics and input/target/timestamp hashes.
- `inference_cost.json`: same GPU/inputbatch8/dtype/TF32/determinism and scopes,
  positive finite timings, model sizes and profiler counts; training cost in
  selection kept separate from historical Original epoch times.
- `performance_complete.json`: selected checkpoint hash, both complete years;
  no early marker that could mislabel partial evaluation complete.
- `erf/summary.json`, `gradient_probes.csv`, gradient-energy map NPZ:480 probes
  per model; finite energies/radii; zero probes accounted for; nonzero maps sum
  to1; own-support leakage negligible; Original outside-Original support0;
  Dilated outside-Original mass computed and not assumed positive.
- Theory JSON/all-coordinate CSV: executed exact support propagation with input
  dimensions and boundary/phase caveats, matching the architecture reviewed.
- `comparison.csv`/JSON:6 model-year rows, correct parameter counts and group
  numbers; `per_channel_comparison.csv`:120 model/year/channel rows, no omitted
  negative gains. Historical affine2018 and frozen affine2022 clearly labeled
  low-capacity, not parameter-matched.
- `paired_model_differences.csv`/JSON:8 year/metric rows; point differences match
  summaries, finite ordered intervals,1000 paired28-frame draws with seed20261026.
  Any standalone per-model bootstrap outputs should likewise match annual SSE.
- Gain/ERF figures: values and legends match numerical sources, no clipping or
  selective negative-result omission. ERF color scale/units must be explicit.
- `analysis_complete.json`: scopes bootstrap separately; does not itself certify
  all uncertainty outputs before those outputs exist.

The audit can be extended when the actual outputs are available. Until then,
Dilated performance, measured ERF and computation outcomes remain uncertified.

## Independent acceptance verifier implemented

Added `scripts/verify_rf_results.py`, NumPy/stdlib only, invoked with `--run`
and optional `--original-results`. It never imports Torch, executes inference,
or reads large training/test cache arrays. It writes `acceptance.json` only
after all required artifacts and checks pass, and does not overwrite a previous
acceptance. Missing artifacts produce an explicit failed result without an
acceptance file.

The verifier recomputes all model/year/channel/group metrics from saved temporal
SSE/counts, checks six comparison rows and120 per-channel rows, history selection
and strict patience12/max200 behavior, checkpoint hashes, structural guards,
2022 alignment metadata,960 ERF probe records, mean-map unit mass/structural
leakage and row/map/summary consistency, per-position summaries, same-run cost
arithmetic, and all1000 paired-bootstrap draws for both years. Negative gains
and differences are preserved; improvement is not an acceptance requirement.
Individual raw gradient maps were not saved, so per-probe radius values can be
checked internally but not independently regenerated from only averaged maps.

Executed verifier component tests:

- Original2018 and2022 SSE reproduce stored summaries including direct19 gains
  25.7522891767% and26.3975763523%.
- Historical affine SSE reproduce its CSV and groups, preserving negative upper
  mean-of-channel gain.
- Independent interval-based RF propagation agrees at every coordinate with
  set-based propagation for97,241,513 axes and both dilation configurations.
- Synthetic after-SSE at1.21 times baseline yields negative10% gain in every
  group, confirming no clipping of negative outcomes.
- Synthetic selection history with best epoch2, stopping epoch14 passes;
  intentionally changing the recorded best epoch to1 is rejected.
- Running against the incomplete preflight directory reports missing outputs;
  no `acceptance.json` is created.

Verifier initial source SHA256:
`b1c656c17678c4f61d85313d6ddb7b82147e5b935ba47c3977e5350ba471abda`.
Actual completed-run acceptance remains pending.

## Completed-run independent acceptance — 2026-10-29

This section supersedes the earlier pending-result status, while preserving the
chronological source-review record above. Actual completed outputs were read from
`research/fourcastnet/results/revision_2026/rf_ablation_d23_seed0_20261029/`.
The already-created remote `acceptance.json` was not modified or overwritten.

The auditor called all NumPy/stdlib verifier functions in a fresh local process,
recomputed the two Original/Dilated annual SSE summaries and channel CSVs,
historical affine2018 and frozen affine2022, strict structural guards, history
minimum/early stopping, selected-checkpoint hash, six-row/120-channel exports,
960 ERF probes/maps/group-and-position summaries, costs,2022 manifests and all
1000 paired bootstrap draws per year. All checks passed. All42 artifact hashes
listed by the remote acceptance match their downloaded local copies; its own
hash was unchanged before/after the audit. Both standalone Dilated bootstrap
interval files were separately regenerated from SSE and also pass.

- Selected Dilated epoch125; stopped137 after12 stale epochs.
- Checkpoint SHA256:
  `e63b764fde93a504d7d5a80863883a91a231542be38d2b173e3bc7ab8133de82`.
- Each annual test has1460 complete frames; both models have1,646,548 parameters.
- All480 probes/model have nonzero gradient;960 total records pass.
- All120 per-channel records are present, including9 negative model/year/channel
  gains; no positive-result requirement or clipping is imposed.
- Remote original checkpoint/statistics/source hashes match preserved guards;
  independent local verification checks their recorded proofs and copied source
  AST, not a new local GPU execution or a rehash of large remote raw arrays.

Scientific interpretation from actual data:

| Year | Original G19 | Dilated G19 | Delta Near G(pp) | Delta Upper G(pp) | Dilated Near/Upper ratio |
|---|---:|---:|---:|---:|---:|
| 2018 | 25.7523% | 25.3769% | -0.5698 | +0.2971 | 8.2753 |
| 2022 | 26.3976% | 26.0093% | -0.5834 | +0.3072 | 8.2363 |

Overall performance is slightly worse; aggregate upper gain improves modestly
while near gain decreases, so the pronounced near/upper asymmetry persists.
The paired intervals for these changes exclude zero conditional on the fixed
checkpoints/years, but do not quantify seed/optimization uncertainty. Dilated
z500 worsens by4.4561%/5.1152%; upper mean-of-channel gain also decreases,
despite improved upper aggregate G. Uniform upper-air improvement is unsupported.

Actual ERF does not simply expand with theoretical RF. Mean all20 R95 across
three positions decreases6.4833 to6.1104 pixels; near7.1314 to6.7788 and
upper5.1597 to4.6042. Central all20 R95 decreases5.0875 to3.9563, whereas
position2 increases7.9500 to8.3313. Mean Dilated normalized squared-gradient
mass outside Original theoretical support is3.2675e-5 (0.0032675%), maximum
1.4768e-4 (0.014768%). This is weak distant tail sensitivity, not evidence of
uniformly broader bulk ERF or use of all theoretically available context.

Same-run Original/Dilated GPU-resident batch8 amortized latency is1.3161443 /
1.3147800 ms/frame, with equal19.862221824 GFLOPs profiler counts. The tiny
timing difference should not be promoted to a speedup. Dilated train/validation
loop time is2277.5200s for137 epochs; Original3394.3s/187epochs is historical,
rounded log evidence and is not a controlled training-speed comparison.

Complete requested metrics, per-channel gains, costs, RF results and measured
limitations are collected in `RF_ABLATION_REPORT.md`. The result supports only
the stated fixed bottleneck-dilation ablation. It does not eliminate all
architecture/context explanations or establish a terrain/codec causal mechanism.
