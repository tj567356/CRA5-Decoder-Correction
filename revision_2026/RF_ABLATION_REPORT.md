# Executed parameter-matched receptive-field ablation

Completed and independently checked on2026-10-29. This is one prespecified
bottleneck-dilation intervention, not a model-architecture search. All numerical
values below come from completed artifacts, not expected outcomes.

Canonical result directory (relative to this report):
`../research/fourcastnet/results/revision_2026/rf_ablation_d23_seed0_20261029/`.
The same relative result directory exists on Ubuntu below
`/home/tong/fourcastnet_us_mpca/results/revision_2026/`.

## Intervention and fixed controls

Original UNet2 is preserved. DilatedUNet2 inherits its constructor and forward,
changing only the two bottleneck3x3 convolution dilation/padding metadata:
first(2,2), second(3,3), versus original(1,1) for both. Encoder, decoder, skips,
widths, kernels, activations, input/output shapes and state-dict tensor shapes
remain unchanged. Identical seeded initial weight tensors are verified.

| Property | Original | Dilated |
|---|---|---|
| Parameters | 1,646,548 | 1,646,548 |
| Input/output shape | Bx20x97x241 | Bx20x97x241 |
| Shape preflight | B=1 and8 passed | B=1 and8 passed |
| Bottleneck dilation rates | 1,1 | 2,3 |
| Training source | CRA5 residual cache | Same CRA5 residual cache |
| Training/validation | 2014-2016 /2017 | Same |
| Test years | Complete2018 and2022 | Same, frozen after selection |

Recipe: AdamW lr3e-4,weight decay1e-4,gradient clipping1,batch8,AMP fp16 with
GradScaler,seed0,four workers,max200 epochs,patience12. Training shuffle and
frozen2014-2016 normalization follow the original recipe. Checkpoint selection
uses2017 mean-batch normalized MSE, including the partial final batch. Neither
test year participates in fitting or selection. The2022 original-cache alignment
manifest/input/target/timestamp hashes were verified before evaluation.

## Frozen performance: both full years

Each year has1460 six-hourly frames. Primary19-channel means exclude derived
tcwv. Shared baseline nRMSE is0.0866141535 in2018 and0.0862452422 in2022.
Near means surface/1000/850hPa selected channels(13); upper means500/50hPa(6).
G is relative reduction in channel-mean nRMSE. Difference units are percentage
points, not percent relative change.

| Model | Year | Parameters | After19 nRMSE | G19(%) | Near G(%) | Upper G(%) | Near/upper | Near-minus-upper(pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original U-Net | 2018 | 1,646,548 | 0.06430903 | 25.7523 | 32.1783 | 3.5225 | 9.1350 | 28.6558 |
| Original U-Net | 2022 | 1,646,548 | 0.06347859 | 26.3976 | 32.7932 | 3.6035 | 9.1005 | 29.1898 |
| Dilated U-Net | 2018 | 1,646,548 | 0.06463417 | 25.3769 | 31.6085 | 3.8196 | 8.2753 | 27.7889 |
| Dilated U-Net | 2022 | 1,646,548 | 0.06381342 | 26.0093 | 32.2099 | 3.9107 | 8.2363 | 28.2992 |
| Affine low-capacity reference | 2018 | 3,620 | 0.08549337 | 1.2940 | 1.4319 | 0.8171 | 1.7524 | 0.6148 |
| Affine low-capacity reference | 2022 | 3,620 | 0.08511215 | 1.3138 | 1.4452 | 0.8454 | 1.7095 | 0.5998 |

Sources: `comparison.csv`, `comparison.json`, annual `summary.json` and
`sufficient_statistics.npz` under the canonical directory. Affine2018 reuses
the earlier completed experiment; affine2022 is a new frozen test of the same
historical selected affine checkpoint, without retraining or reselection.
The affine reference is explicitly **not parameter-matched**.

The Dilated model slightly worsens overall19-channel gain in both years while
slightly increasing aggregate upper gain and reducing near gain. The near/upper
gain ratio remains about8.2. Thus this fixed matched-parameter intervention does
not remove the near/upper asymmetry. It does not establish that receptive-field
limitations generally cannot matter, or that the codec alone causes the gap.

## Conditional paired-model uncertainty

Both models share the sampled timestamp sequence. Circular28-frame(7-day)
blocks,1000 draws,seed20261026; RMSE and gains recomputed for each draw. The
following are **Dilated minus Original**, with percentile95% temporal intervals.

| Year | Metric | Difference(pp) | 95% interval(pp) |
|---|---|---:|---|
| 2018 | G19 | -0.3754 | [-0.4022,-0.3530] |
| 2018 | Near G | -0.5698 | [-0.6040,-0.5397] |
| 2018 | Upper G | +0.2971 | [+0.2670,+0.3257] |
| 2018 | Near-minus-upper | -0.8669 | [-0.9189,-0.8165] |
| 2022 | G19 | -0.3882 | [-0.4166,-0.3602] |
| 2022 | Near G | -0.5834 | [-0.6212,-0.5468] |
| 2022 | Upper G | +0.3072 | [+0.2757,+0.3399] |
| 2022 | Near-minus-upper | -0.8906 | [-0.9482,-0.8349] |

Sources: `paired_model_differences.csv` and JSON. Individual Dilated G19
intervals are[24.9575,25.7914]% in2018 and[25.5223,26.5175]% in2022; see
`bootstrap2018/confidence_intervals.csv` and the corresponding2022 file.
The independent auditor regenerated both standalone and paired intervals.
These intervals condition on the two fitted checkpoints, each year and grid;
they do not measure training-seed or spatial-sampling uncertainty.

## Per-channel gains: negative findings retained

The complete120 model/year/channel records, including baseline and corrected
nRMSE, are in `per_channel_comparison.csv`. Annual Dilated CSV files are also
available under `eval2018/` and `eval2022/`.

| Channel | Original2018 G(%) | Dilated2018 G(%) | Original2022 G(%) | Dilated2022 G(%) |
|---|---:|---:|---:|---:|
| u10 | 36.4505 | 36.1671 | 37.0199 | 36.7008 |
| v10 | 34.0227 | 33.5587 | 34.7247 | 34.1774 |
| t2m | 31.3694 | 30.9054 | 31.8775 | 31.4827 |
| sp | 74.8678 | 73.9496 | 74.8650 | 73.9691 |
| msl | 14.0643 | 12.2466 | 14.3056 | 12.2729 |
| t850 | 12.2977 | 11.5039 | 12.9969 | 12.2290 |
| u1000 | 40.5156 | 40.2424 | 41.2273 | 40.9781 |
| v1000 | 40.6568 | 39.9515 | 41.4977 | 40.7876 |
| z1000 | 15.4662 | 13.6255 | 15.6541 | 13.6343 |
| u850 | 20.1316 | 19.5296 | 20.9882 | 20.3318 |
| v850 | 19.5985 | 18.9699 | 20.0626 | 19.4575 |
| z850 | 3.3645 | 3.4897 | 3.5037 | 3.5856 |
| u500 | 0.0491 | 0.0919 | 0.0463 | 0.0993 |
| v500 | 0.0011 | 1.4326 | -0.0082 | 1.4630 |
| z500 | 0.4349 | **-4.4561** | 0.6788 | **-5.1152** |
| t500 | 0.1028 | 0.0619 | 0.1092 | 0.0447 |
| z50 | 0.9107 | 0.9835 | 1.5025 | 1.8595 |
| r500 | 7.4024 | 7.5531 | 7.5373 | 7.7086 |
| r850 | 15.1482 | 14.6340 | 15.9262 | 15.4286 |
| tcwv(derived; excluded from19) | 59.6056 | 59.1248 | 59.4287 | 58.9743 |

Upper aggregate G increases, but equal-channel fractional gain Gbar decreases
from1.4835% to0.9445% in2018 and1.6443% to1.0100% in2022. This reflects
redistribution across channels, including worse z500, not uniform upper-air
repair. The complete export contains9 negative channel/model/year rows across
all three models; no negative values were clipped or omitted.

## Structural versus effective receptive field

Exact discrete dependency propagation includes all down/up convolutions,
floor pooling, odd-size bilinear `align_corners=False` resize, skip unions and
kernel3 head. Padding is constant zero, not extra observed context.

| Structural quantity | Original | Dilated |
|---|---:|---:|
| Central output(48,120), real-input support | 72x72 | 96x96 |
| Central input row bounds(inclusive) | 12..83 | 0..95 |
| Central input column bounds(inclusive) | 84..155 | 72..167 |
| Unclipped maximum per-axis span | 76 | 100 |
| Actual97x241 maximum axis spans | 76x76 | 97x100 |

The theoretical span is phase-dependent, and the Dilated latitude span is
limited by the97-row domain. These are grid pixels, not kilometres.

Gradient ERF was then actually measured on eight fixed2017 validation frames
`[0,208,416,625,833,1042,1250,1459]`, all20 individual output channels and three
positions`[(48,120),(24,60),(72,180)]`:480 scalar probes/model,960 total, with
no zero-gradient probes. It differentiates normalized correction f(x), not the
identity-plus-correction output; squared gradients are summed across input
channels and normalized per probe before map averaging.

The following are mean per-probe Chebyshev energy **radii**, not diameters and
not radii of a single averaged map:

| Probe scope | Original R90 | Dilated R90 | Original R95 | Dilated R95 |
|---|---:|---:|---:|---:|
| All20, all three positions | 4.3271 | 3.8313 | 6.4833 | 6.1104 |
| Near13, all positions | 4.7115 | 4.1506 | 7.1314 | 6.7788 |
| Upper6, all positions | 3.5972 | 3.1250 | 5.1597 | 4.6042 |
| All20, central position only | 3.4063 | 2.7813 | 5.0875 | 3.9563 |

Despite broader theoretical support, bulk gradient ERF does **not broaden on
average**. Spatial behavior is heterogeneous: at position2, all20 mean R95
increases7.9500 to8.3313, while the other two positions decrease. Dilated
sensitivities extend beyond Original structural support, but only in a tiny
tail: all20 mean squared-gradient mass outside the corresponding Original
support is3.2675e-5, or **0.0032675%**; maximum across probes is1.4768e-4,
or **0.014768%**. Near/upper mean tail masses are3.8446e-5 and1.8940e-5.
Original outside-Original mass is zero; both models' own-support leakage passes.

Therefore do not describe this as proof that the learned model uses the entire
enlarged context. It is evidence of expanded potential support and weak distant
gradient sensitivity under the sampled weights/inputs. Gradients are local
sensitivities, not a causal contribution decomposition. Sources:
`erf/summary.json`, `gradient_probes.csv`, normalized map NPZ and plotted PDF/PNG.

## Training and inference computation

Dilated selected epoch125 by2017 validation MSE0.005744843522361556;
early stopping at137. Training/validation loop wall time was2277.5200s
(37.96min), including loop-side checkpoint/history writes,75076 training steps,
25071 validation steps, and peak allocated CUDA memory556295680 bytes.

Historical Original logs contain187 epochs and3394.3s summed rounded
train-plus-validation times. That is historical evidence, not a contemporaneous
Original retraining benchmark; different stopping epochs and measurement
conditions preclude claiming a training speedup from this comparison.
Historical source/hash: preflight `local_independent_checks.json` and
`revision_2026/cra5_train.log`, SHA256
`105d036ea18832acbaa939bfc7ca5321efc87eb0ebee30dad9873a4d44279dc4`.

| Same-run inference measurement | Original | Dilated |
|---|---:|---:|
| Float32 weight bytes | 6,586,192 | 6,586,192 |
| Profiler-counted FLOPs/frame | 19,862,221,824 | 19,862,221,824 |
| Mean batch8 latency(ms) | 10.5291541 | 10.5182397 |
| Amortized ms/frame | **1.3161443** | **1.3147800** |
| Throughput(frames/s) | 759.7951 | 760.5835 |

RTX3080,PyTorch2.11.0+cu128,float32,TF32 allowed,cuDNN deterministic=True,
benchmark=False;10 warmup/100 timed batches on identical GPU-resident2017
inputs. Timing excludes CRA5 decoding,I/O,transfers and evaluation aggregation.
The approximately0.0014ms/frame difference is not a demonstrated speed advantage
without repeated timing uncertainty. Profiler counts recognized operations,
primarily convolutions, rather than complete hardware operations. Allocated
inference peak417355776 bytes includes other persistent tensors and is not an
isolated-model requirement. Source: `inference_cost.json`.

## Provenance and independent acceptance

- Original checkpoint SHA256 remains
  `44c57a0f2a18e92c9c0745cfe75337079627ab8729e93c6ec098a5ddeb007435`.
- Frozen training-statistics SHA256 remains
  `272cc45d231ec349025c4fa1e9f69366ae874739c98299ce666921821f13dba2`.
- Selected Dilated checkpoint SHA256:
  `e63b764fde93a504d7d5a80863883a91a231542be38d2b173e3bc7ab8133de82`.
- Historical affine checkpoint SHA256 remains
  `31743a1a03669a6b20f7a39c79c7ae3738fa059468ea789a56e6ab7588c64f5d`.
- Executed model-source SHA256:
  `36faeb8702c8670c462d2c1d5825de44a8216d592e44a8503245bd4f67253ca3`.
- Executed training-source SHA256:
  `cdb08120ee3ea2383858e358d2736dbc9727d71cbb433becd0eeb3bd63bc7bc7`.
- Original remote model-source SHA256 remains
  `19f4f5d91e21afa5c68cea7f66b5fbb998e11a21b836fce2e7d22857537b0c52`.
- ERF execution-source SHA256:
  `3ff45a1088859ef67e39e7ee743713c868593a4a7b0f4a58f8bcf0df219a0195`.

Executed code was frozen on Ubuntu under
`revision_2026/rf_ablation_20261029/`. Reproducible entry commands from the
project root, using a fresh output directory for any new run:

```sh
.venv/bin/python -u revision_2026/rf_ablation_20261029/rf_ablation.py --model revision_2026/rf_ablation_20261029/model.py --out results/revision_2026/rf_ablation_d23_seed0_20261029
bash revision_2026/rf_ablation_20261029/run_rf_post.sh
.venv/bin/python revision_2026/rf_ablation_20261029/rf_paired_uncertainty.py --run results/revision_2026/rf_ablation_d23_seed0_20261029
.venv/bin/python revision_2026/rf_ablation_20261029/verify_rf_results.py --run results/revision_2026/rf_ablation_d23_seed0_20261029
```

The executed wrapper serializes ERF after frozen tests/costs complete, then runs
exports and standalone temporal bootstrap; paired-model bootstrap was executed
separately. The current repository wrapper also includes that final paired step.
For a new run, use a fresh output path and update the wrapper run variable rather
than executing these commands against the completed directory. Existing output
directories are exclusive; these commands are documentation, not permission to
overwrite existing artifacts. `train.log` and `post.log` retain actual execution
output; `provenance.json`, `selection.json` and `training_data.json` retain hashes,
timestamps and recipe. No Original checkpoint or prior result was overwritten.

Remote `acceptance.json` reports accepted. A separate auditor subsequently
re-ran all verifier functions locally without Torch/GPU or overwriting that
file:42 required artifact hashes match, both annual SSE/metrics and all exports
recompute,960 ERF rows/maps/summaries pass, paired1000-draw intervals recompute,
and the selected epoch/patience/hash checks pass. The remote acceptance file is
unchanged. Both standalone Dilated intervals were also independently regenerated.
See `RF_ABLATION_INDEPENDENT_AUDIT.md` for verification details and limits.

## Scientific interpretation and remaining limits

This parameter-matched, fixed-dilation experiment modestly redistributes error:
aggregate upper gain rises about0.30pp while near gain falls about0.58pp,
overall19-channel performance is slightly worse, and the large near/upper gap
persists. Theoretical support broadens; bulk gradient ERF does not broaden on
average and only very small sensitivities extend beyond Original support.

This supports a narrow statement about this intervention under this recipe. It
does not prove that broader receptive fields cannot help, that the learned model
used all available context, that nonlinearity alone explains the U-Net advantage,
or that terrain/codec structure causally creates the near/upper gap. One seed,
one region/codec, limited ERF input probes, differing selected epochs, finite
domain boundaries and conditional temporal bootstrap all constrain interpretation.
