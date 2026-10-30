# Independent RF manuscript audit

Date: 2026-10-29 (Asia/Taipei). Scope: read-only review of the current manuscript against completed experiment artifacts. This audit did not modify TeX, retrain a model, execute GPU inference, or regenerate bootstrap/ERF results. It is distinct from the execution acceptance and the independent numerical-verifier audit.

## Verdict

No material numerical mismatch or unsupported RF-mechanism conclusion was found in the reviewed manuscript. The parameter-matched dilation result is correctly presented as one fixed intervention with a small overall deterioration and a persistent near/upper split. The weak affine result is retained and explicitly marked not parameter-matched.

Three minor precision improvements were sent to the root agent; these are editorial refinements rather than failed scientific checks:

1. Abstract/conclusion use approximately 8.2-fold for the dilation split. Exact ratios are 8.2753 in 2018 and 8.2363 in 2022. “About eightfold” or “8.24–8.28-fold” avoids suggesting consistent one-decimal rounding (2018 rounds to 8.3).
2. The 0.0033% outside-original-support energy is a **mean across all 480 probes**, whereas the immediately preceding R90 value is for central probes only. Explicitly naming the averaging scope prevents ambiguity.
3. The paired-model intervals should be labelled 95%. A generic limitations phrase can refer to “the fitted checkpoints” rather than “one checkpoint,” because paired differences condition on both selected models.

## Reviewed evidence

- `paper/main.tex`, `sec3_setup.tex`, `sec5_results.tex`, `sec6_where.tex`, `sec8_limitations.tex`, `sec9_conclusion.tex`.
- `revision_2026/RF_ABLATION_REPORT.md`.
- Canonical executed outputs under `research/fourcastnet/results/revision_2026/rf_ablation_d23_seed0_20261029/`: `comparison.csv`, `paired_model_differences.csv`, both annual `summary.json` files, `erf/summary.json`, `inference_cost.json`, `selection.json`, `provenance.json`, `training_data.json`, `structural_validation.json`, and acceptance metadata.
- Local `research/fourcastnet/scripts/cnn/model.py` and `revision_2026/scripts/rf_ablation.py`; their SHA256 values match executed provenance: model `36faeb8702c8670c462d2c1d5825de44a8216d592e44a8503245bd4f67253ca3`, training `cdb08120ee3ea2383858e358d2736dbc9727d71cbb433becd0eeb3bd63bc7bc7`.

## Claim checks

| Manuscript claim | Evidence and assessment |
|---|---|
| Parameter matching and only two convolutions changed | Source defines `DilatedUNet2(UNet2)` and calls the inherited constructor; it changes only `bottleneck[0]` dilation/padding to 2 and `bottleneck[2]` to 3. There is no subclass forward override. Structural-validation output reports 1,646,548 parameters for each, matching initial tensors/state keys, unchanged Original AST, and B=1/B=8 shapes. This manuscript audit checked source and recorded evidence; it did not rerun Torch shape tests. |
| Frozen protocol | Source persists `selection.json` before opening either 2018 or 2022 cache. Best epoch 125, stop 137, validation MSE 0.0057448435, training time 2277.5200 s match manuscript. Training years are 2014–2016, validation 2017. The 2022 evaluation validates full-year alignment plus input/target/timestamp hashes and original statistics hash. Both years have 1460 frames. |
| Annual performance table | All six rows match `comparison.csv` at displayed precision, including Dilated direct19 after nRMSE 0.06463/0.06381 and gains 25.38/26.01%. Original and Affine rows are preserved. |
| Small redistribution, not a general improvement | Dilated-minus-Original upper G is +0.2971/+0.3072 percentage points; near G is −0.5698/−0.5834 points; direct19 G is −0.3754/−0.3882 points. The manuscript correctly says overall correction is slightly worse. |
| Paired uncertainty | Upper difference intervals round to [0.27,0.33] and [0.28,0.34] percentage points. The manuscript does not treat these conditional temporal intervals as training-seed or spatial uncertainty. |
| Negative findings | Dilated z500 G is −4.4561/−5.1152%, matching −4.46/−5.12%. Upper Gbar is 0.94448/1.00997%, below Original 1.48351/1.64432%; both the worsening channel and lower Gbar remain in text. Thus increased aggregate upper G is not presented as uniform channel improvement. Original v500 slight 2022 worsening also remains in Section 5. |
| Structural versus gradient RF | Central actual support 72×72→96×96, unclipped per-axis maxima 76→100 match `erf/summary.json`. Eight 2017 inputs × 20 scalar outputs × three positions equals 480 probes/model. The diagnostic differentiates correction f(x), sums squared gradients over input channels, and normalises each probe. Central mean R90 3.40625→2.78125 rounds correctly. |
| Bulk ERF and tiny tail | All-position mean R90 also decreases (4.32708→3.83125). Mean outside-original-support mass is 0.00003267505, or 0.003267505%, rounding to 0.0033%. The manuscript explicitly rejects the inference that broader theoretical support means broad effective use. It does not claim every location/probe narrows, nor causal contribution from gradients. |
| Inference costs | Both models have 19,862,221,824 profiler-recognised FLOPs/frame. Same-run GPU-resident times are 1.3161443 and 1.3147800 ms/frame, matching text. No speed advantage or training speedup is claimed. Section 5 states GPU, batch size, TF32 allowance and excluded decoding/I/O/transfer costs. |
| Affine baseline | 3620 parameters, 3×3 biased convolution and no activation are stated. It is explicitly not parameter-matched. Direct19 G 1.2940/1.3138% and negative 2018 upper Gbar −0.90694% match the manuscript. Capacity/context confounding prevents attributing the U-Net gap to nonlinearity alone. |
| Conclusions | Abstract, RF discussion, limitations and conclusion preserve one-seed/dilation-pattern limits, incomplete effective-context broadening, and unresolved broader architecture/context explanations. No general claim that receptive field cannot matter or that the codec causally creates the split was found. |

## Limits of this audit

This is an independent manuscript/source-to-artifact consistency audit. The separate execution and numerical-verifier lanes cover SSE regeneration, bootstrap recomputation, ERF map consistency and artifact hashes. Final compilation, page budget, all-page visual inspection, embedded fonts and source-ZIP consistency remain with the main lane. No claim of PDF visual QA is made by this report.

Executed local standard-library check: parsed the model AST and asserted the subclass inherits UNet2 and defines only `__init__` (no forward override); checked all six manuscript after19/G19 rounded values against CSV; recomputed the percent conversion of all-probe mean tail energy as 0.003267505137996782%. All checks passed without Torch/GPU execution or result-file mutation.
