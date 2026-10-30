# TAAI 2026 camera-ready — final technical report

Updated 2026-10-29: parameter-matched bottleneck RF ablation completed and independently accepted; low-capacity affine control restored. Branch `codex/camera-ready-2026`. Author record verification and external submission remain pending.

## Delivery

- `final/TAAI2026_camera_ready.pdf`: final 11-page English Springer LNCS A4 paper.
- `final/TAAI2026_source.zip`: 16 actual source/dependency files, build instructions and manifest.
- `final/TAAI2026_source.manifest.json`: source and ZIP hashes.
- `figure_visual_qa_report.md`, `qa/rf_ablation_20261029/structural_qa.json`, `qa/rf_ablation_20261029/source_zip_verification.json`, `qa/rf_ablation_20261029/main.log` and the independent rebuild files in that QA directory: acceptance evidence.

## 1. Experiments actually executed

Audited original 2014–2018 arrays, coordinates, complete calendars, finite values and sampled normalization arithmetic. Re-evaluated all 1460 frames of 2018 with the original frozen U-Net and saved time/channel/spatial sufficient statistics. Executed actual-orography terrain curves, western-land, near/upper and fixed-elevation controls. Preserved the original negative proxy-elevation result.

Executed paired circular 7-day moving-block bootstrap (1000 draws) for complete 2018 and 2022. Replayed fixed original quantizers and separate original MPCA-only/e995/e99/e97 checkpoints for unrounded group metrics, without baseline retraining. Downloaded, validated, decoded, aligned and evaluated all 1460 2022 frames using unchanged original codec, U-Net and statistics.

After 2022 completion, trained the optional affine 3×3 corrector on 2014–2016, selected on 2017 and tested once on 2018. Selected epoch 125, stopped at 137. Its weak and negative results are restored in the current manuscript as a low-capacity, non-parameter-matched reference; its fixed checkpoint also completed a new frozen 2022 test. Profiled U-Net size/inference cost and performed independent evidence/manuscript audits. Scripts and logs are in `scripts/`, `logs/` and `../research/fourcastnet/scripts/analysis/`.

## 2. Data and years

Training: 2014–2016; validation: 2017; original corrector test: full 2018. New out-of-time test: full 2022, 1460 six-hourly UTC frames from January 1 00:00 through December 31 18:00, no gaps/duplicates. Grid: 97 × 241, 0.25 degrees, descending latitude 49 to 25 and longitude −130 to −70. Main metrics use 19 direct channels; all-20 metrics additionally include derived tcwv.

2022 CRA5 publisher LFS hashes were checked at pinned dataset revision `0e83ac8c9973815078136dd54b1b59a80f0cdde1`. ERA5 18 fields came from pinned ARCO blocks; native RH at 500/850 hPa came from NCAR GDEX OPeNDAP with daily receipts and timestamp/grid/unit checks. Native RH was not replaced with a derivation. Original float32 normalization arithmetic followed by float16 cache storage was retained. Source-2018 validation and partial/full block decode checks were executed.

Evidence: `data_manifest.json`, `2022_DATA_PREPARATION.md`, `2022_*manifest.json`, `2022_source2018_validation.json`, `2022_native_rh_validation.json`, `2022_acceptance_verification.json`, `provenance_2022.tar.gz`. Large arrays remain on Ubuntu under `/home/tong/fourcastnet_us_mpca/revision_2026/data`, outside the source ZIP.

## 3. Checkpoints and frozen protocol

Original U-Net: `/home/tong/fourcastnet_us_mpca/checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/best.pt`, SHA256 `44c57a0f2a18e92c9c0745cfe75337079627ab8729e93c6ec098a5ddeb007435`, 1,646,548 parameters. Statistics SHA256 `272cc45d231ec349025c4fa1e9f69366ae874739c98299ce666921821f13dba2`.

Official codec: `/home/tong/.cache/torch/hub/checkpoints/cra5_268v_300k.pth`, SHA256 `812a6d09ff039e4ab468915942e7fd00f11e7c58b839cd1b60fe1d0586a40dad`. Published full-resolution training window: 1979–2021. An independent upstream training log is not bundled with the state dictionary; the manuscript preserves this qualification.

MPCA-only: `/home/tong/new_20/checkpoints/s99_stage1only/unet_d2_c64_lr3e-4_clip1/best.pt`, SHA256 `aee68485d07018e0c9acbb1dd2cb7585ed3895f945d88e6232b516ba42876a76`. Separate original e995/e99/e97 weights and steps are documented in the baseline provenance files.

New optional affine checkpoint: `linear_corrector/best.pt`, SHA256 `31743a1a03669a6b20f7a39c79c7ae3738fa059468ea789a56e6ab7588c64f5d`. It did not modify the original U-Net. No 2022 training, selection, fine-tuning or normalization fitting was performed. See `checkpoint_manifest.json` and `linear_corrector/provenance.json`.

## 4. Main results and source files

G is percentage reduction in channel-mean nRMSE. Intervals are conditional temporal block-bootstrap intervals.

| Year / group | Before | After | G (%) | 95% interval |
|---|---:|---:|---:|---|
| 2018 / 19 direct | 0.08661 | 0.06431 | 25.75 | 25.32–26.18 |
| 2022 / 19 direct | 0.08625 | 0.06348 | 26.40 | 25.90–26.93 |
| 2018 / near-surface | 0.09820 | 0.06660 | 32.18 | 31.73–32.62 |
| 2022 / near-surface | 0.09843 | 0.06615 | 32.79 | 32.28–33.31 |
| 2018 / upper-air | 0.06151 | 0.05934 | 3.52 | 3.38–3.68 |
| 2022 / upper-air | 0.05984 | 0.05768 | 3.60 | 3.46–3.76 |

Canonical result root: `../research/fourcastnet/results/revision_2026/`. Sources: `eval2018/`, `eval2022/`, `bootstrap2018/`, `bootstrap2022/`, `2022_full_summary.json`, `2022_full_per_channel.csv`, `2022_full_timestamps.csv`, `2018_vs_2022_fullyear.csv` and PDFs. Completion marker: `2022_pipeline_complete.json`. Individual v500 slightly worsens in 2022 (G −0.0082%); aggregate gain is not universal.

Terrain all-land endpoint before errors: 0.08170 → 0.11482; corrected errors: 0.06479 → 0.07364; gains: 20.7% → 35.9%. Western-land endpoint gains: 18.1% → 36.9%. The all-land low-roughness segment is nonmonotonic. Geography and elevation remain confounders; no causal conclusion is drawn. Sources: `terrain_roughness_all_land.csv`, `terrain_roughness_west_land.csv`, `terrain_roughness_near_upper.csv`, `terrain_roughness_summary.json` and the main curve PDF in the canonical root.

Classical controls: `baselines/revised_table1.csv` and `baselines/PROVENANCE.md`. MPCA-only's original rate 473.1151× and all-channel errors 0.1272884 → 0.0868782 were reproduced. e995/e99/e97 rates are explicitly nonmatched. CRA5 global and classical regional coding domains differ.

Low-capacity affine control (restored to current manuscript): 19-direct G 1.29%; near/upper G 1.43% / 0.82%; upper-air mean per-channel gain −0.91%. Sources: `linear_corrector/summary.json`, `selection.json`, `provenance.json`, `2018_per_channel.csv`. Capacity/receptive field differ, so the U-Net gap cannot be attributed solely to nonlinearity.

Inference: `corrector_cost.json`. About 6.6 MB float32 weights, 19.86 G profiler-counted FLOPs/frame. RTX3080 GPU-resident batch-8 float32 inference, TF32 allowed: the earlier profile measured 1.49 ms/frame. The new matched run measured 1.316/1.315 ms/frame for Original/Dilated with deterministic=True and TF32 allowed; both exclude decoder, I/O and transfers. Tiny runtime differences are not a speed claim.

## 5. Reviewer responses

`reviewer_response_matrix.md` maps every supplied issue to evidence, edits or explicit limitations: actual roughness, western/elevation controls, training overlap via full 2022, uncertainty, classical group metrics, MPCA provenance, rate/domain accounting, explicit architecture limitations, readable figures and bounded claims. All 13 bibliography entries were audited against primary records; see `bibliography_audit.md`. Exact accepted title and verbatim reviews still need author comparison against OpenReview.

## 6. Incomplete work and limitations

No core experiment or packaging step remains blocked. One fixed parameter-matched bottleneck dilation CRA5 ablation is now complete. Broader effective-context/architecture, multi-year/region/seed robustness remain unresolved; the measured gradient energy did not broadly expand. Terrain causality is not established. Author record checks and submission actions remain pending. Earlier progress logs describing pending 2022 data or intermediate 10-page builds are historical; this report supersedes those statuses.

## 7–10. Acceptance outcomes

- Full 2022: **PASS**, all 1460 aligned frames, frozen original U-Net/statistics.
- Terrain main figure: **PASS**, roughness (m) against before/after nRMSE, all-land and western-land panels, included as Figure 2.
- Final format: **11 pages including references**, LNCS A4, embedded fonts, no overfull boxes/unresolved citations, every page visually inspected.
- PDF/source consistency: **PASS**, independent extraction and rebuild; identical text and all 11 rendered page hashes. PDF byte hashes differ due to build metadata.
- PDF SHA256: `bf0b2746a83a37c54325b707c8d62d92f8de0344a79f8a553a6d91dec607872e`.
- ZIP SHA256: `53f1b95be300e5f2c2cfc4ea9ec2d4fd6c73a5ea898d0acd805b63804013dd5e`.

## 11. Authors' remaining OpenReview actions

Use `submission_checklist.md`. Compare title and reviews with the accepted record; confirm names/order, affiliation, corresponding author/email, disclosure and any funding acknowledgment. Existing title/author order were preserved. Authors must choose Publish versus Presentation Only, verify current official requirements, upload the PDF and required source ZIP, complete signed authorization, select the submission's Authors signature group, register/pay and perform final submission. No external submission, signature, payment or publication choice was performed.


## Executed parameter-matched RF ablation — current delivery

See `RF_ABLATION_REPORT.md`, `RF_ABLATION_INDEPENDENT_AUDIT.md` and `RF_MANUSCRIPT_AUDIT.md` for all requested metrics, every channel, source hashes and bounded interpretation. All new outputs are in `../research/fourcastnet/results/revision_2026/rf_ablation_d23_seed0_20261029/`.

Both models have 1,646,548 parameters and preserve20x97x241 shapes. Only bottleneck dilation/padding changes to2,3. CRA5 residual training2014–2016, validation2017, original AdamW recipe; selected epoch125, stopped137. Original UNet2 AST/checkpoint and remote original model source remain unchanged.

| Model | Year | After direct19 | G19 (%) | Near G (%) | Upper G (%) | Near/upper | Near-minus-upper pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 2018 | 0.06431 | 25.75 | 32.18 | 3.52 | 9.14 | 28.66 |
| Dilated | 2018 | 0.06463 | 25.38 | 31.61 | 3.82 | 8.28 | 27.79 |
| Original | 2022 | 0.06348 | 26.40 | 32.79 | 3.60 | 9.10 | 29.19 |
| Dilated | 2022 | 0.06381 | 26.01 | 32.21 | 3.91 | 8.24 | 28.30 |

Full2022 uses1460 frames and the same selected frozen checkpoint, never2022 fitting. Affine2022 direct19G=1.3138%, explicitly not parameter-matched. Per-channel negative results remain, including Dilated z500G=-4.456%/-5.115% in2018/2022; upper mean-of-channel gains also worsen.

Exact central theoretical support72x72→96x96; unclipped phase maximum76→100 peraxis; finite-domain maximum97x100 for Dilated. Measured gradient ERF uses480probes/model on fixed2017 inputs; central mean90%-energy radius3.41→2.78cells, and all-probe mean energy beyond Original support is only0.00327%. Theory expands, bulk effective context does not. This ablation cannot rule out broader architecture/optimization explanations.

Actual train+validation2277.52s for137epochs; historical Original3394.3s for187epochs is not an equal-epoch runtime comparison. Same-run Original/Dilated inference1.3161/1.3148ms perframe atbatch8, both19.862GFLOPs. Scope and all flags are in inference_cost.json.

Independent acceptance recomputes6model/year rows,120channel rows,960ERFprobes,8paired temporal differences and all intervals. PDF now includes the real ablation and low-capacity affine reference, remains11pages, all fonts embedded and all11pages visually checked. Independent ZIP rebuild has identical text and all11page-render hashes. Historical removed-affine version is archived separately.


## Current table presentation update, 2026-10-29

Original elevation-control Table 2 is restored as current Table 3, with all five data rows unchanged. Current Table 1 reports all-20-channel After and G including derived tcwv, matching Figure 1; direct19 and near13/upper6 gains remain separately labelled. Current Table 2 excludes tcwv from its altitude groups and separates After from G with explicit spacing. No experiment or checkpoint was changed for this presentation update.

Current delivery is 11 pages; all fonts embedded, no overfull boxes or unresolved references. Latest QA: qa/table_restore_20261029/. Independent source ZIP rebuild reproduces normalized text and all 11 page renders exactly. Numerical audit: TABLE_REVISION_AUDIT.md.


## Current layout and reviewer coverage review, 2026-10-31

Table 3 is now before the Limitations heading on page 10, following Section 7, rather than interrupting limitations. All table data rows are unchanged. Setup explicitly distinguishes 20-channel training from direct19 evaluation; column-integrated tcwv is excluded from altitude groups. Figure 1 matched PCA retains 4,384 components, distinct from the Table 2 energy settings. Disclosure stays on page 10; references start on page 11. Repeated wording was condensed without changing claims or font/margin/class settings.

Full reviewer coverage and bounded remaining issues: LAYOUT_REVIEW_20261031.md. Main requested technical experiments/analyses are present; terrain causality, general architecture effects and broader shift robustness remain unresolved mechanisms/scope, not fabricated positive claims. Exact title equality with the accepted OpenReview record is still unverified.

Latest QA: qa/layout_review_20261031/. All 11 A4 pages inspected; all fonts embedded; no overfull or unresolved-reference warnings. Independent ZIP rebuild reproduces text and all 11 page renders. No new training/results/checkpoints or external submission changes.
