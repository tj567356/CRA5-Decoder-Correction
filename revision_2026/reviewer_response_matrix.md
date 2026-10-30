# Reviewer response matrix — final technical disposition

Updated 2026-10-31. Reviewer labels/issues follow the supplied screenshots and specification. Exact OpenReview wording and accepted-record title still require author verification. Evidence paths are repository-relative.

| Reviewer | Issue | Executed response / evidence | Status |
|---|---|---|---|
| dpEH | Error magnitude versus recoverable fraction | Actual before/after roughness curve, distinct gain interpretation; paper/sec7_terrain.tex and Figure 2 | Complete |
| dpEH | Local terrain complexity | Actual ERA5 orography/LSM; 5×5 land-only SD; western-land and fixed-elevation controls; terrain_roughness_*.csv in research/fourcastnet/results/revision_2026 | Complete; descriptive association |
| UG2Q | 2018 overlaps codec training | Complete 1460-frame 2022 evaluation, original frozen U-Net/statistics, native RH, exact timestamps; revision_2026/2022_acceptance_verification.json | Complete; limited to one unseen year |
| UG2Q | Architecture/receptive field | Executed fixed parameter-matched bottleneck dilation2,3 on CRA5 residuals, full2018/2022, exact structuralRF and480gradientprobes/model; low-capacity affine restored, explicitly unmatched. See RF_ABLATION_REPORT.md and paper/sec6_where.tex. | Experiments complete; theoreticalRF expands but bulkERF does not; broader architectural explanation remains open |
| UG2Q | Compression-rate comparability | Measured coded rates; matched/nonmatched status; global 268-channel versus regional 20-channel scope; paper/sec4_encoder.tex | Complete |
| UG2Q | Missing group baselines | Actual fixed-quantizer frozen replay for five systems; research/fourcastnet/results/revision_2026/baselines/revised_table1.csv | Complete; appears as Table 2 after annual table insertion |
| UG2Q | MPCA-only provenance/retraining | Original stage-one-only checkpoint and fixed step; exact old rate/error reproduced; baselines/PROVENANCE.md in result root | Complete; no original baseline retraining |
| UG2Q | True terrain and mask | Actual surface geopotential/9.80665 and lsm>0.5; negative proxy-elevation sensitivity preserved; revision_2026/static/provenance.json | Complete |
| UG2Q | Missing confidence intervals | Paired circular 7-day block bootstrap, 1000 draws, both full years; bootstrap2018/ and bootstrap2022/ in result root | Complete; conditional uncertainty qualified |
| UG2Q | Title consistency | Supplied title retained in paper/main.tex | Author must compare accepted OpenReview record |
| UG2Q | Figure labels | Vector figures regenerated; all 11 final pages individually rendered and inspected; revision_2026/figure_visual_qa_report.md | Complete |
| UG2Q | Captions/units/consistency | Explicit direct/derived channels, gain versus compression ratios, channel-mean versus pooled metrics, bounded claims; paper/sec3_setup.tex through sec8_limitations.tex | Complete; independent audits retained |

Original U-Net and original results were preserved. No partial-year result is labeled full-year. Final external submission, publication choice, signatures and payment remain author actions. See revision_2026/FINAL_REPORT.md for exact source files and limitations.

Latest complete layout/coverage review: LAYOUT_REVIEW_20261031.md. Table 3 now precedes the Limitations heading; all table data rows are unchanged. QA: qa/layout_review_20261031/. Internal title/metadata consistency is checked; accepted OpenReview title equality remains unverified.
