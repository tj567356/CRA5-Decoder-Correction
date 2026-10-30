# Full manuscript layout and reviewer coverage review

Date: 2026-10-31 (Asia/Taipei). Scope: current 11-page manuscript, supplied reviewer screenshots/specification, local experimental evidence and original table data. This is not an OpenReview account inspection or a new literature/venue assessment.

## Reading order and changes

The argument remains: predictable decoded residuals -> data/metric/training -> measured rate control -> 2018/2022 repair -> channel-altitude split and fixed architecture controls -> actual roughness and original elevation control -> limitations -> conclusion. No evidence warrants replacing the original U-Net or deleting the original negative elevation finding.

- Table 3 still occupies the top of page 10, but now appears **before** the Section 8 Limitations heading. Its introduction and all remaining Section 7 text precede it; it no longer interrupts a limitations paragraph. The caption says identical *ground grid columns*, avoiding confusion with table columns. All five numerical rows are unchanged.
- Setup explicitly states all correctors train with 20 input/output channels; direct19 is an evaluation subset. Section 6 explains column-integrated tcwv belongs to neither near nor upper group. Table 2 repeats its exclusion.
- Section 4 now identifies the Figure 1 matched MPCA+PCA as retaining 4,384 components, distinct from Table 2 e995/e99/e97. Global/regional coding tasks remain explicitly non-equivalent.
- Table 1 preserves all-20 After/G and separately labelled G19/near13/upper6. Table 2 preserves widened After/G spacing and all before/after values. Captions were shortened without dropping metric/provenance qualifications.
- The direct19/all20 headline result is complete at the end of page 5; the deployment-cost paragraph starts on page 6. The within-variable comparison on page 7 is a complete paragraph, not a table-interrupted sentence. Disclosure remains with the conclusion on page 10; references start on page 11.
- Repeated qualifications were condensed, not removed. Bootstrap wording now allows fitted checkpoints in paired model comparisons. No font, margin, class, model, checkpoint or experimental result was changed.
- Standard page-top placement of Figures 1/2 and Tables 1/2 is retained close to their discussion; ordinary cross-page paragraph continuations remain. Forcing every object inline would reduce page use and is unnecessary for reading the argument.

## Reviewer coverage

| Supplied issue | Evidence in current paper | Disposition and remaining scope |
|---|---|---|
| Terrain complexity disconnected from repair | Section 7 / Fig. 2: actual orography, 5x5 local elevation SD, before/after curves on identical cells/times, gains, western and elevation-band controls | Requested analysis supplied. Associations do not prove terrain causality. |
| 2018 is inside CRA5 training; test 2022/2023 | Setup, Section 5 / Table 1: all 1,460 timestamps in 2022, frozen original checkpoint and statistics, no fine-tuning | Full 2022 supplied. One unseen year is not robustness across arbitrary shifts. |
| Small U-Net receptive field; larger corrector or linear baseline | Section 6: fixed equal-parameter bottleneck dilation, exact theoretical support, 480 gradient probes/model, full 2018/2022; affine low-capacity control | Both experiments supplied. Dilation does not broaden bulk gradient sensitivity; general architecture explanation remains open. |
| Classical rows have different CR; group baseline nRMSE missing | Table 2: measured CR/status R/M/N, near/upper Before, After and G for all five systems; Section 4 domain accounting | Supplied. Higher-CR rows are explicitly non-matched; Figure 1 is matched average rate, not equivalent-domain coding. |
| MPCA-only caption/text disagree about retraining | Setup and Table 2 caption: own historical weights, stage-1-only reconstruction training, no retraining for this table; baselines/PROVENANCE.md | Consistent. Historical training budgets and quantization train/deploy mismatch are disclosed. |
| ERA5 already provides orography and land-sea mask | Section 7: actual geopotential/9.80665, lsm>0.5; old proxy sensitivity separately retained as Table 3 | Supplied; static fields are analysis covariates, not added U-Net inputs. |
| Missing group confidence intervals | Setup and Section 5: paired circular seven-day blocks, 1,000 draws, both full years; paired ablation differences in Section 6 | Supplied with checkpoint/year/grid conditional scope; not seed or spatial uncertainty. |
| PDF title differs from system title | Title, running title and PDF metadata checked internally; manuscript title preserved from original local source | Open: accepted OpenReview title has not been independently retrieved; internal consistency does not certify system-title equality. |
| Figure 1 text small | Page 5 vector plot visually checked at 1,500-pixel page rendering; axes, values, legend and caption readable | Current version readable; no quantitative font-size claim relative to reviewer version. |

## Evidence and verification

The fresh manuscript auditor independently inspected the prior delivered manuscript and local evidence for issue coverage, without modifying sources or experiments. The main lane inspected the entire revised PDF and corrected the layout. Both passes use the supplied reviews; neither claims direct access to OpenReview.

Current numerical scope: Fig. 1 / Table 1 After20/G20 include tcwv; direct19 excludes it; near13 and upper6 are separate subsets. Fig. 2 is direct19. Table 3 uses the original pooled squared-error reduction, not G or Gbar. All three tables' data rows must match the archived 2026-10-29 source exactly; structural/numerical verification and independent ZIP rebuild are recorded in qa/layout_review_20261031/.

PDF QA: 11 A4 pages; embedded fonts; no overfull boxes or unresolved references/citations. A single underfull Figure 1 caption line remains visually acceptable. All pages rendered and inspected. Independent source ZIP reconstruction must match normalized text and all page-render hashes before delivery. No external submission, author metadata change, registration, payment or signature was performed.
