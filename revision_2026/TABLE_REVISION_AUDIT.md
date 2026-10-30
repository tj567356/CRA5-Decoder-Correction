# Table revision audit

Date: 2026-10-29 (Asia/Taipei). Read-only independent source-to-result verification; no TeX or result artifacts modified.

## Verdict

No discrepancy found. The six Table 1 rows match the completed annual metrics, restored Table 3 data rows match the original Git manuscript, and Table 2/Table 3 explicitly exclude derived `tcwv` from the near/upper groups. Figure 1 and Table 1 now consistently use the all-20-channel metric in the labelled columns.

## Table 1: all-20 versus direct-channel metrics

An executed standard-library check parsed all six current `sec5_results.tex` rows and compared their displayed After20, G20, G19, GN and GU values with the original, dilated and affine annual JSONs. All 30 values pass at displayed precision.

| Corrector | Year | After20 | G20 (%) | G19 (%) | GN (%) | GU (%) |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 2018 | 0.06321 | 27.78 | 25.75 | 32.18 | 3.52 |
| U-Net | 2022 | 0.06236 | 28.32 | 26.40 | 32.79 | 3.60 |
| Dilated | 2018 | 0.06354 | 27.40 | 25.38 | 31.61 | 3.82 |
| Dilated | 2022 | 0.06270 | 27.93 | 26.01 | 32.21 | 3.91 |
| Affine | 2018 | 0.08471 | 3.20 | 1.29 | 1.43 | 0.82 |
| Affine | 2022 | 0.08425 | 3.16 | 1.31 | 1.45 | 0.85 |

Sources: `eval2018/summary.json`, `eval2022/summary.json`, the two RF run annual summaries, `revision_2026/linear_corrector/summary.json`, and the RF run `affine_eval2022/summary.json`. Shared all-20 baselines 0.0875180793 and 0.0870018767 correctly round to 0.08752/0.08700 in the caption. Primary direct19 values remain separately labelled as G19, not substituted for After20.

Figure 1 caption now explicitly says all 20 channels including derived `tcwv`, matching Table 1's After20/G20 columns. Its CRA5 2018 corrected value 0.06321 is therefore consistent. The caption's agreement refers to the metric; the additional model/year rows are separate experiments.

## Table 2: group columns and metric separation

The current table keeps separate Before, After and G columns. Its source places explicit `2em` spacing between After and G rather than concatenating them. Caption states 13 near-surface and 6 upper-air **direct** channels and says both exclude `tcwv`; their union contains 19 channels, not all 20. This agrees with group definitions and the previously audited executed baseline table. CR status R/M/N and quantized classical reconstruction qualifications remain present.

Visual readability and overfull-box acceptance remain the main lane's compiled-PDF responsibility; this report does not equate source column separation with completed visual QA.

## Table 3: restored elevation control

Executed a read-only `git show 13b9c866:paper/sec7_terrain.tex`, extracted the five data rows, and compared them with current Table 3 after whitespace normalization. The rows are identical, including counts, displayed gains, ratios and bold final row. Independently recomputed ratios from unrounded `research/fourcastnet/results/elevation_controls.csv` western proxy-land near/upper gains:

| Elevation band | Cells | Near (%) | Upper (%) | Unrounded gain ratio | Display |
|---|---:|---:|---:|---:|---:|
| 0–200 m | 415 | 39.9 | 5.0 | 7.92164524 | 7.9× |
| 200–600 m | 850 | 39.2 | 5.2 | 7.56061794 | 7.6× |
| 600–1200 m | 2311 | 33.5 | 4.7 | 7.06256314 | 7.1× |
| 1200–2000 m | 3113 | 40.1 | 4.7 | 8.58251869 | 8.6× |
| Above 2000 m | 984 | 41.9 | 4.3 | 9.65235094 | 9.7× |

The ratios correctly use unrounded gains: dividing the displayed one-decimal percentages need not reproduce them exactly. Caption correctly identifies 2018 western **proxy-land**, identical ground columns, near13/upper6 with `tcwv` excluded, and the legacy pooled-error reduction. Per-cell normalized RMSE is squared and pooled across channels/cells before the before/after square-root ratio; this differs from channel-mean nRMSE G and equal-weight fractional Gbar. The actual-orography roughness analysis remains separately labelled and its different cell counts are not mixed with this legacy table.

## Scope

This audit checks numerical and caption consistency. It does not re-execute inference, regenerate original terrain arrays, compile the latest paper, inspect its pages, or certify the final ZIP/PDF pair. Those checks remain with the main execution lane.
