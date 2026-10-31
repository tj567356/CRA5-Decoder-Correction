# TAAI 2026 — Recoverable Error in Neural Climate Compression

**Recoverable Error in Neural Climate Compression: A Rate-Matched Study of CRA5 over CONUS**

本倉庫保存最終11頁論文、LaTeX、論文實驗程式、已量測結果、修正器權重及來源紀錄。
原始U-Net與既有結果保留；Dilated是固定的感受野消融，不是架構搜尋。

## 重現實驗

請先讀[逐步重現指南](docs/REPRODUCING.md)。已實際驗證：五組圖表數值重建、4筆真實2017樣本的CPU推論、兩年度bootstrap與理論感受野。指南明確區分這些檢查與完整年度資料處理/重訓；沒有宣稱全新環境一鍵重現。

## 論文與原始檔

- [最新版論文PDF](paper/main.pdf)
- [LaTeX入口](paper/main.tex)：九節正文、兩張圖、參考文獻、LNCS樣式均在`paper/`。
- [交付用LaTeX ZIP](revision_2026/final/TAAI2026_source.zip)
- [Reviewer逐項回應](revision_2026/reviewer_response_matrix.md)
- [全文排版與邏輯檢查](revision_2026/LAYOUT_REVIEW_20261031.md)

LaTeX編譯方式（已用MiKTeX 25.12驗證）：

```sh
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

交付PDF為11頁A4；原始碼ZIP曾獨立重建，文字與11頁渲染結果一致。

## 圖表、實驗與程式對照

| 項目 | 主要程式 | 結果 |
|---|---|---|
| CRA5解碼、20通道、衍生tcwv | `research/fourcastnet/scripts/cra5/` | `research/fourcastnet/results/cra5_rmse.json` |
| 原U-Net訓練 | `research/fourcastnet/scripts/cnn/train_cnn.py`、`model.py`、`dataset.py` | `artifacts/correctors/cra5/`、`provenance/training/` |
| 原ERA5資料預處理、MPCA係數 | `research/fourcastnet/mpca_train_basis_bundle/scripts/` | 訓練統計量與MPCA基底在`artifacts/` |
| Fig.1碼率匹配 | `research/new_20/scripts/rate_matched_val.py` | `research/new_20/results/rate_matched_val.json` |
| Table 1原U-Net年度評估 | `revision_2026/scripts/evaluate_frozen.py` | `research/fourcastnet/results/revision_2026/eval2018/`、`eval2022/` |
| Table 2傳統方法 | `revision_2026/scripts/baseline_*.py` | `research/fourcastnet/results/revision_2026/baselines/` |
| 全年2022資料流程 | `revision_2026/scripts/*2022*.py`及相關shell wrappers | preparation manifests、`2022_acceptance_verification.json` |
| Bootstrap | `research/fourcastnet/scripts/analysis/bootstrap_gain.py` | `bootstrap2018/`、`bootstrap2022/` |
| Fig.2粗糙度 | `revision_2026/scripts/fetch_static.py`、`research/fourcastnet/scripts/analysis/terrain_roughness.py` | `research/fourcastnet/results/revision_2026/terrain/` |
| Table 3原海拔控制 | `research/fourcastnet/scripts/analysis/elevation_controls.py`、`sp_terrain_control.py` | 原始elevation/control CSV與JSON |
| Dilated感受野消融 | `revision_2026/scripts/rf_ablation.py`、`rf_theory.py`、`rf_erf.py`與postprocessing | `research/fourcastnet/results/revision_2026/rf_ablation_d23_seed0_20261029/` |
| 低容量Affine | `revision_2026/scripts/linear_corrector.py` | `revision_2026/linear_corrector/`、RF結果中的`affine_eval2022/` |
| 運算成本 | `revision_2026/scripts/profile_corrector.py`與RF evaluator | `corrector_cost.json`、`inference_cost.json` |

## 權重與基底

已附8份修正器權重：CRA5原U-Net、MPCA-only、e995/e99/e97、匹配碼率n4384、Dilated、Affine。
正規化統計量、通道順序、兩組MPCA第一階段基底與PCA metadata亦已附上。

- `artifacts/correctors/`：從Ubuntu補收的6份原修正器權重。
- `revision_2026/linear_corrector/best.pt`：Affine。
- RF結果目錄中的`best.pt`：DilatedUNet2。
- `artifacts/statistics/`：2014–2016統計量。
- `artifacts/transforms/`：MPCA基底及PCA metadata。
- `artifacts/ARTIFACT_AVAILABILITY.json`：來源與SHA256。

四份較大的PCA模型在同一倉庫的
[camera-ready-2026-10-31 Release](https://github.com/tj567356/NEWTAAI2026/releases/tag/camera-ready-2026-10-31)。
倉庫維持Private，附件也需要倉庫讀取權限。下載後依
`artifacts/RELEASE_ASSETS.json`的`install_path`放置，並驗證SHA256。
不下載也能查閱論文、程式與已量測結果。

## 統計口徑

所有修正器以20個輸入/輸出通道訓練。direct19僅在評估時排除衍生tcwv；
近地表13、高空6也排除整根大氣柱的tcwv。Fig.1與Table 1的After20/G20包含它。
Fig.1的匹配global PCA保留4,384成分，與Table 2的e995/e99/e97設定不同。
各壓縮器使用自己的修正權重。CRA5全球268通道與MPCA區域20通道任務不同，
平均碼率匹配不等於完全相同領域的壓縮器排名。Table 3使用舊版pooled-error指標。

## 重現前須知

這是有版本證據的研究程式快照，**尚未宣稱乾淨環境一鍵重現**：

1. 部分腳本保留原Ubuntu路徑`/home/tong/fourcastnet_us_mpca`與`/home/tong/new_20`。
   本倉庫將來源放在`research/`下；執行前需調整資料/模型路徑或建立對應工作根目錄。
2. 不少歷史腳本在import時直接執行工作。先閱讀程式，不要任意import整個腳本樹。
3. 原始ERA5/CRA5年度資料、完整解碼陣列、訓練cache不附在Git內。
   [2022資料取得與核對流程](revision_2026/2022_DATA_PREPARATION.md)描述來源與檢查。
4. 官方CRA5大權重與外部decoder package由上游取得，不在這裡重新分發。
   `research/fourcastnet/scripts/cra5/README.md`與`run_decode.sh`說明獨立解碼環境。
5. `research/environment/`保留原訓練環境快照，並不保證適用所有平台/GPU。
6. 原始腳本、資料來源與雜湊保留於provenance；新加入的下載工具不重新選模型或超參數。

`REPRODUCTION_GAPS.md`記錄仍需使用者設定的事項，`UPLOAD_MANIFEST.json`記錄倉庫檔案雜湊。
Ubuntu原始model.py另存在`provenance/ubuntu_sources/`，其UNet2類別與目前版本AST一致；
目前版本只另加DilatedUNet2，不覆寫原類別。

未擅自替作者指定新的軟體授權。既有程式、LNCS樣式及外部資料/模型的原授權與使用條款仍適用。

上傳副本的Python/Shell檔已統一為Unix LF換行；Shell檔保留可執行權限。科學程式內容未改，換行前後雜湊記錄於`provenance/CODE_LINE_ENDINGS.json`；本機與Ubuntu原始檔未改。
