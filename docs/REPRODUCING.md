# 實驗重現指南

本指南分成三個層次，請不要混為一談：

| 層次 | 實際做的事 | 是否重新執行U-Net |
|---|---|---|
| A：圖表數值重建 | 從已量測的JSON/CSV重建Fig.1、Fig.2、Table 1–3 | 否 |
| B：真實樣本推論 | 4筆2017驗證資料，載入原權重重新預測修正量 | 是 |
| C：完整科學實驗 | 原始資料準備、全年推論、編碼測量、訓練及消融 | 需要原年度資料、工作路徑及完整環境 |

A、B已在Ubuntu Python 3.12.11與原研究環境實際執行通過；B使用CPU。
這不等於已在全新電腦從零安裝、下載全部年度資料並重訓成功。
所有新指令都寫入新的輸出目錄，不能覆寫論文原結果。

## 1. 取得倉庫與安裝環境

目前倉庫是Private，必須先有讀取權限。接受GitHub協作者邀請後：

```sh
git clone https://github.com/tj567356/NEWTAAI2026.git
cd NEWTAAI2026
python3 -m venv .venv-reproduce
source .venv-reproduce/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy matplotlib
```

這是A層所需的最小環境。Windows可用`py -m venv .venv-reproduce`，
PowerShell啟用指令為`.\.venv-reproduce\Scripts\Activate.ps1`。

B層另需PyTorch；不需要GPU的安裝方式：

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

原研究環境的完整版本快照在`research/environment/requirements-fourcastnet.txt`。
已測試的研究環境包含PyTorch 2.11.0+cu128、NumPy 2.5.1、Matplotlib 3.11.1。
上述最小安裝指令未另做全新環境安裝驗證，且不同平台/版本不保證逐位元一致。
CPU smoke test以逐通道誤差的明示容忍度核對，而非要求GPU/CPU完全相同。
完整CRA5解碼使用另一套環境，不能把兩者依賴盲目混裝；見
`research/fourcastnet/scripts/cra5/README.md`與`run_decode.sh`。

## 2. 權重、PCA與必要資料

### A層：不用額外下載

圖表重建需要的JSON/CSV已在倉庫，不需要年度氣象資料、GPU或PCA大檔。

### B層：不用額外下載PCA

以下已在倉庫：

- `examples/cra5_2017_smoke.npz`：原2017驗證cache的第0–3筆正規化輸入與殘差目標，約6.7 MB。
- `examples/cra5_2017_smoke.json`：來源、通道順序、雜湊及逐通道預期結果。
- `artifacts/correctors/cra5/best.pt`：原U-Net，SHA256以`44c57a0f...`開頭。
- `artifacts/statistics/train_2014_2016_channel_stats.npz`：固定訓練統計量。

樣本是ERA5/CRA5衍生的研究驗證資料；資料來源引用見樣本JSON及論文參考文獻。
它不是2018/2022測試集，也不代表全年精確度。

### C層：大基底與原始資料

在同一倉庫的Release下載四份PCA檔：

https://github.com/tj567356/NEWTAAI2026/releases/tag/camera-ready-2026-10-31

以`artifacts/RELEASE_ASSETS.json`逐項核對SHA256，按`install_path`放置。
Private倉庫的Release也需要存取權限。這些大檔不影響A/B層。
八份小型修正器權重、兩組MPCA空間基底及統計量已在倉庫。

完整ERA5/CRA5年度資料、解碼陣列及正規化cache不附在Git。
完整2022資料來源、原生r500/r850、時間/座標/通道核對與下載腳本，見
`revision_2026/2022_DATA_PREPARATION.md`。2014–2018的原始預處理程式在
`research/fourcastnet/mpca_train_basis_bundle/scripts/`。
官方CRA5大權重與decoder package由上游取得，不在本倉庫重新分發。

## 3. 設定資料路徑

A/B層的新指令以倉庫根目錄為基準，不依賴作者的`/home/tong`路徑。
例如B層可用`--repo /absolute/path/NEWTAAI2026`指定不同倉庫位置。

歷史的完整訓練/壓縮腳本仍可能使用原路徑：

```text
/home/tong/fourcastnet_us_mpca
/home/tong/new_20
```

完整C層需要配置研究工作目錄、正規化cache、資料、權重及擬合基底。
不能把`research/`直接當作完整的原研究工作目錄，也不要直接執行會寫入舊結果的腳本。
部分歷史程式在import時就執行工作；請先閱讀並配置隔離輸出位置。
`--fcn-results`、`--mpca-results`、`--revision-root`可讓A層讀取另一份結果目錄。

## 4. 少量真實樣本跑通原U-Net

在倉庫根目錄執行：

```sh
python scripts/smoke_corrector.py --device cpu --out outputs/smoke_cpu
```

預期：

- `passed: true`；輸入/輸出皆為`[4,20,97,241]`。
- 原U-Net參數數量`1646548`。
- 這4筆樣本的20通道平均nRMSE約`0.09145616 -> 0.06539719`。
- 逐通道結果通過`rtol=1e-3, atol=1e-5`核對。
- 完整紀錄在`outputs/smoke_cpu/verification.json`。

程式只載入固定權重推論，不更新權重，不挑模型，不使用測試年調參。
這四筆資料是可攜式功能檢查，**不能拿它代替論文的全年結果**。
若輸出資料夾已存在，請換一個新名字，例如`outputs/smoke_cpu_2`。

## 5. 分別重建各圖表的數值

以下只讀取已儲存的量測結果，不重新跑模型、編碼器或訓練。
新畫的圖呈現相同數值，但不承諾與論文PDF逐像素相同。

```sh
python scripts/reproduce_paper.py --target fig1   --out outputs/fig1
python scripts/reproduce_paper.py --target table1 --out outputs/table1
python scripts/reproduce_paper.py --target table2 --out outputs/table2
python scripts/reproduce_paper.py --target fig2   --out outputs/fig2
python scripts/reproduce_paper.py --target table3 --out outputs/table3
```

也可一次執行：

```sh
python scripts/reproduce_paper.py --target all --out outputs/paper_all
```

每組輸出有CSV及`verification.json`；兩張圖另有PDF。預期核對如下：

| 項目 | 預期結果 |
|---|---|
| Fig.1 | CRA5修正後`0.06321`；MPCA-only `0.08688`；匹配global PCA `0.12599`；均為20通道 |
| Table 1 | 6列：U-Net/Dilated/Affine，各含2018/2022。After20依序`0.06321,0.06236,0.06354,0.06270,0.08471,0.08425` |
| Table 2 | 5系統×near/upper，共10列；含Before/After/G，兩組排除tcwv |
| Fig.2 | 全部陸地及西部陸地各10組、各有修正前後曲線；使用19直接通道 |
| Table 3 | 格點數`415,850,2311,3113,984`；近/高空比`7.9,7.6,7.1,8.6,9.7`；使用原pooled-error指標 |

這些指令已用倉庫中的對應JSON/CSV在隔離Ubuntu測試目錄執行通過。
測試紀錄摘要在`docs/REPRODUCTION_VALIDATION.json`。

## 6. 重新計算bootstrap與理論感受野

Bootstrap可從倉庫已附的逐時間點平方誤差統計量重新抽樣，無需下載年度原始場：

```sh
python research/fourcastnet/scripts/analysis/bootstrap_gain.py research/fourcastnet/results/revision_2026/eval2018/sufficient_statistics.npz --out outputs/bootstrap2018
python research/fourcastnet/scripts/analysis/bootstrap_gain.py research/fourcastnet/results/revision_2026/eval2022/sufficient_statistics.npz --out outputs/bootstrap2022
python revision_2026/scripts/rf_theory.py --out outputs/rf_theory
```

預設bootstrap為28時間點區塊、1000次、seed 20261026。原U-Net直接19通道G的95%區間，
2018約`[25.32,26.18]%`，2022約`[25.90,26.93]%`。
理論中央感受野為`72x72 -> 96x96`；有效感受野需要另外的梯度診斷，不能混稱。

## 7. 完整年度推論與從頭訓練

全年凍結推論入口為`revision_2026/scripts/evaluate_frozen.py`，支援
`--root`、`--mem`、`--checkpoint`、`--year`、`--out`；完整2018/2022各必須1460筆。
2022額外核對完整manifest、資料雜湊、時間與原權重。
它的`--root`必須是已配置的原研究工作目錄，內含`scripts/cnn/`及
`mpca_train_basis_bundle/stats/`；不是未配置的Git倉庫根目錄。

```sh
python revision_2026/scripts/evaluate_frozen.py --root /path/to/configured/fourcastnet_workspace --year 2018 --mem /path/to/cache2018 --checkpoint artifacts/correctors/cra5/best.pt --out outputs/full2018
python revision_2026/scripts/evaluate_frozen.py --root /path/to/configured/fourcastnet_workspace --year 2022 --mem /path/to/cache2022 --checkpoint artifacts/correctors/cra5/best.pt --out outputs/full2022
```

從頭重訓與重跑碼流還需下載/預處理原資料、擬合基底、配置獨立工作目錄及GPU環境。
此包裝工作沒有重新執行這整條鏈；原實驗的程式、結果與執行證據已保留，
但不能將A層重建圖表或B層樣本推論宣稱成C層已完成。
重訓必須仍使用2014–2016、2017驗證選模型，2018/2022只評估，並另存新結果。

## 倉庫是否要拆開

目前仍保留`paper/`、`research/`、`docs/`在同一倉庫，沒有搬走或刪除論文。
這樣能以同一提交版本對照論文與程式。若作者之後決定論文另放新倉庫，
程式倉庫仍保留本指南，並以固定commit/tag互相連結。
