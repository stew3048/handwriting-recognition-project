# 日誌 2026-02-08 — Robustness STEP 1：破壞資料生成與視覺化

## 目標

進入 robustness 分析階段：刻意對 MNIST 做幾何破壞，之後用來觀察各模型（MLP / CNN / ResNet）的表現變化。本日只做 **STEP 1：資料生成 + 肉眼檢查用圖**，不做模型評估（STEP 2）。

## 完成內容

### 1. 破壞種類與強度

- **6 種幾何破壞**：translation、rotation、scale、shear、perspective、elastic。
- **每種 3 個強度**（mild / medium / strong），參數寫在 `tools/mnist_corruptions.py` 並有註解。
- **可重現**：固定 seed（42）、固定 64 筆 test 索引；每種 type 用固定 seed 偏移（不用 `hash()`，避免跨執行不一致）。

### 2. 流程：先存檔再畫圖

- **先產生破壞圖** → **存到 `data/corrupt_data/`**（.npy / .pt）→ **再從存檔讀出並畫 PNG** 到 `outputs/corrupt_viz/`。
- 這樣肉眼看到的 PNG 與 STEP 2 要讀的資料完全一致；STEP 2 直接讀 `data/corrupt_data/` 即可。

### 3. 產出位置與命名

- **資料**（`data/corrupt_data/`）：`indices.npy`、`labels.npy`、`clean.pt`、`<type>_s1.pt` / `_s2.pt` / `_s3.pt`（各 64×1×28×28）。
- **視覺化**（`outputs/corrupt_viz/`）：每種 type 三張 8×8 網格（`<type>_s1/s2/s3.png`）、一張 4 欄比較圖（`compare_<type>.png`）。
- 說明與讀檔範例見 `data/corrupt_data/README.md`。

### 4. 程式位置

- **`tools/mnist_corruptions.py`**：`get_fixed_indices`、`apply_corruption`、`save_grid`、`save_comparison`，以及各破壞參數。
- **`scripts/make_corrupt_viz.py`**：從專案根目錄執行 `python scripts/make_corrupt_viz.py`，會寫入 `data/corrupt_data/` 再畫出 `outputs/corrupt_viz/`。

### 5. 其他

- 資料目錄放在 **`data/corrupt_data/`**，與專案「資料從 `/data` 讀」的習慣一致。
- 若刪掉資料夾後要重做，再跑一次 `make_corrupt_viz.py` 即可。

## 下一步

- **STEP 2**：讀取 `data/corrupt_data/` 的破壞資料，對 MLP / CNN / ResNet 做 robustness 評估（準確率變化、worst case、embedding 變化等）。
