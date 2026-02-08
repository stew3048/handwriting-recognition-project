# 日誌 2026-02-08 — Robustness STEP 1：破壞資料生成與視覺化

## 目標

進入 robustness 分析階段：刻意對 MNIST 做幾何破壞，之後用來觀察各模型（MLP / CNN / ResNet）的表現變化。本日只做 **STEP 1：資料生成 + 肉眼檢查用圖**，不做模型評估（STEP 2）。

## 完成內容

### 1. 破壞種類與強度

- **6 種幾何破壞**（英文／中文與意思）：
  | 英文 | 中文 | 意思 |
  |------|------|------|
  | **translation** | 平移 | 整張圖在 x、y 方向隨機位移（例如 ±2 / ±4 / ±6 像素），模擬書寫或拍攝時位置偏移。 |
  | **rotation** | 旋轉 | 以影像中心為軸心隨機旋轉（例如 ±10° / ±20° / ±30°），模擬紙張或相機歪斜。 |
  | **scale** | 縮放 | 整體放大或縮小（例如 scale 0.9–1.1 / 0.8–1.2 / 0.7–1.3），模擬遠近或解析度差異。 |
  | **shear** | 剪切 | 沿水平方向做剪切變形（例如 ±10° / ±20° / ±30°），模擬透視或書寫傾斜。 |
  | **perspective** | 透視 | 四角隨機位移造成透視變形，模擬從斜角拍攝或紙張不平。 |
  | **elastic** | 彈性變形 | 局部不規則位移（平滑隨機位移場），模擬紙張皺褶或 MNIST 常見的彈性形變。 |
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
