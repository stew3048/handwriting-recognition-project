# 手寫辨識專案 (Handwriting Recognition Project)

本專案用於手寫辨識的實驗與開發，先以**手寫數字**為目標，架構設計保持通用，之後可擴充至英文、中文等其它字元。

---

## 目錄結構

```
handwriting-recognition-project/
├── .gitignore
├── README.md
├── requirements.txt
├── data/              # 資料集與衍生資料（如 MNIST、EMNIST 或自建資料）
├── models/            # 模型定義
├── scripts/           # 訓練、評估、前處理等腳本
├── outputs/           # 訓練產出、checkpoints、預測結果
├── configs/           # 設定檔（YAML/JSON）
└── notes/             # 實驗筆記與紀錄
```

---

## 環境設定

1. 建議使用 Python 3.8+ 與虛擬環境。
2. 安裝依賴：

   ```bash
   pip install -r requirements.txt
   ```

3. 若使用 CUDA，請依環境安裝對應的 PyTorch 版本。

---

## 資料集

- **MNIST**：手寫數字 0–9，執行訓練腳本時會自動下載到 `data/mnist/`。
- **EMNIST Letters**：英文字母 A–Z（26 類），自動下載到 `data/emnist/`。

---

## 訓練

從專案根目錄執行（首次會自動下載資料）：

```bash
# MNIST + MLP（預設 10 epochs）
python scripts/train_mnist_mlp.py

# MNIST + CNN
python scripts/train_mnist_cnn.py

# 英文字母 + CNN（或 --model mlp）
python scripts/train_emnist_letters.py --model cnn
```

常用參數：`--epochs`、`--batch_size`、`--lr`、`--augment`、`--save_best`、`--out_dir`。若遇 CUDA/cuDNN 錯誤可加 `--cpu` 強制用 CPU。快速驗證可加 `--epochs 1`。

實驗紀錄會寫入 `outputs/experiments_*.csv`，checkpoint 存於 `outputs/`。調參可參考 `configs/train_mnist_example.yaml`。

---

## 推論

（待補充：如何載入模型與執行推論。）
