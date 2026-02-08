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

（待補充：資料來源、下載方式、放置路徑與格式說明。）

---

## 訓練

（待補充：訓練指令與參數說明。）

---

## 推論

（待補充：如何載入模型與執行推論。）
