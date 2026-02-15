# 怎麼讓 EMNIST36 訓練明天早上前跑完

## 目前進度（你睡前）

- **Run**：`20260210_231003_emnist36_cnn`（batch_size=10, CPU）
- **狀態**：約 8 分鐘時仍在**第 1 個 epoch**（0/10）
- 一個 epoch 在 CPU 上可能要 **10～20+ 分鐘**，10 個 epoch 粗估 **2～4 小時**

---

## 要「保證」明天訓練好，你需要做兩件事

### 1. 不要讓電腦休眠

- **Windows**：設定 → 系統 → 電源與睡眠 → 螢幕／電腦改為「永不」或至少設成 4 小時以上  
- 若用筆電，**接上電源**，並在電源計畫裡把「接上電源時：電腦進入睡眠」設為**永不**

只要電腦一休眠，背景訓練會停，明天就不會跑完。

### 2. 用「過夜腳本」跑（建議）

現在 Cursor 裡的那個訓練是**掛在 Cursor 的終端機**上的，關掉 Cursor 或關機就會停。

已經幫你加了一個**可以獨立跑**的腳本，關掉 Cursor 也會繼續：

1. **用檔案總管** 打開專案資料夾  
   `handwriting-recognition-project`
2. 到 **`scripts`** 資料夾，**雙擊**  
   **`run_emnist36_train_overnight.bat`**
3. 會跳出一個**黑色命令視窗**，裡面會一直印訓練進度
4. **不要關那個視窗**，也不要關機／休眠，讓它跑到結束
5. 跑完後，結果會寫在：
   - **最新 run 資料夾**：`outputs\runs\<timestamp>_emnist36_cnn\`
   - **日誌**：`outputs\emnist36_overnight_log.txt`

這樣你睡覺時是「用 .bat 在跑」，不是靠 Cursor，比較能保證明天早上前跑完。

---

## 若繼續用 Cursor 背景跑

- **不要關 Cursor**，也不要讓電腦休眠
- 明天打開 Cursor 後跟我說「現在跑到哪了」，我幫你查  
  `outputs\runs\` 裡最新的 `*_emnist36_cnn` 和 `experiments.csv`

---

## 明天怎麼確認有沒有跑完

- 看 **`outputs\runs\`** 底下有沒有新的 **`*_emnist36_cnn`** 資料夾
- 打開該資料夾裡的 **`experiments.csv`**，若有 **10 行**（1 行表頭 + 10 行 epoch），就是 10 個 epoch 都跑完了
- 或直接跟我說「現在跑到哪了」，我幫你對照 run 和 csv 回報
