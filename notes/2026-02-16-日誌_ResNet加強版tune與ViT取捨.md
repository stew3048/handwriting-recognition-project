# 2026/02/16 日誌 — ResNet 加強版 tune 與 ViT 取捨

## 今日重點

1. **下午：ResNet 64 維 + 延遲 Center Loss 評估**：64 維（不放大到 128 維）、前 5 個 epoch 僅 CE、**後五個 epoch（第 6–10）才用 metric learning（Center Loss）**，同一份平衡資料 5000/類、10 epochs，評估結果與易混字對分析已補於下方。
2. **ViT 暫不採用**：以 timm 預訓練 ViT-Tiny 跑 EMNIST36，CPU 下 2 小時仍無 epoch 輸出，改把重心放在 ResNet + Center Loss 加強版。
3. **ResNet + Center Loss 加強版**：實作三項策略並開始 20 epochs 訓練（進行中）。

---

## 一、下午：ResNet 64 維 + 延遲 Center Loss 評估結果

**設定**：ResNet **64 維 embedding**（不升維到 128）、前 5 epoch 僅 CrossEntropy，**第 6–10 epoch 才加入 Center Loss**（alpha=0.05）。平衡抽樣每類 5000 訓練、800 測試，batch_size=10，10 epochs。

**評估結果（同一份完整 test set）**：

| 指標 | 數值 |
|------|------|
| **Test Accuracy** | **91.03%** |
| **Test Loss** | 0.2572 |

此版本優於一般 ResNet（90.48%）、CNN（89.34%）與 128 維從頭加 Center Loss（88.91%）。

### 相似字母與數字配對分析（易混字對）

以下為 64 維 + 延遲 Center Loss 在各易混字對上的**類別準確率**（該類被預測正確的比例）與**樣本數**。

| 字對 | acc_O / acc_0 | acc_I / acc_1 | acc_Z / acc_2 | acc_S / acc_5 | acc_A / acc_4 | acc_G / acc_6 | acc_B / acc_8 |
|------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| **64 維延遲 CL** | 64.8% / 73.5% | 66.3% / 76.8% | 91.8% / 91.9% | 95.1% / 92.2% | 95.5% / 95.7% | 77.5% / 94.9% | 96.4% / 97.7% |
| n（該類樣本數） | 800 / 4000 | 800 / 4000 | 800 / 4000 | 800 / 4000 | 800 / 4000 | 800 / 4000 | 800 / 4000 |

**簡要解讀**：

- **O/0**：字母 O 64.8%、數字 0 73.5%；O→0 混淆率約 1−0.648≈35%，0→O 約 1−0.735≈27%。
- **I/1**：I 66.3%、1 76.8%；易互相誤判，與其他模型類似。
- **Z/2、S/5**：Z/2 與 S/5 皆約 91–95%，64 維延遲版在 S/5 明顯優於 128 維版（S 從 66.1% 提升到 95.1%）。
- **A/4、B/8**：A/4、B/8 皆 95% 以上，混淆低。
- **G/6**：G 僅 77.5%、6 為 94.9%；G→6 混淆較多，是後續可加強的配對。

**與 128 維從頭加 Center Loss 的差異**：128 維版在 S/5 大崩（S 約 66%），64 維延遲版在整體與多數易混字對上較均衡，且整體 Test Accuracy 最高（91.03%）。

---

## 二、ViT 取捨

- **原計畫**：用 timm 的 ViT-Tiny（預訓練）、28x28→224x224 適配、平衡資料 5000/類、與 CNN/ResNet 比較。
- **實際**：CPU 訓練極慢，約 2 小時仍無第一個 epoch 完成，不適合目前環境。
- **決定**：暫不做 ViT，改專注 tune ResNet 加強版（64 維 + 延遲 Center Loss）。

---

## 三、ResNet 加強版三項策略

在既有「64 維 embedding、前 5 epoch 僅 CE、第 6 起 alpha=0.05」基礎上，今日實作：

| 策略 | 內容 |
|------|------|
| **1. CosineAnnealingLR** | 對 model optimizer 使用 `torch.optim.lr_scheduler.CosineAnnealingLR`，T_max=20，eta_min=1e-5。學習率從 1e-3 依餘弦曲線下降。 |
| **2. Alpha 漸進式增強** | 第 6–10 epoch：alpha=0.05（穩定）；第 11–20 epoch：alpha 從 0.05 線性增加到 0.1。 |
| **3. Epochs 增至 20** | 預設訓練 20 個 epoch，觀察後段 alpha 漸增與 CosineAnnealing 的綜合效果。 |

**訓練設定**：平衡抽樣 5000/類 train、800/類 test，batch_size=10，CPU，--save_best。  
**Run 目錄**：`outputs/runs/20260216_214309_emnist36_centerloss/`（訓練進行中）。

---

## 四、訓練進行狀況（撰寫日誌當下）

- 已跑完 Epoch 1–7。
- Epoch 1–5：CE only，val acc 升至約 90.27%。
- Epoch 6–7：加入 Center Loss（alpha=0.05），val acc 約 90%。
- 學習率依 CosineAnnealing 正常下降。
- 預計再約 3–4 小時可跑完 20 epochs。

---

## 五、後續待辦

- 訓練結束後：用 `evaluate_emnist_digits_letters.py --model resnet_centerloss_64` 評估 best checkpoint，產出 Test accuracy、混淆矩陣、易混字對（O/0, I/1, Z/2, S/5, A/4, G/6, B/8）。
- 與先前 91.03%（10 epochs、固定 alpha=0.05）比較，看 CosineAnnealing + alpha 漸進 + 20 epochs 是否再提升。

---

## 六、檔案與指令

- **訓練腳本**：`scripts/train_emnist36_centerloss.py`（已含 CosineAnnealingLR、get_alpha() 漸進、預設 20 epochs）。
- **訓練指令**：`python scripts/train_emnist36_centerloss.py --samples_per_class_train 5000 --samples_per_class_test 800 --batch_size 10 --epochs 20 --save_best --cpu`
- **評估指令**：`python scripts/evaluate_emnist_digits_letters.py --checkpoint <run_dir>/best_emnist36_centerloss.pt --model resnet_centerloss_64 --output_dir outputs/emnist36_eval_centerloss_64 --cpu`
