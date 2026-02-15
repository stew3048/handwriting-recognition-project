# 2026/2/16 日誌 — EMNIST36 平衡抽樣與 Center Loss 實驗

## 今日實驗設定

- **資料**：EMNIST 36 類（0–9 + A–Z），平衡抽樣每類 5000 張訓練、800 張測試，同一份 train/test 給三種模型。
- **訓練**：batch_size=10，epochs=10，皆存 best checkpoint。
- **比較模型**：
  1. **CNN**（一般）
  2. **ResNet**（一般，無 embedding 層）
  3. **ResNet + Center Loss**（128 維 embedding + Center Loss，loss = CE + 0.1×Center）

---

## 評估結果（同一份完整 test set）

| 模型 | Test accuracy | Test loss |
|------|----------------|-----------|
| CNN | **89.34%** | 0.3240 |
| ResNet（一般） | **90.48%** | 0.2777 |
| ResNet + Center Loss | **88.91%** | 0.3514 |

**結論：即使加上 Center Loss，整體 test accuracy 沒有比較好，甚至略低於一般 ResNet。**

---

## 分析

1. **為何 Center Loss 沒贏？**
   - **資料量與訓練長度**：每類 5000、10 epochs 可能還不足以讓 128 維 embedding + 類別中心同時收斂得好；Center Loss 多一組參數（36×128 的 centers）和一個優化器，需要更穩定的訓練。
   - **權重 0.1**：`total_loss = CE + 0.1 * Center` 是常見設定，但在目前設定下可能讓 embedding 被拉向中心、反而削弱分類邊界；可嘗試調小（如 0.01）或調大，或配合 learning rate schedule。
   - **易混字是否改善？**：需要看 `metrics.json` 裡 O/0、I/1、Z/2、S/5 的個別準確率與 2×2 混淆矩陣。有可能「整體 acc 略降、但易混字對的互混減少」——若如此，代表 Center Loss 有把相似字拉開，但其他類別略受影響。
   - **baseline 已不弱**：一般 ResNet 在平衡資料上已達 90.48%，要再提升本來就難；Center Loss 的效益在「特徵空間類內緊湊、類間分開」，不見得直接反映在 overall accuracy，而更可能反映在易混字對的混淆上。

2. **後續可做的**
   - 比對三份 `metrics.json` 的 **acc_O, acc_0, acc_I, acc_1, acc_Z, acc_2, acc_S, acc_5**：若 Center Loss 在這幾對明顯較好，可寫成「整體略降但易混字改善」的結論。
   - 拉長訓練（例如 20–30 epochs）或調 Center Loss 權重 / center 學習率再跑一輪。
   - 看三份 **embedding_tsne.png**：Center Loss 的 128 維 t-SNE 是否在 O/0、I/1、Z/2、S/5 上更分群。

3. **紀錄用途**
   - 如實寫「加 Center Loss 在這次設定下整體沒有比較好」，對之後調參或改設定有幫助，也避免只報好的結果。

---

## 相似字對在 Center Loss 下 embedding 有沒有分比較開？

依三份評估的 metrics 與 2×2 混淆矩陣整理（數字為該方向錯誤數，越少代表越分得開）。

### 互混數量

| 字對 | 混淆方向 | CNN | ResNet | Center Loss |
|------|----------|-----|--------|-------------|
| **O/0** | O→預測成0 | 301 | 243 | 256 |
| | 0→預測成O | 835 | 1139 | **1014** |
| **I/1** | I→預測成1 | 200 | 208 | **159** |
| | 1→預測成I | 410 | 438 | 643 |
| **Z/2** | Z→預測成2 | 42 | 29 | 82 |
| | 2→預測成Z | 368 | 421 | **188** |
| **S/5** | S→預測成5 | 40 | 13 | **253** |
| | 5→預測成S | 334 | 410 | **107** |

### 逐對解讀

- **O/0**：Center Loss 的 0→O 略少（1014 vs ResNet 1139），O→0 差不多。有稍微好一點，不算明顯分更開。
- **I/1**：I→1 在 Center Loss 最少（159），字母 I 較少被認成 1，embedding 在「I」這一側有分比較開；但 1→I 變多（643），數字 1 更容易被認成 I。
- **Z/2**：2→Z 在 Center Loss 明顯變少（188 vs 421），數字 2 較少被認成 Z；Z→2 變多（82 vs 29），字母 Z 更容易被認成 2。2 那邊分得比較開，Z 那邊沒有。
- **S/5**：S→5 在 Center Loss 暴增（253 vs 13），很多 S 被認成 5；5→S 變少（107）。S 和 5 沒有分比較開，S 反而更黏在 5。

### 小結

只有 **I/1 的字母 I**、**Z/2 的數字 2**、**O/0** 略好，在 Center Loss 下 embedding 有在「單邊」分比較開；**S/5**、**Z→2**、**1→I** 反而更混。**Center Loss 在部分相似字對單側有分開，並非每一對都變好，S/5 明顯變差。**

---

## 下一步

- **直接測試 ViT**（Vision Transformer）：用同一套平衡 train/test 資料，接 ViT 做 EMNIST36 分類，再與 CNN、ResNet、ResNet+Center Loss 比較整體與易混字對表現。

---

## 輸出位置

- 訓練 run：`outputs/runs/20260215_*_emnist36_{cnn,resnet,centerloss}/`
- 評估結果：`outputs/emnist36_eval_cnn/`、`outputs/emnist36_eval_resnet/`、`outputs/emnist36_eval_centerloss/`（內含 metrics.json、混淆矩陣、t-SNE 圖等）
