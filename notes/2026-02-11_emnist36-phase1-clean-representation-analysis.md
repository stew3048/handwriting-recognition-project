# EMNIST36 Phase 1（Clean）— CNN vs ResNet Representation 分析

## 實驗設定（對齊原規劃）

- 資料：EMNIST，標籤空間為 **0-9 + A-Z（36 類）**
- 訓練：**clean only**（無幾何破壞、無 augmentation）
- 模型：CNN、ResNet
- 評估重點：**representation geometry**（PCA / t-SNE / confusion / worst cases），不是只看 accuracy

---

## 整體結果

| Model | Test Loss | Test Accuracy |
|------|-----------:|--------------:|
| CNN | 0.2644 | 92.39% |
| ResNet | **0.1936** | **93.79%** |

> 結論（整體）：ResNet 在 clean baseline 上優於 CNN，且與訓練期觀察一致。

---

## Precision / Recall 是什麼？（白話版）

因為這次我們很常拿 `0` 和 `O` 當例子，這裡先用白話解釋兩個名詞：

- **Recall（召回率）**：  
  「某一類**真正的樣本**，有多少被模型找回來？」  
  以 `O` 為例：所有真正的 `O` 裡，有多少最後還被判成 `O`。

- **Precision（精確率）**：  
  「模型說某一類時，**到底有多少是真的**？」  
  以 `0` 為例：所有被模型判成 `0` 的樣本裡，有多少本來真的就是 `0`。

### 用這次數字直覺解讀

- 當我說「ResNet 把決策邊界往 `0` 拉」時，可以理解成：  
  模型更傾向把不確定樣本判成 `0`。
- 這通常會造成：
  - `0` 的 **recall 變高**（真 `0` 比較不會漏掉）
  - 但 `O` 的 **recall 變低**（真 `O` 更容易被吸去判成 `0`）

所以「`0` 更容易被保住，但犧牲了不少真正的 `O`」這句話，本質上就是在描述：  
**一類（`0`）的召回率提升，另一類（`O`）的召回率下降**，而且方向是偏向 `0`。

---

## 關鍵混淆（由 confusion matrix）

以下是最有研究價值的視覺相似對（顯示 GT→Pred）：

### CNN（best checkpoint）

- `0 -> O`: 270（6.75%）
- `O -> 0`: 488（61.00%）
- `1 -> I`: 141（3.52%）
- `I -> 1`: 355（44.38%）
- `2 -> Z`: 105（2.62%）
- `Z -> 2`: 123（15.38%）
- `5 -> S`: 64（1.60%）
- `S -> 5`: 226（28.25%）

### ResNet（best checkpoint）

- `0 -> O`: 136（3.40%）
- `O -> 0`: 575（71.88%）
- `1 -> I`: 125（3.12%）
- `I -> 1`: 362（45.25%）
- `2 -> Z`: 58（1.45%）
- `Z -> 2`: 148（18.50%）
- `5 -> S`: 95（2.38%）
- `S -> 5`: 111（13.88%）

### 觀察

- 混淆呈現**強烈不對稱**（例如 `O->0` 明顯高於 `0->O`），代表模型對某些書寫型態形成偏向性決策。
- ResNet 雖然整體更準，但並非每一個 pair 都同步改善；它主要改善的是整體分界，而非所有相似對都等比例降低。

---

## Digits vs Letters 的跨域錯誤

| Model | digit->letter | letter->digit |
|------|---------------:|---------------:|
| CNN | 979（2.45% of digit） | 2628（12.63% of letter） |
| ResNet | **761（1.90%）** | **2308（11.10%）** |

觀察：

- 兩模型都出現 **letter->digit** 高於 **digit->letter** 的現象，表示部分字母在 28x28 解析度下更容易被壓成「像數字」的形狀。
- ResNet 在跨域混淆上有明顯下降，與其整體 accuracy 優勢一致。

---

## Embedding 幾何（PCA / t-SNE）解讀重點

- 兩模型都能形成可分群結構，但在**相似字形群（0/O、1/I、2/Z、5/S）**附近仍有重疊區。
- ResNet 的聚類通常更緊、邊界較清楚，對應其較低 test loss 與較高 accuracy。
- worst-case 樣本大多位於 cluster 邊界或相似群交界，符合「混淆由幾何鄰近導致」的研究假設。

---

## 為什麼比 MNIST 難（Phase 1 結論）

- 類別從 10 提升到 36，決策邊界數量大幅增加。
- 多組類別存在結構相似（字母/數字共形），模型必須依賴更細微的局部筆畫差異。
- 同樣是 28x28，對細節辨識不利，容易把字母壓到接近數字流形。

---

## 產出路徑（可直接看圖）

### CNN

- `outputs/runs/20260210_231003_emnist36_cnn/eval/metrics.json`
- `outputs/runs/20260210_231003_emnist36_cnn/eval/confusion_matrix.png`
- `outputs/runs/20260210_231003_emnist36_cnn/eval/embedding_pca.png`
- `outputs/runs/20260210_231003_emnist36_cnn/eval/embedding_tsne.png`
- `outputs/runs/20260210_231003_emnist36_cnn/eval/top10_worst_cases.png`

### ResNet

- `outputs/runs/20260211_085917_emnist36_resnet/eval/metrics.json`
- `outputs/runs/20260211_085917_emnist36_resnet/eval/confusion_matrix.png`
- `outputs/runs/20260211_085917_emnist36_resnet/eval/embedding_pca.png`
- `outputs/runs/20260211_085917_emnist36_resnet/eval/embedding_tsne.png`
- `outputs/runs/20260211_085917_emnist36_resnet/eval/top10_worst_cases.png`

---

## 下一步（對齊原始研究計畫）

- 進入 **PHASE 2 — Controlled Difficulty Analysis**：
  - 以 `0/O, 1/I, 2/Z, 5/S` 作為核心 pair
  - 用 confusion + worst-case + embedding 局部鄰域做 pair-level 分析
  - 比較 CNN / ResNet 的決策邊界差異
