# 為什麼 MLP 容易失敗，而 CNN 能有效改善？  
## —— 從 Inductive Bias 角度理解手寫數字辨識

## 背景說明

在本專案中，我們比較 **多層感知器（Multi-Layer Perceptron, MLP）** 與  
**卷積神經網路（Convolutional Neural Network, CNN）** 在 MNIST 手寫數字資料集上的表現。

雖然兩種模型都能達到不錯的整體準確率（accuracy），  
但在**錯誤型態（failure modes）**與**內部表示（embedding / representation）**上，存在本質差異。

本文件從 **inductive bias（歸納偏置）** 的角度，說明這些差異為何會發生。

---

## MLP：高度彈性，但缺乏空間結構假設

MLP 在處理影像時，會先將影像攤平成一維向量：

```
28 × 28 影像 → flatten → 784 維向量 → 全連結層
```

### 核心特性

- **沒有顯式的空間結構**
  - 相鄰的像素在數學上只是不同維度
  - 模型本身不知道哪些像素屬於同一條筆畫

- **Inductive bias 很弱**
  - 不假設局部性（locality）
  - 不假設平移不變性（translation invariance）
  - 假設越少，hypothesis space 越大

- **資料效率低、容易 overfitting**
  - 因為沒有結構性限制，模型需要更多資料才能學到穩定規則
  - 在資料有限時，容易記住像素層級的偶然關係

### 在 MNIST 上的實際影響

對 MLP 而言：

- 同一個視覺 pattern，只要出現在不同位置，就會被視為「不同證據」
- 輕微的平移、傾斜、筆畫位置改變，都可能大幅影響預測
- 容易出現以下混淆：
  - 7 ↔ 9
  - 3 ↔ 8
  - 1 ↔ 3

---

## CNN：符合影像特性的結構性假設

CNN 在設計上引入了**符合影像本質的 inductive bias**。

### 主要 inductive bias

1. **局部性（Locality）**
   - 卷積只關注局部區域
   - 前幾層自然學到邊緣、角點、短筆畫

2. **權重共享（Weight Sharing）**
   - 同一個 filter 會套用在所有空間位置
   - pattern 不論出現在左上或右下，都能被同樣偵測

3. **近似平移不變性**
   - pooling 與層級式特徵組合
   - 降低模型對小幅平移的敏感度

### 典型 CNN 架構流程

```
影像
 ↓
Convolution + ReLU
 ↓
Pooling
 ↓
（重複多層）
 ↓
Feature Map → Flatten / Global Pooling
 ↓
Embedding → 分類器
```

其中，**flatten 或 pooling 前的 feature map，本質上就是影像的 embedding 表示**。

---

## 為什麼 CNN 學得更快、泛化能力更好？

因為 CNN 事先假設：

- 影像具有局部連續結構
- 相同的視覺 pattern 會在不同位置重複出現

這讓 CNN 能夠：

- 用更少的資料學到有意義的表示
- 關注「這是什麼 pattern」，而非「它出現在哪一個像素座標」
- 對平移、輕微變形具有更好的魯棒性

相較之下，MLP 必須**完全依賴資料本身**去學這些不變性，效率與穩定性都較差。

---

## 總結比較

| 面向 | MLP | CNN |
|----|----|----|
| 空間結構感知 | ❌ 無 | ✅ 明確 |
| 局部性假設 | ❌ | ✅ |
| 平移不變性 | ❌ | ✅（近似） |
| Inductive bias | 弱 | 強（符合影像） |
| 資料效率 | 低 | 高 |
| 常見錯誤 | 位移／傾斜造成誤判 | 較偏向語意層級錯誤 |

---

## 重點結論

> MLP 雖然具有高度表達彈性，但缺乏對影像結構的基本假設；  
> CNN 透過局部性與權重共享等 inductive bias，大幅降低 sample complexity，  
> 因此在影像任務（如手寫數字辨識）中，能更快學習、且具有更好的泛化能力。

這正是為什麼在 MNIST 等視覺任務中，  
CNN 不僅準確率較高，其 embedding 結構也更清晰、穩定。
