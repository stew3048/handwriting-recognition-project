# 🧠 深度學習學習筆記（MNIST × MLP / CNN / ResNet）

> 本筆記整理了一整天對 **MLP、CNN、ResNet、Embedding、Residual、BatchNorm** 的深入理解，  
> 目標不是記 API，而是**真的看懂模型在幹嘛、為什麼這樣設計、結果怎麼解讀**。

---

## 🔑 一定要先記住的關鍵名詞（超重要）

### 🔹 Embedding
- **模型對輸入資料的內部表示**
- 通常是一個向量（vector）
- 在影像分類中：
  - 指的是「分類器（FC）之前、已經壓掉空間資訊」的那一層輸出
- 用途：
  - 畫 t-SNE / PCA
  - 看語意結構
  - 分析 decision boundary

---

### 🔹 FC（Fully Connected Layer，全連結層）
- 數學形式：`y = Wx + b`
- 特性：
  - 每一個輸入維度都影響每一個輸出維度
  - **不保留任何空間結構**
- 角色：
  - 在「已經整理好的向量空間」中做決策（分類 / 回歸）
- 關鍵觀念：
  - **FC 是決策層，不是理解層**

---

### 🔹 Embedding 投影的 2D 圖（t-SNE / PCA）
- 是將 **高維 embedding 投影到 2D 空間的視覺化**
- 用來看：
  - 類別是否形成 cluster
  - 群集是否緊密
  - 錯誤樣本是否集中在邊界
- 注意：
  - t-SNE 的「全局距離」不可靠
  - 只適合觀察 **局部結構**

---

## 🏗️ 三種方法的架構與特性

### 1️⃣ MLP（Multi-Layer Perceptron）

#### 架構
```
Image (28×28)
 ↓ flatten
784-d vector
 ↓ FC + ReLU
 ↓ FC + ReLU
 ↓ FC
logits
```

#### 特性
- 沒有空間結構概念
- 對位置、傾斜非常敏感
- 表達能力高，但 inductive bias 弱
- 需要大量資料，否則容易 overfitting

#### Embedding 特徵
- cluster 鬆散、拉長
- 容易出現結構性錯誤（位置偏移就錯）

---

### 2️⃣ CNN（Convolutional Neural Network）

#### 架構
```
Image
 ↓ stem (Conv + BN + ReLU)
Low-level feature maps
 ↓ Conv blocks
High-level feature maps
 ↓ Global Pool / Flatten
Embedding
 ↓ FC
logits
```

#### 核心 inductive bias
- 局部性（locality）
- 權重共享（weight sharing）
- 近似平移不變性

#### Embedding 特徵
- 群集明顯、較緊
- 錯誤多為語意相近型
- 比 MLP 穩定很多

---

### 3️⃣ ResNet（Residual Network）

#### 架構
```
Image
 ↓ stem
 ↓ Residual Block × N
 ↓ Global Pool
Embedding
 ↓ FC
logits
```

#### BasicBlock（核心模組）
```
x ─────────────┐
 ↓              │
Conv → BN → ReLU│
 ↓              │
Conv → BN       │
 ↓              │
 +─────────────┘
 ↓
ReLU → output
```

#### Residual 的本質
- 數學形式：`output = x + F(x)`
- 每一層只學「微調」，而不是重來
- 限制每層對 embedding 空間的變形幅度

#### Embedding 特徵
- cluster 更圓、更均勻
- 邊界平滑、outlier 少
- decision boundary 穩定

---

## 🧩 Stem / BatchNorm / Residual 的幾何影響

### 🔹 Stem（Conv + ReLU）
- 功能：把「像素座標系」轉成「視覺特徵座標系」
- 視覺特徵 = 邊緣、方向、彎曲、局部結構
- 還不是 embedding（仍保有 H×W）

📐 幾何效果：
- 對位置不那麼敏感
- 同一 pattern 在不同位置 → 表示更接近

---

### 🔹 BatchNorm
- 對每個 channel 做標準化（mean=0, var=1）
- 再學習 scale（γ）與 shift（β）

📐 幾何效果：
- 把被拉爆的座標軸壓回來
- 讓 embedding 群集更圓、更集中
- train / val loss 更貼近

---

### 🔹 Residual
- 限制每一層對空間的彎折幅度
- 讓 embedding 流形變形是「連續、漸進的」

📐 幾何效果：
- 邊界平滑
- outlier 不易被甩飛
- 泛化更穩定

---

## 📊 MNIST 實驗結果分析總結

### MLP
- accuracy 尚可
- embedding 扭曲、鬆散
- worst cases 多為結構性錯誤

### CNN
- accuracy 提升
- embedding 群集清楚
- 錯誤多為語意模糊型

### ResNet
- accuracy 小幅提升
- **embedding 最穩定**
- worst cases 全部低信心、人類也會猶豫
- residual 的幾何效果「真的看得出來」

---

## 🔍 如何在 embedding 圖上看出 residual 的痕跡？

1. **Cluster 外形是否圓滑**
2. **是否幾乎沒有孤立 outlier**
3. **worst cases 是否都在 cluster 邊界**
4. **群集之間不是硬推開，而是自然分離**

---

## 🧠 今天反覆確認、一定要記住的觀念

- FC 是決策層，不是理解層
- Embedding ≠ feature map（空間壓掉才是）
- 每一層都可視為 representation，但只有最後一層適合當 embedding
- train / val loss 貼近 → 表示空間泛化一致
- CNN / ResNet 的強，不只是 accuracy，而是 **representation 的品質**

---

## ✨ 最後一句總結

> 這一整天學到的不是模型名字，  
> 而是「模型如何在內部組織與理解世界」。

當你能用 embedding、幾何、錯誤型態來解釋模型行為時，  
你就已經站在 **representation learning** 的那一側了。
