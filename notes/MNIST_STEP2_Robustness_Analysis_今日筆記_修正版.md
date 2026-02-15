# 📘 MNIST Robustness Study — STEP 2 今日學習筆記（修正版）
（Corrupted Data × Representation Analysis）

> 本文件記錄 **STEP 2：Robustness Evaluation** 的完整觀察與分析，  
> 目標是理解：**當世界被刻意破壞時，不同模型的表示空間（embedding）如何退化。**  
>  
> ⚠️ 本版本已修正先前結論：  
> **最具破壞性的幾何變形是 Perspective（透視變形），Translation 為次嚴重。**

---

## 🎯 今日目的回顧

今天我們做的事情不是「再把模型訓練得更準」，而是：

> **固定模型，刻意破壞輸入資料，觀察 accuracy、錯誤型態，以及 embedding 幾何結構的變化。**

這一步的核心在於 **驗證 inductive bias 是否真的存在，而不是只在乾淨資料上碰巧有效。**

---

## 🧪 實驗設定摘要

- Dataset：MNIST test set  
- Corruption：7 種情境（Clean + 6 種幾何變形）  
  - Translation / Rotation / Scale / Shear / **Perspective** / Elastic  
- Severity：S3（worst case）  
- 評估方式：
  - 固定 random seed
  - 從 test set 抽樣 **同一組 64 張影像**
  - 同一組資料分別丟給 **MLP / CNN / ResNet**
- 模型：
  - 不 retrain
  - 不加 augmentation

---

## 📊 Robustness Summary Table（關鍵修正後解讀）

整體趨勢仍然成立：

- **ResNet 在所有 corruption 下最穩定**
- CNN 次之
- MLP 最脆弱

但在「破壞強度排序」上，正確結論為：

### ❗ 破壞力排序（由強到弱，S3）
1. **Perspective（最嚴重）**
2. Translation
3. Elastic
4. Scale
5. Rotation ≈ Shear

---

## ❓ 為什麼 Perspective 是最致命的幾何變形？（關鍵理解）

### Perspective 變形本質上破壞了什麼？

Perspective 並不是單純的：
- 位移（translation）
- 旋轉（rotation）
- 等比例縮放（scale）

而是同時改變了：
- **局部形狀比例**
- **筆畫之間的相對幾何關係**
- **整體投影結構（projective geometry）**

👉 對模型來說，這等於：
>「同一個數字，被投影成『另一種結構』。」

---

### 各模型在 Perspective 下的反應

#### 🔴 MLP
- 幾乎無法維持任何類別流形
- embedding 完全混雜
- 表現接近隨機

#### 🟡 CNN
- 局部特徵仍可辨（邊緣、短線段）
- 但整體結構關係被破壞
- cluster 大量重疊

#### 🟢 ResNet
- 仍能保留部分群集
- worst cases 集中於邊界
- 但相較其他 corruption，結構退化最明顯

👉 **Perspective 超出了「平移不變性」或「局部性」可保護的範圍。**

---

## 🧠 全部幾何破壞的 Embedding 總覽解讀（對照圖）

（以下觀察來自 7 Scenarios × 3 Models 的 t-SNE grid）

### 1️⃣ Clean
- 三種模型皆可形成某種程度的 cluster
- ResNet 最平滑、CNN 次之、MLP 最鬆散

---

### 2️⃣ Translation
- MLP：流形破碎（位置即語意）
- CNN：群集拉長、邊界扭曲
- ResNet：**整體位移但結構仍在**

👉 Translation 是最能暴露「是否具備空間 inductive bias」的測試。

---

### 3️⃣ Rotation / Shear
- 三種模型皆相對穩定
- cluster 變形但不崩壞
- 顯示 MNIST 本身對旋轉仍有一定容忍度

---

### 4️⃣ Scale
- MLP 明顯退化
- CNN / ResNet 能部分吸收尺度變化
- 表示多層卷積能學到某種 scale tolerance

---

### 5️⃣ **Perspective（最關鍵）**
- 三種模型 embedding 結構同時嚴重退化
- 群集重疊、邊界消失
- worst cases 大量出現在「群集內部」而非邊界

👉 這代表 **模型已失去穩定 decision boundary**。

---

### 6️⃣ Elastic
- 對局部筆畫破壞強
- ResNet 明顯優於 CNN / MLP
- 顯示 residual 對「局部連續變形」特別有幫助

---

## 📌 今日最重要的修正後結論

1. **Perspective 是最能擊穿所有 inductive bias 的幾何變形**
2. Translation 是第二致命，但主要測試「空間一致性」
3. CNN / ResNet 的優勢，主要來自對「局部結構」與「連續變形」的保護
4. 當破壞涉及 **整體幾何關係改寫（projective change）**，所有模型都會顯著退化
5. Robustness 的本質，是 **表示空間是否仍保有可分離的語意流形**

---

## 🧭 為什麼數字（MNIST）可以在這裡收尾？

因為你已經：
- 驗證了模型差異不是巧合
- 用 embedding 幾何解釋 performance
- 找到哪些破壞「超出 inductive bias 能力範圍」

👉 MNIST 已經完成它作為 **toy world** 的任務了。

---

## 🚀 下一步方向（明天）

依照老闆建議：

> **進入「數字 + 英文混合」的辨識任務，提升語意與結構複雜度。**

這會帶來：
- 更多類別（0–9 + A–Z / a–z）
- 更高的筆畫結構相似性
- 更接近真實 OCR 問題

明天開始，我們將：
- 重新定義資料集與 label space
- 重新檢視 embedding 是否仍能形成穩定語意結構

---

## 🌙 收尾一句

> 今天我們學到的不只是「哪個模型比較強」，  
> 而是 **哪些世界變化，會真正改寫模型的理解方式。**

MNIST 的故事到這裡，已經完整了。
