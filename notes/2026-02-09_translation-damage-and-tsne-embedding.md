# 為何 Translation 破壞性大？與 t-SNE Embedding 分析

## 一、為何 Translation（平移）造成的破壞性這麼大？

在我們的 7 情境裡，**準確率掉最多的是 Perspective（透視）**，**其次就是 Translation（平移）**。兩者都對三模型造成明顯傷害，以下分開說明「為什麼平移特別傷」。

### 1. 從「輸入怎麼被看到」來想

- **MLP（全連接）**  
  - 輸入是 **flatten 的 784 維**，每個維度對應**固定像素位置**。  
  - 訓練時學到的是「某個數字在**某個位置**長什麼樣子」。  
  - **平移** = 同一個數字整塊移到別格，等於**同一類別在輸入空間裡被搬到完全不同的維度**，模型幾乎像看到新分佈，所以一平移就崩很快（S3 只剩約 32.8%）。

- **CNN**  
  - 卷積有**局部性**與一定**平移不變性**，但：  
    - 我們的圖只有 28×28，receptive field 覆蓋整張圖很快，**大平移**會讓數字貼邊、被裁掉、或跑到模型「沒學好的邊界區域」。  
    - 訓練資料（MNIST）數字多半在**畫面中央**，平移後分佈偏離訓練分佈，所以 S3 掉到約 70%。

- **ResNet**  
  - 同樣是卷積＋殘差，但容量與層次更多，對**中等平移**還能撐住；**大平移**仍會讓準確率掉（S3 約 79.7%），只是比 MLP/CNN 好。

### 2. 小結：為什麼「平移」特別傷？

- **位置敏感**：模型（尤其 MLP）隱式學到「數字出現在哪裡」，平移直接破壞這個假設。  
- **邊界與裁切**：28×28 很小，平移一大就容易**出框、缺角**，資訊丟失。  
- **訓練／測試分佈不一致**：MNIST 原圖數字多置中，平移後測試分佈與訓練分佈偏離，泛化難。

**補充**：**Perspective（透視）** 更慘，因為除了「位置變」還加上**非線性變形、局部縮放**，結構扭曲更大，所以三模型在 perspective S3 的準確率都比 translation S3 更低。

---

## 二、t-SNE 組合圖與 Embedding 分析

已將 **clean + 6 種幾何破壞 S3** 下、**三模型（MLP / CNN / ResNet）** 的 t-SNE 2D 投影圖組合成一張圖，方便比較同一情境下不同模型的 embedding 表現。

- **檔案位置**：`outputs/robustness/tsne_combined_s3.png`  
- **版面**：7 列（Clean, Translation S3, Rotation S3, Scale S3, Shear S3, Perspective S3, Elastic S3）× 3 欄（MLP, CNN, ResNet），共 21 張縮圖。  
- **每張子圖**：同一批 64 筆的 **最後一層 embedding** 做 t-SNE 降成 2D，點依**真實標籤（0–9）**上色，紅色星為 **Worst 10**（該情境下 loss 最高的 10 筆）。

### 如何解讀 t-SNE

- **類別分群越清楚**（同一顏色聚在一起、不同顏色分開）→ 該模型在該情境下 embedding 仍保有較好的**類別可分性**，通常對應較高準確率。  
- **點混在一起、紅星散落各處**→ 該情境下 embedding 混亂，錯誤多、準確率低。  
- **紅星多落在類別邊界或錯類**→ 模型在邊界樣本上信心不足或判錯，與「Worst 10」定義一致。

### 從組合圖看 Embedding 的幾點觀察

1. **Clean**  
   - 三模型多數都能把 10 類大致分開（依資料量 64 點、類別數 10，本來就會有些重疊）。  
   - ResNet / CNN 通常分群較緊；MLP 在 clean 也還不錯，對應 clean 上 100% / 98.44% / 100% 的表現。

2. **Translation S3**  
   - **MLP**：點常混在一起，紅星多，對應 32.81% 準確率；embedding 幾乎失去類別結構。  
   - **CNN**：分群變模糊但仍有結構，對應約 70%。  
   - **ResNet**：仍能維持一定分群，對應約 79.7%，最穩。

3. **Perspective S3**  
   - 三模型 embedding 都明顯變亂，MLP 最散、CNN 次之、ResNet 相對仍有一點結構，與準確率 26.56% / 48.44% / 67.19% 一致。  
   - 可視為「透視破壞」同時打亂位置與形狀，對 embedding 的傷害最大。

4. **Rotation / Scale / Shear S3**  
   - ResNet、CNN 多數子圖仍維持不錯分群；MLP 在 rotation/scale 會稍亂，shear 尚可。  
   - 與表格中這三種破壞下 ResNet/CNN 仍高準確率、MLP 略降相符。

5. **Elastic S3**  
   - ResNet 分群仍清楚（約 95.3%）；CNN、MLP 有部分混在一起，對應 90.63% / 78.13%。

### 一句話總結 Embedding

- **Clean**：三模型 embedding 都能區分 10 類。  
- **Translation / Perspective S3**：對 MLP 傷害最大（embedding 混亂、紅星多）；CNN、ResNet 仍保有一定可分性，ResNet 最佳。  
- **Rotation / Scale / Shear**：對 CNN/ResNet 的 embedding 影響小；**Elastic** 次之。  
- 整體上，t-SNE 的「分群品質」與各情境的 **accuracy** 一致：越穩的模型與情境，點越按類別聚攏；越不穩的，點越混、Worst 10 越散。

---

**組合圖路徑**：`outputs/robustness/tsne_combined_s3.png`  
**產生腳本**：`scripts/create_tsne_combined.py`
