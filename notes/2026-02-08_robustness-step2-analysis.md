# Robustness STEP 2：結果說明與分析

## 「corruption clean」是什麼意思？

**clean** 代表**沒有做任何幾何破壞**的那 64 張圖。

- 在 STEP 1 裡，我們用**同一組 64 個 MNIST test 索引**產生了：
  - **clean**：這 64 張原圖（只做 normalize，沒有平移／旋轉／縮放等）
  - **translation_s1/s2/s3**、**rotation_s1/s2/s3**、… 等 18 組：同一批圖各自套上不同破壞
- 在 STEP 2 的表格裡，**clean** 那一行 = 三個模型在這 64 張「原圖」上的表現，當作**基準**。
- 其他每一行（例如 translation s1、perspective s3）的 **Drop %**、**Loss Increase** 都是**相對於該模型在 clean 上的結果**來算的。

所以：**clean = 同 64 張圖、無破壞的 baseline**；後面所有破壞類型都是在跟這個 baseline 比。

---

## 三模型整體誰比較穩？

- **ResNet 最穩**：在 clean 上 64/64 全對，多數破壞下 accuracy 仍很高；只有 **translation s3**（約 79.7%）、**perspective s3**（約 67.2%）掉比較多，其餘大多在 95% 以上甚至滿分。
- **CNN 次之**：clean 上 63/64；**translation**、**perspective** 加強時會明顯掉，例如 perspective s3 約 48.4%，translation s3 約 70.3%；其餘破壞多數還在 90% 以上。
- **MLP 最不穩**：clean 上 64/64，但一加破壞就掉很快。**Translation s3** 約 32.8%、**perspective s3** 約 26.6%，等於每 3 張就錯 1 張以上；translation s1 就只剩約 84.4%。

結論：**對幾何破壞的容忍度是 ResNet > CNN > MLP**，符合「有局部結構、殘差」的架構在這種干擾下較穩的直覺。

---

## 哪一種破壞最傷？

- **Perspective（透視）** 對三個模型殺傷力都最大：強度一高（s2、s3），accuracy 都明顯掉，MLP 掉最慘（s3 約 26.6%），CNN 約 48.4%，ResNet 約 67.2%。
- **Translation（平移）** 次之：MLP 從 s1 就開始掉，s3 約 32.8%；CNN、ResNet 到 s3 才掉到約 70%、80%。
- **Rotation、Shear、Scale**：對 ResNet 幾乎沒影響（多數仍接近 100%）；對 CNN 影響也不大；MLP 在 rotation、scale 加強時會慢慢掉。
- **Elastic（彈性變形）**：ResNet 仍很穩（s3 約 95.3%）；CNN、MLP 在 s2、s3 會掉到約 78–92%。

所以：**最難的是 perspective，其次是 translation**；rotation／shear／scale 對有卷積的模型相對友善。

---

## 小結（一句話）

- **clean** = 同一批 64 張圖、**沒做任何幾何破壞**的 baseline；所有「Drop %」都是相對這個 baseline。
- **ResNet 最耐幾何破壞，MLP 最不耐**；**透視與平移**最容易讓準確率掉，**旋轉／剪切／縮放**對 CNN／ResNet 影響較小。
