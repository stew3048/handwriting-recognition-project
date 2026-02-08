# 2026-02-08：MLP、CNN 與 ResNet 跑同一份 Test Data 分析

**目的**：用同一份 MNIST 官方 test set（10,000 張）評估 MLP、CNN 與 ResNet，比較三模型整體指標並分析差異。

---

## 一、Test 設定

- **資料**：MNIST 官方 test set，10,000 張，三模型使用**完全相同**的 test data。
- **MLP run**：`20260208_144400_mnist_mlp`，checkpoint `best_mnist_mlp.pt`
- **CNN run**：`20260208_155516_mnist_cnn`，checkpoint `best_mnist_cnn.pt`
- **ResNet run**：`20260208_204222_mnist_resnet`，checkpoint `best_mnist_resnet.pt`（小 ResNet，28×28 灰階、10 類；訓練腳本 `train_mnist_resnet.py`，評估時 `--model resnet`）

---

## 二、三模型整體指標（同一份 Test）

| 指標 | MLP | CNN | ResNet |
|------|-----|-----|--------|
| **Test Loss** | 0.0894 | 0.0288 | 0.0246 |
| **Test Accuracy** | 97.83% | 99.13% | **99.26%** |
| 預測正確 | 9,783 | 9,913 | 9,926 |
| 預測錯誤 | 217 | 87 | 74 |

**ResNet**：評估已完成，產出見下方產出位置。若 run 僅部分完成需續訓或重跑評估，可執行 `python scripts/train_mnist_resnet.py --epochs 10 --save_best` 續訓，再執行 `python scripts/evaluate_mnist.py --checkpoint outputs/runs/<run>/best_mnist_resnet.pt --model resnet --output_dir outputs/runs/<run>`，將結果填入上表與下方產出位置。

---

## 三、分析

1. **正確率**：CNN 比 MLP 高約 1.3 個百分點，錯誤數從 217 降到 87，約少 **60%** 錯誤。
2. **Loss**：CNN 的 test loss 明顯較低（0.0288 vs 0.0894），代表在相同 test set 上預測更穩、信心較高。
3. **原因對照**：CNN 具局部性與平移不變性，在影像上 sample efficiency 較好；MLP 無空間結構假設，同一份 test 上更容易在邊界樣本或幾何變形上出錯。
4. **CNN 的錯誤型態**：worst 10 多為 4/9、3/5/8、7/2 等視覺相似或書寫模糊的樣本，較少「位置或幾何變形」造成的結構性錯誤，符合 inductive bias 與影像任務匹配的預期。
5. **ResNet 與三模型比較**：ResNet 表現最佳（Test acc 99.26% > CNN 99.13% > MLP 97.83%，Test loss 最低 0.0246）；比 CNN 錯誤數更少（74 vs 87），預測更穩。ResNet 的錯誤型態：worst 10 仍以視覺相似與書寫模糊為主，但 **2→8 混淆特別集中**（見下方 6.3）。

---

## 四、總結

相較於 MLP，CNN 不僅在整體 accuracy 上略有提升，更重要的是其學到的 embedding 結構更為緊密且具語意一致性（組內變異小，組間切得更乾淨）；CNN 的錯誤案例多屬於人類也難以判斷的模糊樣本，而非由位置或幾何變形造成的結構性錯誤，顯示其 inductive bias 與影像任務高度匹配。ResNet 在 CNN 基礎上藉 residual 與 BatchNorm 再略為提升，三者的 worst case 多屬人類也難判的模糊樣本。

---

## 五、MLP vs CNN 筆記（來源：`C:\Users\yiching\handwriting-recognition-project\node`）

### 總結

因為 MLP 是全連結神經網路，它不像 CNN 內建局部性與平移等 inductive bias，所以本身沒有明確的空間結構假設；
相對地，CNN 透過局部卷積與權重共享，自然保留了影像的空間關係。
MLP 的表達能力其實非常彈性，但正因為缺乏這些結構性假設，它通常需要更多資料才能學到穩定、可泛化的表示，否則容易 overfitting；
在這個 MNIST 的例子中，MLP 不容易快速學到「某個視覺 pattern 在不同位置出現仍屬於同一概念」，因此在效率與穩定性上不如 CNN。

### 自問自答 QA

- **Q：那為什麼不用 MLP 就好？**
- **A：** 在影像這種高度結構化的資料上，適當的 inductive bias 其實比模型彈性更重要，因為它能大幅降低 sample complexity。

---

## 六、runs 內 .md 統整（Worst 10 分析）

三 run 目錄內各有 **worst10_analysis.md**，內容統整如下。

### 6.1 CNN run（20260208_155516_mnist_cnn）— 本 run 的 Worst 10 明細

| Rank | GT | Pred | Confidence |
|------|-----|------|------------|
| 1 | 8 | 9 | 0.3551 |
| 2 | 9 | 4 | 0.3951 |
| 3 | 7 | 2 | 0.4205 |
| 4 | 8 | 2 | 0.4721 |
| 5 | 9 | 7 | 0.4825 |
| 6 | 9 | 5 | 0.4909 |
| 7 | 5 | 3 | 0.4946 |
| 8 | 2 | 3 | 0.5142 |
| 9 | 5 | 9 | 0.5193 |
| 10 | 4 | 6 | 0.5247 |

**混淆對**：8→9、9→4、7→2、8→2、9→7、9→5、5→3、2→3、5→9、4→6（各 1 筆）。多屬視覺相似（4/9、3/5/8、7/2、4/6 等）或書寫模糊邊界樣本；CNN 仍可能在此類樣本上低 confidence 出錯。

### 6.2 MLP run（20260208_144400_mnist_mlp）— 通用 Worst 10 說明

該 run 的 worst10_analysis.md 為**通用版**：說明「Top 10 worst」定義、MNIST 常見數字混淆對（4↔9、3↔5↔8、7↔1、2↔7、5↔6、0↔6/8）、為何會進 worst 10（ambiguous writing、MLP 無局部結構、normalization 與邊界），以及如何對應 run（看 top10_worst_cases.png、對照 GT/Pred）。小結：視覺相似、書寫模糊、模型限制（MLP 對細微筆畫較不敏感）；改善方向為 data augmentation、換 CNN、或 dropout。

### 6.3 ResNet run（20260208_204222_mnist_resnet）— 本 run 的 Worst 10 明細

| Rank | GT | Pred | Confidence |
|------|-----|------|------------|
| 1 | 0 | 8 | 0.3195 |
| 2 | 2 | 8 | 0.3672 |
| 3 | 4 | 6 | 0.3756 |
| 4 | 2 | 3 | 0.3914 |
| 5 | 5 | 0 | 0.3993 |
| 6 | 9 | 4 | 0.3997 |
| 7 | 1 | 3 | 0.4212 |
| 8 | 2 | 8 | 0.4290 |
| 9 | 2 | 8 | 0.4649 |
| 10 | 9 | 5 | 0.4725 |

**混淆對統計 (GT → Pred)**：**2→8**（3 筆）、0→8、4→6、2→3、5→0、9→4、1→3、9→5（各 1 筆）。ResNet 的 worst 10 較集中在 **2↔8**，其餘為常見易混對（4/6、4/9、2/3、9/5 等）；視覺相似與書寫模糊仍為主要原因。

### 6.4 對照

- **CNN**：有本 run 的 10 筆明細表與混淆對統計，錯誤多為人類也難判的模糊樣本。
- **MLP**：為通用原因說明，強調無局部結構導致邊界樣本易進 worst 10；實務上可對照該 run 的 top10_worst_cases.png 解讀。
- **ResNet**：有本 run 的 10 筆明細與混淆對，**2→8 佔 3 筆**，其餘分散；整體錯誤最少。

---

**產出位置**：

- **CNN run** `outputs/runs/20260208_155516_mnist_cnn/`：eval_results.txt、embedding_pca.png、embedding_tsne.png、top10_worst_cases.png、**worst10_analysis.md**、train_curves.png、experiments_cnn.csv。
- **MLP run** `outputs/runs/20260208_144400_mnist_mlp/`：同上結構，含 **worst10_analysis.md**。
- **ResNet run** `outputs/runs/20260208_204222_mnist_resnet/`：已含評估產出（eval_results.txt、embedding_pca.png、embedding_tsne.png、top10_worst_cases.png、worst10_analysis.md）；訓練產出 experiments_resnet.csv、train_curves.png、best_mnist_resnet.pt、last_mnist_resnet.pt。
- 整體對照表：`outputs/cnn_vs_mlp_test_metrics.md`（可擴充為 MLP / CNN / ResNet 三欄）。
