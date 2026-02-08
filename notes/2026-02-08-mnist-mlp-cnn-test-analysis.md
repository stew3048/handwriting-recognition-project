# 2026-02-08：MLP 與 CNN 跑同一份 Test Data 分析

**目的**：用同一份 MNIST 官方 test set（10,000 張）評估 MLP 與 CNN，比較兩者整體指標並分析差異。

---

## 一、Test 設定

- **資料**：MNIST 官方 test set，10,000 張，兩模型使用**完全相同**的 test data。
- **MLP run**：`20260208_144400_mnist_mlp`，checkpoint `best_mnist_mlp.pt`
- **CNN run**：`20260208_155516_mnist_cnn`，checkpoint `best_mnist_cnn.pt`

---

## 二、兩份資料：整體指標

| 指標 | CNN | MLP | 差異 (CNN − MLP) |
|------|-----|-----|------------------|
| **Test Loss** | 0.0288 | 0.0894 | −0.0606（CNN 較低） |
| **Test Accuracy** | **99.13%** | 97.83% | **+1.30%**（CNN 較高） |
| 預測正確 | 9,913 | 9,783 | CNN 多對 130 張 |
| 預測錯誤 | 87 | 217 | CNN 少錯 130 張 |

---

## 三、分析

1. **正確率**：CNN 比 MLP 高約 1.3 個百分點，錯誤數從 217 降到 87，約少 **60%** 錯誤。
2. **Loss**：CNN 的 test loss 明顯較低（0.0288 vs 0.0894），代表在相同 test set 上預測更穩、信心較高。
3. **原因對照**：CNN 具局部性與平移不變性，在影像上 sample efficiency 較好；MLP 無空間結構假設，同一份 test 上更容易在邊界樣本或幾何變形上出錯。
4. **CNN 的錯誤型態**：worst 10 多為 4/9、3/5/8、7/2 等視覺相似或書寫模糊的樣本，較少「位置或幾何變形」造成的結構性錯誤，符合 inductive bias 與影像任務匹配的預期。

---

## 四、總結

相較於 MLP，CNN 不僅在整體 accuracy 上略有提升，更重要的是其學到的 embedding 結構更為緊密且具語意一致性（組內變異小，組間切得更乾淨）；CNN 的錯誤案例多屬於人類也難以判斷的模糊樣本，而非由位置或幾何變形造成的結構性錯誤，顯示其 inductive bias 與影像任務高度匹配。

---

**產出位置**：CNN run 目錄 `outputs/runs/20260208_155516_mnist_cnn/` 含 eval_results.txt、embedding_pca.png、embedding_tsne.png、top10_worst_cases.png、worst10_analysis.md；整體對照表見 `outputs/cnn_vs_mlp_test_metrics.md`。
