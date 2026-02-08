# MNIST：MLP vs CNN 比較筆記

同一 MNIST、相同 epoch 數下，記錄兩種模型的 val accuracy，體會 CNN 的 inductive bias（局部性、平移不變性）。

## 實驗設定（建議固定以便比較）

- batch_size=64, epochs=10, lr=1e-3, optimizer=Adam
- 資料：MNIST train/val split（例如 val_ratio=0.1）

## 結果紀錄（請跑完兩支腳本後填寫）

| 模型 | Epoch 10 Val Acc | 備註 |
|------|------------------|------|
| MLP  | (待填)           | 784→256→128→10 |
| CNN  | (待填)           | 3 層 conv+pool + FC |

## 小結

- CNN 在影像上通常用較少參數即可達到較高準確度，且收斂較穩。
- Phase 3 可在此基礎上調參（lr、dropout、augmentation）把準確度 tune 到 >98%。
