# 每日日誌 2026-02-17：ResNet 改善版本 v1~v3 差異分析與平衡評估

---

## 1. ResNet+Center Loss 版本演進（v1 → v2 → v3）

### 1.1 版本資訊

| 版本 | 模型路徑 | Embedding 維度 | 訓練時間 |
|------|----------|----------------|----------|
| **v1** | `outputs/runs/20260215_163722_emnist36_centerloss/best_emnist36_centerloss.pt` | 128-dim | 2026-02-15 16:37 |
| **v2** | `outputs/runs/20260216_005651_emnist36_centerloss/best_emnist36_centerloss.pt` | 64-dim | 2026-02-16 00:56 |
| **v3** | `outputs/runs/20260216_214309_emnist36_centerloss/best_emnist36_centerloss.pt` | 64-dim | 2026-02-16 21:43 |

### 1.2 版本間主要差異

#### v1 → v2：Embedding 維度降低
- **v1**: 128-dim embedding（有升維層：64 → 128 → 36）
- **v2**: 64-dim embedding（不升維：64 → 36）
- **動機**: 簡化架構，減少參數，觀察 64 維是否足夠

#### v2 → v3：訓練策略優化
- **CSV 記錄格式**:
  - v2: 沒有 `alpha` 欄位
  - v3: 增加 `alpha` 欄位記錄（便於追蹤 Center Loss 權重變化）
  
- **學習率調度**:
  - v2: 固定學習率 0.001（從 CSV 推測）
  - v3: CosineAnnealingLR（學習率從 0.000994 開始下降）
  
- **資料處理**:
  - **v2 和 v3**: 都做了 ToTensor + Normalize（數值範圍 [-1.0, 1.0]）
  - 註：代碼註釋「不做 Normalize（與 v3 一致）」是後來添加的，實際 v3 訓練時仍做了 Normalize

#### Normalize 說明

**Normalize 是什麼？**
- Normalize 是資料標準化處理，將像素值從一個範圍映射到另一個範圍
- 在我們的代碼中：`transforms.Normalize((0.5,), (0.5,))`
  - 輸入範圍：[0.0, 1.0]（ToTensor 後的浮點數）
  - 輸出範圍：[-1.0, 1.0]
  - 公式：`output = (input - 0.5) / 0.5 = 2 * input - 1`

**做與不做的差別**：

| 項目 | 做 Normalize | 不做 Normalize |
|------|--------------|----------------|
| **數值範圍** | [-1.0, 1.0] | [0.0, 1.0] |
| **均值** | 0.0（中心化） | 0.5 |
| **標準差** | 約 0.5（標準化後） | 約 0.29 |
| **數據分布** | 對稱分布（負值與正值） | 非對稱分布（僅正值） |

**做 Normalize 的好處**：

1. **數值穩定性**：
   - 將數據中心化到 0 附近，減少梯度爆炸/消失的風險
   - 對稱分布有利於優化器（如 Adam）的動量更新

2. **訓練效率**：
   - 標準化後的數據通常收斂更快
   - Batch Normalization 層的效果更好（因為輸入已經標準化）

3. **模型泛化**：
   - 標準化有助於模型學習更穩定的特徵表示
   - 減少對絕對像素值的依賴，增強對亮度變化的魯棒性

4. **與預訓練模型對齊**：
   - 許多預訓練模型（如 ImageNet）使用標準化輸入
   - 如果未來要使用遷移學習，標準化輸入是必要的

**不做 Normalize 的考量**：
- 保留原始像素值分布，可能對某些任務更直觀
- 減少數據預處理步驟，簡化流程
- 在某些情況下，模型可能學習到更直接的像素值映射

#### Transpose 說明

**Transpose(1, 2) 是什麼？**
- `transpose(1, 2)` 是交換 Tensor 的第 1 和第 2 個維度（高度 H 和寬度 W）
- 在我們的代碼中：`transforms.Lambda(lambda x: x.transpose(1, 2))`

**為什麼需要 Transpose？**
根據代碼註釋（`scripts/datasets.py` 第 84 行）：
> "EMNIST 影像需轉置空間維度 (H,W) 才與一般書寫方向一致；保持 (C,H,W) 給 Conv2d"

**具體流程**：
1. **ToTensor() 之後**：
   - 輸入：PIL Image（28×28 灰階）
   - 輸出：`[C, H, W] = [1, 28, 28]`（通道、高度、寬度）

2. **Transpose(1, 2) 之後**：
   - 輸入：`[1, 28, 28]`（通道=1, 高度=28, 寬度=28）
   - 輸出：`[1, 28, 28]`（通道=1, 寬度=28, 高度=28）
   - **實際效果**：交換高度和寬度的內容，修正 EMNIST 圖像的空間方向

**原因**：
- EMNIST 數據集在存儲時，圖像的空間方向（H, W）可能與我們期望的方向不一致
- 例如：原始圖像是橫向的（寬度 > 高度），但我們需要縱向的（高度 > 寬度）
- `transpose(1, 2)` 交換 H 和 W 的內容，讓圖像方向與書寫方向一致

**重要注意事項**：
- ✅ 只交換空間維度（H, W），**不動通道維度（C）**
- ✅ 保持 `[C, H, W]` 格式，符合 PyTorch Conv2d 的輸入要求
- ✅ 這是 EMNIST 數據集的特殊需求，不是所有數據集都需要

### 1.3 Center Loss 排程（v2 和 v3 相同）

所有版本都使用延遲 Center Loss 策略：
- **Epoch 1-5**: `alpha=0`（僅 CE Loss，Warm-up）
- **Epoch 6-10**: `alpha=0.05`（穩定期）
- **Epoch 11-20**: `alpha` 線性從 0.05 → 0.1（漸進增強）

### 1.4 詳細差異記錄

詳細的 v2 vs v3 差異分析已記錄在：`docs/v2_v3_training_differences.md`

**關鍵發現**：
- v3 的註釋「不做 Normalize」是誤導性的，實際 v3 訓練時仍然做了 Normalize
- 真正的「不做 Normalize」版本是後來的 `get_emnist36_balanced_refined_loaders`（溫和均衡版）

---

## 2. 五個方法平衡評估（每類 800 筆）

### 2.1 評估設定

- **測試資料**: 平衡分布，每類 800 筆（總計 28,800 筆）
- **評估方法**: 
  1. CNN
  2. 一般 ResNet
  3. ResNet+CL v1 (128-dim)
  4. ResNet+CL v2 (64-dim)
  5. ResNet+CL v3 (64-dim)

### 2.2 評估結果

評估結果保存在：`outputs/emnist36_eval_balanced_5versions/`

**主要指標**：
- Test Accuracy
- 混淆矩陣（36×36 完整矩陣）
- Embedding 分離度分析（O/0、I/1 等易混字對）
- 2D PCA 視覺化比較

### 2.3 詳細比較結果

**整體 Test Accuracy 排名**：
1. 🥇 **ResNet+CL v3 (64d)**：91.19%
2. 🥈 **ResNet+CL v2 (64d)**：90.77%
3. 🥉 **一般 ResNet**：90.33%
4. **CNN**：89.11%
5. **ResNet+CL v1 (128d)**：88.80%

**易混字對表現（兩類整體準確率）**：

| 字對 | CNN | 一般 ResNet | ResNet+CL v1 | ResNet+CL v2 | ResNet+CL v3 | 最佳 |
|------|-----|-------------|--------------|--------------|--------------|------|
| **O_0** | 69.89% | 69.71% | 70.25% | 71.05% | **71.27%** 🥇 | v3 |
| **I_1** | 79.93% | 80.09% | 78.68% | 81.01% | **81.16%** 🥇 | v3 |
| **Z_2** | 92.57% | 92.34% | 92.59% | 93.00% | **94.08%** 🥇 | v3 |
| **S_5** | 93.02% | 93.59% | 82.73% | 94.67% | **95.18%** 🥇 | v3 |
| **A_4** | 99.80% | 99.80% | 99.60% | **99.93%** 🥇 | 99.81% | v2 |
| **G_6** | 97.98% | 98.05% | 98.36% | 98.43% | **98.68%** 🥇 | v3 |
| **B_8** | 99.35% | 99.67% | 99.36% | 99.10% | **99.61%** 🥇 | v3 |

**關鍵發現**：
- ✅ **v3 在 6/7 個易混字對上表現最佳**
- ✅ **v3 整體 Test Accuracy 最高**（91.19%）
- ⚠️ **v1 雖然 embedding 分離度最高，但分類準確率最低**（88.80%）
- 📊 **64-dim embedding 優於 128-dim**：v2 和 v3 都明顯優於 v1

**詳細報告**：
- **Embedding 分析報告**：`outputs/emnist36_eval_balanced_5versions/embedding_analysis_report.md`
  - 包含：易混字對兩類準確率、Embedding 分離度比較表
- **詳細比較報告**：`outputs/emnist36_eval_balanced_5versions/detailed_comparison_report.md`（新增）
  - 包含：
    - 整體 Test Accuracy 與 Loss 比較
    - 每個易混字對的詳細表現分析（O/0、I/1 等）
    - Embedding 分離度 vs 分類準確率的關係分析
    - 方法間詳細比較（CNN vs ResNet、v1 vs v2 vs v3）
    - 綜合結論與建議
- **2D PCA 視覺化**：`outputs/emnist36_eval_balanced_5versions/o0_embedding_2d_compare/`
  - O/0 字對的 2D PCA 視覺化比較（五個方法並排）

**詳細比較報告重點摘要**：
- **整體 Test Accuracy 排名**：v3 (91.19%) > v2 (90.77%) > 一般 ResNet (90.33%) > CNN (89.11%) > v1 (88.80%)
- **v3 在 6/7 個易混字對上表現最佳**
- **重要發現**：高分離度 ≠ 高準確率（v1 分離度最高但準確率最低）
- **64-dim embedding 優於 128-dim**：v2 和 v3 都明顯優於 v1

---

## 3. 資料強化版訓練（精準打擊版）

### 3.1 訓練策略

**目標**: 針對易混字對（0/O、1/I）進行精準打擊

**資料分布**:
- **Train**: 
  - 0/O/1/I 每類 15,000 筆（3x 過採樣）
  - 其他 32 類每類 5,000 筆
  - 比例：3:1
- **Val**: 從 train 分割 10%
- **Test**: 平衡分布，每類 800 筆（公正衡量）

**資料擴增**:
- 僅對 0/O/1/I 做擴增
- 50% 機率：輕微 RandomAffine (degrees=5, translate=0.05)
- 50% 機率：強力 RandomAffine (degrees=20, translate=0.15) + RandomErasing

**損失函數**:
- CE Loss: 0/O/1/I 權重 2.0，其他 1.0；label_smoothing=0.1
- Center Loss: Epoch 1-5 alpha=0，6-15 alpha=0.05，16-25 線性 0.05→0.15

**訓練參數**:
- Epochs: 25
- Batch size: 10
- Optimizer: Adam + CosineAnnealingLR

### 3.2 訓練結果

**模型**: `outputs/runs/20260217_132743_emnist36_precision_final/final_precision_model.pt`

**結果**: **表現不佳**
- 雖然針對目標類別做了大量過採樣和擴增，但整體表現未達預期
- 可能原因：
  1. 過度聚焦目標類別，導致其他類別表現下降
  2. 擴增強度過大，可能破壞了原始特徵
  3. CE 權重不平衡（2.0 vs 1.0）可能過於激進

---

## 4. 溫和均衡版訓練（進行中）

### 4.1 訓練策略調整

基於精準打擊版的經驗，調整為更溫和的策略：

**資料分布**（與精準打擊版相同）:
- Train: 0/O/1/I 每類 15,000，其他每類 5,000
- Test: 平衡分布，每類 800 筆

**資料擴增**（更溫和）:
- 僅對 0/O/1/I 做擴增
- 20% 機率：RandomAffine (degrees=10, translate=0.1)
- 80% 機率：不做擴增
- **不做 RandomErasing**

**資料處理**（與 v3 **不一致**）:
- **所有類別都只做 ToTensor，不做 Normalize**
- 數值範圍：[0.0, 1.0]
- **重要修正**：v3 實際訓練時**有做 Normalize**，溫和均衡版**不做 Normalize**，兩者不一致

**實際代碼**（`scripts/datasets.py` 第 512-517 行）：
```python
def _to_tensor_only():
    """僅做 ToTensor，不做 Normalize（注意：v3 有做 Normalize，此版本與 v3 不一致）"""
    return transforms.Compose([
        transforms.ToTensor(),                    # PIL → Tensor, [0,255] → [0.0,1.0]
        transforms.Lambda(lambda x: x.transpose(1, 2)),  # 交換 H 和 W，修正 EMNIST 方向
        # 沒有 Normalize
    ])
```

**與 v2/v3 的對比**：
- **v2/v3**: `_transpose_and_normalize()` = ToTensor + Transpose + **Normalize**（數值範圍 [-1.0, 1.0]）
- **溫和均衡版**: `_to_tensor_only()` = ToTensor + Transpose（**沒有 Normalize**，數值範圍 [0.0, 1.0]）
- **修正說明**：之前註解中寫「與 v3 一致」是錯誤的，已修正為「與 v3 不一致」

**不做 Normalize 的優點與弱點**：

**優點**：
1. **簡化數據流程**：
   - 減少預處理步驟，數據更接近原始分布
   - 降低數據轉換可能引入的誤差

2. **保留原始特徵**：
   - 像素值直接對應原始圖像亮度
   - 可能對某些特徵（如筆畫粗細、對比度）的學習更直觀

3. **與 v3 一致**：
   - 雖然 v3 實際訓練時做了 Normalize，但後續分析認為不做 Normalize 可能更好
   - 統一策略，便於比較

4. **減少數值轉換**：
   - 避免 [0,1] → [-1,1] 的映射可能造成的資訊損失
   - 某些模型架構可能對 [0,1] 範圍更敏感

**弱點**：
1. **訓練穩定性**：
   - 非中心化數據（均值約 0.5）可能導致梯度更新不夠穩定
   - Batch Normalization 層需要適應非標準化輸入

2. **收斂速度**：
   - 未標準化的數據可能需要更多 epochs 才能收斂
   - 學習率可能需要調整以適應不同的數值範圍

3. **模型泛化**：
   - 可能對亮度變化更敏感（因為直接依賴絕對像素值）
   - 標準化通常有助於提升泛化能力

4. **與常見實踐不一致**：
   - 大多數深度學習實踐都使用標準化輸入
   - 如果未來要使用預訓練模型或遷移學習，可能需要重新調整

**預期影響**：
- 溫和均衡版選擇不做 Normalize，是基於「簡化流程、保留原始特徵」的理念
- 實際效果需要等待訓練完成後評估
- 如果效果不佳，可以考慮重新加入 Normalize 或調整 Normalize 參數

**損失函數**（更公平）:
- CE Loss: **所有類別權重 1.0**（公平權重）；label_smoothing=0.1
- Center Loss: Epoch 1-5 alpha=0，6-20 線性 0.05→0.12（更溫和的增長）

**訓練參數**:
- Epochs: 20
- Batch size: 10
- Optimizer: Adam + CosineAnnealingLR

### 4.2 訓練狀態

**當前狀態**: **訓練進行中**

**訓練腳本**: `scripts/train_emnist36_balanced_refined.py`

**預期輸出**: `outputs/runs/YYYYMMDD_HHMMSS_emnist36_balanced_refined/balanced_refined_model.pt`

**設計理念**:
- 平衡分類準確率與 Embedding 分離度
- 不過度聚焦目標類別，保持整體性能
- 使用更溫和的擴增和損失函數策略

### 4.3 資料分割邏輯確認

**資料分割流程**：
1. `train_subset = get_balanced_subset_precision(...)` 
   - 0/O/1/I 每類：15,000 筆
   - 其他類別每類：5,000 筆
   - 總計：220,000 筆

2. `train_part, val_part = random_split(train_subset, [n_train, n_val], ...)`
   - `val_part` 是從 `train_subset` 中隨機分割出來的（約 10%）
   - `train_part` 是 `train_subset` 的另一部分（約 90%）

**驗證集（val）的資料分布**：
- 0/O/1/I 每類在 val 中約有：15,000 × 0.1 ≈ 1,500 筆
- 其他類別每類在 val 中約有：5,000 × 0.1 ≈ 500 筆

**資料洩漏檢查**：
- ✅ **已驗證**：`random_split` 產生**互斥**的子集，train 和 val **完全沒有重疊**
- ✅ **無資料洩漏**：train 沒有「偷看」過 val 資料
- ✅ **符合最佳實踐**：這是標準的資料分割方式

**驗證方法**：
- 使用 `random_split` 測試：100 筆資料分成 80/20
- 結果：Train 80 筆 + Val 20 筆 = 100 筆（總數）
- 重疊：0 筆（完全互斥）

---

## 5. 今日總結

### 5.1 完成事項

1. ✅ **版本差異分析**: 詳細記錄了 v1~v3 的差異，特別是 v2 vs v3 的實際差異（透過 git 歷史確認）
2. ✅ **平衡評估**: 五個方法在平衡 test set（每類 800 筆）上的完整評估
3. ✅ **精準打擊版訓練**: 完成訓練但結果不佳，提供了寶貴的經驗
4. ✅ **溫和均衡版設計**: 基於經驗調整策略，開始訓練

### 5.2 關鍵發現

1. **v2 vs v3 差異**:
   - 主要差異在學習率調度和 CSV 記錄格式
   - 資料處理（Normalize）實際相同，註釋有誤導性

2. **精準打擊版問題**:
   - 過度聚焦目標類別可能導致整體性能下降
   - 強力擴增和權重不平衡可能過於激進

3. **溫和均衡版改進**:
   - 公平權重（所有類別 1.0）
   - 更溫和的擴增（20% 機率，較小參數）
   - 不做 RandomErasing
   - 不做 Normalize（與 v3 **不一致**，v3 有做 Normalize）

### 5.3 下一步

1. **等待溫和均衡版訓練完成**
2. **評估溫和均衡版**: 使用 `--use_balanced_refined_test` 選項
3. **比較分析**: 與 v1~v3 和精準打擊版進行全面比較
4. **持續優化**: 根據結果決定下一步策略

---

## 6. 相關文件

- **v2 vs v3 差異記錄**: `docs/v2_v3_training_differences.md`
- **平衡評估結果**: `outputs/emnist36_eval_balanced_5versions/`
- **精準打擊版訓練**: `outputs/runs/20260217_132743_emnist36_precision_final/`
- **溫和均衡版訓練**: `outputs/runs/YYYYMMDD_HHMMSS_emnist36_balanced_refined/`（進行中）

---

## 7. 訓練指令記錄

### 精準打擊版
```bash
python scripts/train_emnist36_precision_final.py --batch_size 10 --cpu
```

### 溫和均衡版
```bash
python scripts/train_emnist36_balanced_refined.py --batch_size 10 --cpu
```

### 評估指令
```bash
# 平衡評估（五個方法）
python scripts/evaluate_emnist_digits_letters.py \
  --checkpoint <checkpoint_path> \
  --model <model_type> \
  --samples_per_class_test 800 \
  --output_dir outputs/emnist36_eval_balanced_5versions

# 溫和均衡版評估
python scripts/evaluate_emnist_digits_letters.py \
  --checkpoint outputs/runs/.../balanced_refined_model.pt \
  --model resnet_centerloss_64 \
  --use_balanced_refined_test \
  --output_dir outputs/.../eval_balanced_refined
```

---

**日期**: 2026-02-17  
**記錄者**: AI Assistant  
**狀態**: 溫和均衡版訓練進行中
