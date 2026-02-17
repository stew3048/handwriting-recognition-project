# ResNet+CL v2 vs v3 訓練差異記錄

## 版本資訊
- **v2**: `outputs/runs/20260216_005651_emnist36_centerloss/best_emnist36_centerloss.pt`
- **v3**: `outputs/runs/20260216_214309_emnist36_centerloss/best_emnist36_centerloss.pt`

## 訓練腳本
兩者都使用：`scripts/train_emnist36_centerloss.py`

## 已確認的差異

### 1. CSV 記錄格式
- **v2**: CSV 沒有 `alpha` 欄位
  ```
  epoch,train_loss,train_acc,val_loss,val_acc,lr_model,lr_center
  ```
- **v3**: CSV 有 `alpha` 欄位
  ```
  epoch,train_loss,train_acc,val_loss,val_acc,lr_model,lr_center,alpha
  ```

### 2. 資料處理（需要確認）
根據代碼分析 `get_emnist36_balanced_loaders`：
- **預設行為** (`augment_train=False`):
  - 所有類別：ToTensor + Normalize
  - 數值範圍：[-1.0, 1.0]（經過 Normalize((0.5,), (0.5,))）

- **v3 註釋說明**：
  - 代碼中有註釋：「所有類別都只做 ToTensor，不做 Normalize（與 v3 一致）」
  - 這表示 v3 可能修改了資料處理邏輯

### 3. 訓練參數（從 CSV 推測）
- **v2**: 
  - lr_model: 0.001 (固定)
  - 沒有記錄 alpha 值
  
- **v3**:
  - lr_model: 使用 CosineAnnealingLR（從 0.000994 開始下降）
  - alpha: 記錄了 alpha 值（Epoch 1-4 都是 0.0000）

## 需要確認的項目

### 資料處理差異
**問題**: v3 是否真的沒有做 Normalize？

**檢查方法**:
1. 檢查 v3 訓練時的 `get_emnist36_balanced_loaders` 代碼版本
2. 檢查是否有其他資料載入函數被使用
3. 檢查訓練日誌或配置文件

### 訓練指令差異
**需要記錄的指令**:
- v2 訓練指令：`python scripts/train_emnist36_centerloss.py [參數]`
- v3 訓練指令：`python scripts/train_emnist36_centerloss.py [參數]`

## 當前代碼狀態

### `get_emnist36_balanced_loaders` (當前版本)
```python
def _transpose_and_normalize():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2)),
        transforms.Normalize((0.5,), (0.5,)),  # 有 Normalize
    ])

transform_test = _transpose_and_normalize()
transform_train = _transpose_and_normalize()  # 預設有 Normalize
```

### `get_emnist36_balanced_refined_loaders` (新版本，參考 v3)
```python
def _to_tensor_only():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2)),
        # 沒有 Normalize
    ])
```

## 實際確認的差異（從 git 歷史和代碼）

### Git Commit 檢查
- **Commit**: `923cc3c08d169aed076d4100324c5956f2985e15` (2026-02-17 01:49:33)
- **檢查結果**: v3 訓練時（2026-02-16 21:43:09）的代碼版本中，`get_emnist36_balanced_loaders` **仍然有 Normalize**

### 確認的差異

1. **CSV 記錄格式**：
   - **v2**: 沒有 `alpha` 欄位（舊版腳本）
   - **v3**: 有 `alpha` 欄位（新版腳本，commit 923cc3c 修改了腳本）

2. **學習率調度**：
   - **v2**: 可能使用固定學習率（從 CSV 看 lr_model=0.001 固定）
   - **v3**: 使用 CosineAnnealingLR（從 CSV 看 lr_model 從 0.000994 開始下降）

3. **資料處理**：
   - **v2 和 v3**: **都做了 Normalize**（從 git 歷史確認）
   - 註釋「所有類別都只做 ToTensor，不做 Normalize（與 v3 一致）」是**後來添加的**，不是 v3 實際訓練時的狀態

## 結論

**v2 和 v3 的實際差異**：
1. ✅ **CSV 記錄格式**：v3 增加了 alpha 欄位記錄（腳本版本不同）
2. ✅ **學習率調度**：v3 使用 CosineAnnealingLR，v2 可能使用固定學習率
3. ❌ **資料處理**：**兩者都做了 Normalize**（從 git 歷史確認）

**重要發現**：
- v3 的註釋「不做 Normalize」是**誤導性的**，實際 v3 訓練時仍然做了 Normalize
- 真正的「不做 Normalize」版本是後來的 `get_emnist36_balanced_refined_loaders`

## 建議

1. ✅ **已完成**：創建此記錄文件
2. **未來改進**：
   - 在訓練腳本中添加配置記錄功能（自動記錄所有參數到 `config.json`）
   - 記錄每次訓練的完整指令
   - 在代碼註釋中標註版本和日期，避免混淆
