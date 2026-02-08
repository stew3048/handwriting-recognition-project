# 專案規則

## 日誌命名

- 檔名格式：**`YYYY-MM-DD_統整名詞.md`**（例如 `2026-02-08_mnist-arch-and-eval-rules.md`）。
- 統整名詞：當天內容的簡短摘要（英文或中文皆可）。

---

## 用 Test Data 評估時的產出

**每當你請我用 test data 做評估時，我都必須產出並放到該 run 的 checkpoint 資料夾中：**

1. **pred_vs_gt_grid.png** — 隨機取樣（例如 25 張）的「影像 + GT + 預測 + confidence」網格
2. **top10_worst_cases.png** — 預測錯誤且 confidence 最低的 10 筆，網格顯示影像 / GT / Pred / confidence
3. **embedding_pca.png** — embedding 用 PCA 降成 2D，依 GT 上色
4. **embedding_tsne.png** — embedding 用 t-SNE 降成 2D（固定 random_state、init='pca'），依 GT 上色
5. **eval_results.txt** — test loss、test accuracy 等文字紀錄

以上產出都要與該次 run 的 checkpoint 放在**同一個 run 資料夾**（例如 `outputs/runs/YYYYMMDD_HHMMSS_mnist_mlp/`）。

**重要**：執行 `evaluate_mnist.py` 時**必須產出 embedding_tsne.png**，不得使用 `--no_tsne`，除非使用者明確要求跳過 t-SNE。
