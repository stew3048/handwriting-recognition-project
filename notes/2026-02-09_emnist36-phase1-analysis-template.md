# EMNIST 36-Class Phase 1 — Analysis Template

**After training and running evaluation**, fill in this short analysis (or copy to a dated note).

---

## 1. Embedding structure

- **Do digits and letters separate?**  
  (e.g. In PCA/t-SNE, do 0–9 form one broad region and A–Z another? Or mixed?)

- **Do visually similar classes cluster together?**  
  (e.g. 0/O, 1/I, Z/2, S/5 — do they sit close in embedding space?)

- **Where do worst-case errors appear?**  
  (e.g. On the boundary between two clusters? Inside the wrong cluster?)

---

## 2. Types of confusions

- **From confusion matrix**: Which (GT → Pred) pairs have the most errors?  
  (e.g. O→0, 0→O, I→1, 1→I, 2→Z, 5→S, …)

- **From top-10 worst**: Do the worst cases align with these pairs? Any surprise?

---

## 3. Why this task is qualitatively harder than MNIST

- **More classes**: 36 vs 10 → more boundaries, more overlap.
- **Shared visual structure**: Digits and letters share shapes (0/O, 1/I, etc.) → model must use finer cues.
- **Same resolution**: Still 28×28 → less pixel-level detail for disambiguation.

---

## 4. Model comparison (if you ran CNN and ResNet)

- Which model gives cleaner digit/letter separation in embeddings?
- Which has fewer confusions on the critical pairs (0/O, 1/I, …)?

---

**Figures to refer to**: `embedding_pca.png`, `embedding_tsne.png`, `confusion_matrix.png`, `top10_worst_cases.png`, `metrics.json`.
