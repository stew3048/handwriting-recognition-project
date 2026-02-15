# Research Plan: Digits + Letters Representation Structure

## Context

- **Completed**: Full robustness and representation analysis on MNIST (MLP / CNN / ResNet): clean vs corrupted, accuracy / worst-case errors, embedding geometry (PCA / t-SNE).
- **New phase**: Handwritten **digits + letters** recognition — harder and more realistic; focus on **representation structure**, not raw accuracy.

---

## Primary Research Goal

Understand how neural networks organize representations when **multiple classes share highly similar visual structures**, e.g.:

- 0 vs O  
- 1 vs I  
- Z vs 2  
- S vs 5  

We study:

- Whether embeddings naturally **cluster by semantic/structural similarity**
- How **class confusion** emerges in representation space
- How **model inductive biases** affect this structure  

**Accuracy is NOT the main goal in the first stage. Representation geometry is.**

---

## Task Definition

| Item | Choice |
|------|--------|
| **Dataset** | EMNIST (digits + uppercase letters) |
| **Label space** | 0–9 + A–Z → **36 classes** |
| **Models** | CNN and ResNet (MLP optional as weak baseline) |
| **Training** | Clean data only at first; **no data augmentation** in initial experiments |
| **Letter choice** | **Uppercase** first (consistent geometry); lowercase in a later phase. Pipeline designed so switching to lowercase is easy. |

---

## Experiment Phases

### PHASE 1 — Clean Representation Baseline (current)

- Train models on **clean** data only.
- Evaluate on **clean** test data.
- Extract **embeddings** from the layer before the classifier.
- Visualize with **PCA** and **t-SNE**.
- Analyze:
  - Do digits and letters separate?
  - Do visually similar classes cluster together?
  - Where do worst-case errors appear in embedding space?

**Deliverable**: Trained baseline, clean embedding visualizations, short analysis (embedding structure, types of confusions, why this task is qualitatively harder than MNIST).

### PHASE 2 — Controlled Difficulty Analysis (later)

- Identify naturally confusing class pairs (e.g. 0/O, 1/I).
- Analyze confusion matrices and worst cases.
- Relate errors back to embedding geometry.

### PHASE 3 — Robustness Extension (later)

- Apply geometric corruptions (translation, rotation, perspective, etc.).
- Reuse methodology from MNIST robustness.
- Compare how representation structure degrades.

---

## Engineering

- **Reusable**: Embeddings, worst cases, and metrics usable across phases.
- **Modular**: Evaluation scripts modular (data / model / viz separated where possible).
- **Reproducibility**: Fixed random seeds; clear naming for plots and results.
- **Extensibility**: Easy to switch to lowercase letters or add augmentation later.

---

## File / Naming Conventions

- **Data**: `scripts/datasets.py` — `get_emnist_digits_uppercase_loaders()` (36 classes); later add variant for lowercase.
- **Training**: `scripts/train_emnist_digits_letters.py` — CNN / ResNet (optional MLP), no aug, checkpoint under `outputs/runs/<timestamp>_emnist36_<model>/`.
- **Evaluation**: `scripts/evaluate_emnist_digits_letters.py` — embeddings, PCA/t-SNE, confusion matrix, worst cases; output under `outputs/runs/.../eval/` or specified dir.
- **Analysis**: Notes in `notes/` with date and topic (e.g. `YYYY-MM-DD_emnist36-phase1-analysis.md`).
