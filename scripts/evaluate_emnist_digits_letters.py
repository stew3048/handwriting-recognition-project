"""
Phase 1 — Evaluate EMNIST 36-class model: metrics, embeddings (PCA/t-SNE), confusion matrix, worst cases.
Reusable for Phase 2 (confusion analysis). Run from project root:
  python scripts/evaluate_emnist_digits_letters.py --checkpoint <path> --model cnn [--output_dir <dir>]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.mlp import MLP
from models.cnn import CNN
from models.resnet import ResNet
from scripts.datasets import (
    get_emnist_digits_uppercase_loaders,
    EMNIST36_NUM_CLASSES,
    EMNIST36_LABEL_NAMES,
)
from scripts.train_utils import evaluate

SEED = 42


def build_model(model_type, device):
    if model_type == "mlp":
        model = MLP(input_size=784, hidden_sizes=(256, 128), num_classes=EMNIST36_NUM_CLASSES, dropout=0.0)
    elif model_type == "cnn":
        model = CNN(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0)
    elif model_type == "resnet":
        model = ResNet(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0)
    elif model_type == "resnet_centerloss":
        model = ResNet(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0, embedding_dim=128)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    return model.to(device)


def collect_predictions(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    rows = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            confidences = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
            losses = criterion(logits, labels)
            for i in range(images.size(0)):
                rows.append({
                    "idx": len(rows),
                    "image": images[i].cpu().squeeze(0),
                    "gt": labels[i].item(),
                    "pred": preds[i].item(),
                    "confidence": confidences[i].item(),
                    "loss": losses[i].item(),
                })
    return rows


def collect_embeddings(model, test_loader, device):
    model.eval()
    embs, labels_list = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            embedding, _ = model.forward_embedding(images)
            embs.append(embedding.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labels_list, axis=0)


def plot_embedding_2d(emb_2d, labels_gt, worst10_indices, path, title, label_names):
    """Scatter with color by class; 36 classes use colorbar + tick labels."""
    n_classes = len(label_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        cmap = plt.colormaps["nipy_spectral"].resampled(n_classes)
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("nipy_spectral", n_classes)
    sc = ax.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=labels_gt, cmap=cmap, alpha=0.6, s=8, vmin=0, vmax=n_classes - 1,
    )
    if worst10_indices:
        ax.scatter(
            emb_2d[worst10_indices, 0], emb_2d[worst10_indices, 1],
            marker="*", s=280, c="none", edgecolors="red", linewidths=2,
            label="Worst 10", zorder=5,
        )
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(n_classes), shrink=0.8)
    cbar.ax.set_yticklabels(label_names, fontsize=6)
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_worst_grid(rows, path, title, label_names, n_cols=5):
    n = len(rows)
    n_cols = min(n_cols, n)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = np.atleast_2d(axes)
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < n:
            r = rows[idx]
            ax.imshow(r["image"].numpy(), cmap="gray")
            gt_name = label_names[r["gt"]]
            pred_name = label_names[r["pred"]]
            ax.set_title(f"GT:{gt_name} Pred:{pred_name}\nconf={r['confidence']:.2f}")
    fig.suptitle(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, path, label_names):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, fontsize=7)
    ax.set_yticklabels(label_names, fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            v = cm[i, j]
            ax.text(j, i, int(v) if v else "", ha="center", va="center", color="black" if v < cm.max() * 0.6 else "white", fontsize=5)
    ax.set_title("Confusion Matrix (GT = row, Pred = col)")
    fig.colorbar(im, ax=ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Evaluate EMNIST 36-class model: embeddings, confusion, worst cases")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, required=True, choices=["mlp", "cnn", "resnet", "resnet_centerloss"])
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no_tsne", action="store_true")
    p.add_argument("--tsne_sample", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64, help="評估時 batch 大小，顯存不足可改小或加 --cpu")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir or ROOT / "outputs" / "emnist36_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(args.model, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_loader = get_emnist_digits_uppercase_loaders(batch_size=args.batch_size, num_workers=0, seed=args.seed)
    criterion = nn.CrossEntropyLoss()
    if args.model == "resnet_centerloss":
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * x.size(0)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += x.size(0)
        test_loss, test_acc = total_loss / total, correct / total
    else:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}  Test acc={test_acc:.4f}")

    metrics = {
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "num_classes": EMNIST36_NUM_CLASSES,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    all_rows = collect_predictions(model, test_loader, device)
    incorrect = [r for r in all_rows if r["gt"] != r["pred"]]
    incorrect.sort(key=lambda r: r["confidence"])
    worst10 = incorrect[:10]
    worst10_indices = [r["idx"] for r in worst10] if worst10 else []

    if worst10:
        plot_worst_grid(worst10, out_dir / "top10_worst_cases.png", "Top 10 worst (lowest confidence)", EMNIST36_LABEL_NAMES)
    else:
        print("No incorrect predictions; skipping top10_worst_cases.png")

    # Confusion matrix
    preds_all = np.array([r["pred"] for r in all_rows])
    labels_all = np.array([r["gt"] for r in all_rows])
    cm = confusion_matrix(labels_all, preds_all)
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png", EMNIST36_LABEL_NAMES)
    np.save(out_dir / "confusion_matrix.npy", cm)

    # O/0, I/1, Z/2, S/5 個別準確率（EMNIST36: 0-9 數字, 10-35 A-Z；5→5, O→24, I→18, Z→35, S→28）
    idx_0, idx_O = 0, 24
    idx_1, idx_I = 1, 18
    idx_2, idx_Z = 2, 35
    idx_5, idx_S = 5, 28

    def per_class_acc(labels_all, preds_all, class_idx):
        mask = labels_all == class_idx
        n = mask.sum()
        if n == 0:
            return 0.0, 0
        correct = (preds_all[mask] == class_idx).sum()
        return float(correct) / float(n), int(n)

    acc_0, n_0 = per_class_acc(labels_all, preds_all, idx_0)
    acc_O, n_O = per_class_acc(labels_all, preds_all, idx_O)
    acc_1, n_1 = per_class_acc(labels_all, preds_all, idx_1)
    acc_I, n_I = per_class_acc(labels_all, preds_all, idx_I)
    acc_2, n_2 = per_class_acc(labels_all, preds_all, idx_2)
    acc_Z, n_Z = per_class_acc(labels_all, preds_all, idx_Z)
    acc_5, n_5 = per_class_acc(labels_all, preds_all, idx_5)
    acc_S, n_S = per_class_acc(labels_all, preds_all, idx_S)

    metrics["acc_O"] = acc_O
    metrics["acc_0"] = acc_0
    metrics["n_O"] = n_O
    metrics["n_0"] = n_0
    metrics["acc_I"] = acc_I
    metrics["acc_1"] = acc_1
    metrics["n_I"] = n_I
    metrics["n_1"] = n_1
    metrics["acc_Z"] = acc_Z
    metrics["acc_2"] = acc_2
    metrics["n_Z"] = n_Z
    metrics["n_2"] = n_2
    metrics["acc_S"] = acc_S
    metrics["acc_5"] = acc_5
    metrics["n_S"] = n_S
    metrics["n_5"] = n_5
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 可選：每對 2×2 混淆（僅該兩類）
    for name, (a, b) in [("O_0", (idx_O, idx_0)), ("I_1", (idx_I, idx_1)), ("Z_2", (idx_Z, idx_2)), ("S_5", (idx_S, idx_5))]:
        mask = (labels_all == a) | (labels_all == b)
        if mask.sum() == 0:
            continue
        cm_ab = confusion_matrix(labels_all[mask], preds_all[mask], labels=[a, b])
        np.save(out_dir / f"confusion_2x2_{name}.npy", cm_ab)

    # Embeddings
    embeddings, labels_gt = collect_embeddings(model, test_loader, device)

    # PCA
    pca = PCA(n_components=2, random_state=args.seed)
    emb_2d_pca = pca.fit_transform(embeddings)
    plot_embedding_2d(
        emb_2d_pca, labels_gt, worst10_indices,
        out_dir / "embedding_pca.png",
        "Embedding (PCA 2D) — 36 classes, Worst 10 marked",
        EMNIST36_LABEL_NAMES,
    )

    # t-SNE
    if not args.no_tsne:
        n_tsne = min(args.tsne_sample, len(embeddings))
        n_tsne = max(n_tsne, len(worst10_indices))
        if n_tsne < len(embeddings):
            other = [i for i in range(len(embeddings)) if i not in set(worst10_indices)]
            need = n_tsne - len(worst10_indices)
            idx = list(worst10_indices) + list(np.random.choice(other, size=min(need, len(other)), replace=False))
            emb_sub = embeddings[idx]
            lab_sub = labels_gt[idx]
            worst10_in_sub = list(range(len(worst10_indices)))
        else:
            emb_sub, lab_sub = embeddings, labels_gt
            worst10_in_sub = worst10_indices
        tsne = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, len(emb_sub) - 1), init="pca")
        emb_2d_tsne = tsne.fit_transform(emb_sub)
        plot_embedding_2d(
            emb_2d_tsne, lab_sub, worst10_in_sub,
            out_dir / "embedding_tsne.png",
            "Embedding (t-SNE 2D) — 36 classes, Worst 10 marked",
            EMNIST36_LABEL_NAMES,
        )

    print(f"Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
