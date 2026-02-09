"""
Robustness STEP 2: Evaluate MLP/CNN/ResNet on corrupted MNIST data.

Run from project root:
  python scripts/evaluate_robustness.py [--model mlp|cnn|resnet|all]

For each model × corruption × severity:
- Calculate accuracy/loss
- Extract worst 10 cases
- Generate PCA/t-SNE embeddings
- Save metrics.json and visualizations to outputs/robustness/<model>/<corruption>/<severity>/
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.mlp import MLP
from models.cnn import CNN
from models.resnet import ResNet
from scripts.train_utils import evaluate

# Normalization (same as training)
NORMALIZE_MEAN = 0.1307
NORMALIZE_STD = 0.3081

# Fixed seeds
SEED = 42
TSNE_SEED = 42

# Corruption types and severities
CORRUPTION_TYPES = ["translation", "rotation", "scale", "shear", "perspective", "elastic"]
SEVERITIES = [1, 2, 3]

# Model checkpoints
MODEL_CHECKPOINTS = {
    "mlp": ROOT / "outputs" / "runs" / "20260208_144400_mnist_mlp" / "best_mnist_mlp.pt",
    "cnn": ROOT / "outputs" / "runs" / "20260208_155516_mnist_cnn" / "best_mnist_cnn.pt",
    "resnet": ROOT / "outputs" / "runs" / "20260208_204222_mnist_resnet" / "best_mnist_resnet.pt",
}

DATA_DIR = ROOT / "data" / "corrupt_data"
OUT_BASE = ROOT / "outputs" / "robustness"


def build_model(model_type, device):
    """Build MLP, CNN, or ResNet with same arch as training."""
    if model_type == "mlp":
        model = MLP(input_size=784, hidden_sizes=(256, 128), num_classes=10, dropout=0.0)
    elif model_type == "cnn":
        model = CNN(num_classes=10, dropout=0.0)
    else:
        model = ResNet(num_classes=10, dropout=0.0)
    return model.to(device)


def normalize_images(images):
    """Apply same normalization as training: (x - 0.1307) / 0.3081."""
    return (images - NORMALIZE_MEAN) / NORMALIZE_STD


def load_corrupted_data(corruption_type, severity):
    """
    Load corrupted data. corruption_type='clean' for baseline, or one of CORRUPTION_TYPES.
    severity: 1/2/3 for corruptions, ignored for 'clean'.
    Returns (images_tensor, labels_array) where images is (64, 1, 28, 28) normalized.
    """
    labels = np.load(DATA_DIR / "labels.npy")
    if corruption_type == "clean":
        images = torch.load(DATA_DIR / "clean.pt")
    else:
        images = torch.load(DATA_DIR / f"{corruption_type}_s{severity}.pt")
    images = normalize_images(images)
    return images, labels


def collect_predictions(model, images, labels, device):
    """Collect (image, gt, pred, confidence, loss) per sample."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    rows = []
    images = images.to(device)
    labels_tensor = torch.from_numpy(labels).long().to(device)
    
    with torch.no_grad():
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        confidences = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
        losses = criterion(logits, labels_tensor)
        
        for i in range(len(images)):
            rows.append({
                "idx": i,
                "image": images[i].cpu().squeeze(0),
                "gt": labels[i],
                "pred": preds[i].item(),
                "confidence": confidences[i].item(),
                "loss": losses[i].item(),
            })
    return rows


def collect_embeddings(model, images, device):
    """Extract embeddings using forward_embedding. Returns (embeddings, logits) numpy arrays."""
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        embedding, logits = model.forward_embedding(images)
        embeddings = embedding.cpu().numpy()
    return embeddings


def plot_worst10(rows, path, title):
    """Plot 10 worst cases (incorrect, lowest confidence)."""
    incorrect = [r for r in rows if r["gt"] != r["pred"]]
    incorrect.sort(key=lambda r: r["confidence"])
    worst10 = incorrect[:10]
    
    if not worst10:
        return
    
    n = len(worst10)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n > 1 else np.array([[axes]])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            r = worst10[idx]
            img = r["image"].numpy()
            # Denormalize for display
            img = img * NORMALIZE_STD + NORMALIZE_MEAN
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"GT:{r['gt']} Pred:{r['pred']}\nconf={r['confidence']:.2f}")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_embedding_pca(embeddings, labels, worst10_indices, path, title):
    """Plot PCA 2D embedding colored by GT label, mark worst 10."""
    pca = PCA(n_components=2, random_state=SEED)
    emb_2d = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(10):
        mask = labels == c
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=str(c), alpha=0.6, s=10)
    
    if worst10_indices:
        ax.scatter(
            emb_2d[worst10_indices, 0],
            emb_2d[worst10_indices, 1],
            marker="*",
            s=280,
            c="none",
            edgecolors="red",
            linewidths=2,
            label="Worst 10",
            zorder=5,
        )
    ax.legend()
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_embedding_tsne(embeddings, labels, worst10_indices, path, title):
    """Plot t-SNE 2D embedding colored by GT label, mark worst 10."""
    tsne = TSNE(n_components=2, random_state=TSNE_SEED, perplexity=min(30, len(embeddings) - 1), init="pca")
    emb_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(10):
        mask = labels == c
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=str(c), alpha=0.6, s=10)
    
    if worst10_indices:
        ax.scatter(
            emb_2d[worst10_indices, 0],
            emb_2d[worst10_indices, 1],
            marker="*",
            s=280,
            c="none",
            edgecolors="red",
            linewidths=2,
            label="Worst 10",
            zorder=5,
        )
    ax.legend()
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def evaluate_one_config(model, model_type, corruption_type, severity, device, clean_baseline):
    """Evaluate one model × corruption × severity configuration."""
    if corruption_type == "clean":
        sev_str = "clean"
    else:
        sev_str = f"s{severity}"
    print(f"  Evaluating {model_type} / {corruption_type} / {sev_str}...")
    
    # Load data
    images, labels = load_corrupted_data(corruption_type, severity)
    
    # Create DataLoader-like structure (batch of 64)
    images_batch = images.unsqueeze(0)  # (1, 64, 1, 28, 28)
    labels_tensor = torch.from_numpy(labels).long()
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
        loss = criterion(logits, labels_tensor.to(device)).item()
        preds = logits.argmax(dim=1).cpu().numpy()
        accuracy = (preds == labels).mean()
        num_correct = (preds == labels).sum()
    
    # Collect predictions for worst 10
    rows = collect_predictions(model, images, labels, device)
    
    # Worst 10 indices
    incorrect = [r for r in rows if r["gt"] != r["pred"]]
    incorrect.sort(key=lambda r: r["confidence"])
    worst10 = incorrect[:10]
    worst10_indices = [r["idx"] for r in worst10] if worst10 else []
    
    # Extract embeddings
    embeddings = collect_embeddings(model, images, device)
    
    # Output directory
    if corruption_type == "clean":
        out_dir = OUT_BASE / model_type / "clean"
    else:
        out_dir = OUT_BASE / model_type / corruption_type / f"s{severity}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate relative metrics
    accuracy_drop_pct = (clean_baseline["accuracy"] - accuracy) * 100 if clean_baseline else 0.0
    loss_increase = loss - clean_baseline["loss"] if clean_baseline else 0.0
    
    # Save metrics.json
    metrics = {
        "model": model_type,
        "corruption": corruption_type,
        "severity": severity if corruption_type != "clean" else None,
        "accuracy": float(accuracy),
        "loss": float(loss),
        "num_correct": int(num_correct),
        "num_total": len(labels),
        "accuracy_drop_pct": float(accuracy_drop_pct),
        "loss_increase": float(loss_increase),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Visualizations
    if worst10:
        sev_label = f" / s{severity}" if corruption_type != "clean" else ""
        plot_worst10(rows, out_dir / "worst10.png", f"{model_type} / {corruption_type}{sev_label} — Worst 10")
    
    sev_label = f" / s{severity}" if corruption_type != "clean" else ""
    title_base = f"{model_type} / {corruption_type}{sev_label}"
    plot_embedding_pca(
        embeddings, labels, worst10_indices,
        out_dir / "pca.png",
        f"{title_base} — Embedding (PCA 2D)"
    )
    plot_embedding_tsne(
        embeddings, labels, worst10_indices,
        out_dir / "tsne.png",
        f"{title_base} — Embedding (t-SNE 2D)"
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate robustness on corrupted MNIST")
    parser.add_argument("--model", type=str, default="all", choices=["mlp", "cnn", "resnet", "all"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    models_to_run = ["mlp", "cnn", "resnet"] if args.model == "all" else [args.model]
    
    all_metrics = []
    
    for model_type in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {model_type.upper()}")
        print(f"{'='*60}")
        
        # Load model
        checkpoint_path = MODEL_CHECKPOINTS[model_type]
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = build_model(model_type, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Clean baseline
        print("  Evaluating clean baseline...")
        clean_metrics = evaluate_one_config(model, model_type, "clean", None, device, None)
        clean_baseline = {"accuracy": clean_metrics["accuracy"], "loss": clean_metrics["loss"]}
        all_metrics.append(clean_metrics)
        
        # Corrupted variants
        for ctype in CORRUPTION_TYPES:
            for sev in SEVERITIES:
                metrics = evaluate_one_config(model, model_type, ctype, sev, device, clean_baseline)
                all_metrics.append(metrics)
    
    # Print summary table
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(f"{'Model':<8} | {'Corruption':<12} | {'Sev':<4} | {'Accuracy':<10} | {'Drop %':<8} | {'Loss':<8} | {'Increase':<8}")
    print("-" * 80)
    
    for m in all_metrics:
        model = m["model"]
        corr = m["corruption"]
        sev = str(m["severity"]) if m["severity"] else "—"
        acc = f"{m['accuracy']:.4f}"
        drop = f"{m['accuracy_drop_pct']:.2f}" if m["corruption"] != "clean" else "—"
        loss = f"{m['loss']:.4f}"
        inc = f"{m['loss_increase']:.4f}" if m["corruption"] != "clean" else "—"
        print(f"{model:<8} | {corr:<12} | {sev:<4} | {acc:<10} | {drop:<8} | {loss:<8} | {inc:<8}")
    
    print(f"\nAll results saved to: {OUT_BASE}")


if __name__ == "__main__":
    main()
