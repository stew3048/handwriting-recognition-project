"""
MNIST 評估與視覺化：載入 checkpoint，在官方 test set 上評估並產出圖表。
不重新訓練、不修改權重。從專案根目錄執行：
  python scripts/evaluate_mnist.py --checkpoint outputs/best_mnist_cnn.pt --model cnn
"""
import argparse
import random
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.mlp import MLP
from models.cnn import CNN
from scripts.datasets import get_mnist_loaders
from scripts.train_utils import evaluate


def build_model(model_type, device):
    """Build MLP or CNN with same arch as training. num_classes=10 for MNIST."""
    if model_type == "mlp":
        model = MLP(input_size=784, hidden_sizes=(256, 128), num_classes=10, dropout=0.0)
    else:
        model = CNN(num_classes=10, dropout=0.0)
    return model.to(device)


def collect_predictions(model, test_loader, device):
    """Collect (image, gt, pred, confidence, loss) per sample. Images are 1x28x28."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    rows = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            confidences = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
            losses = criterion(logits, labels)
            for i in range(images.size(0)):
                rows.append({
                    "image": images[i].cpu().squeeze(0),
                    "gt": labels[i].item(),
                    "pred": preds[i].item(),
                    "confidence": confidences[i].item(),
                    "loss": losses[i].item(),
                })
    return rows


def plot_grid(rows, path, title, n_cols=5):
    """Plot grid: image, GT, pred, confidence per cell. rows is list of dicts."""
    n = len(rows)
    n_cols = min(n_cols, n)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.atleast_1d(axes).reshape(1, -1)
    elif n_cols == 1:
        axes = np.atleast_1d(axes).reshape(-1, 1)
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            r = rows[idx]
            ax.imshow(r["image"].numpy(), cmap="gray")
            ax.set_title(f"GT:{r['gt']} Pred:{r['pred']}\nconf={r['confidence']:.2f}")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# MNIST 常見混淆對與可能原因（用於 worst10_analysis.md）
CONFUSION_REASONS = {
    (4, 9): "4 與 9：上方封口 vs 開口、下半部圓弧相似；4 開口大或 9 圈小時易混。",
    (9, 4): "9 與 4：同上。",
    (3, 5): "3 與 5：上半/下半圓弧與開合方向類似。",
    (5, 3): "5 與 3：同上。",
    (3, 8): "3 與 8：手寫 3 若上半像 8 或 8 斷開易混。",
    (8, 3): "8 與 3：同上。",
    (5, 8): "5 與 8：圓弧數量與方向類似。",
    (8, 5): "8 與 5：同上。",
    (7, 1): "7 與 1：7 加橫槓、1 加底座或斜線時在 28×28 下結構相似。",
    (1, 7): "1 與 7：同上。",
    (2, 7): "2 與 7：2 的上半弧與 7 的斜線在低解析度下難區分。",
    (7, 2): "7 與 2：同上。",
    (5, 6): "5 與 6：5 的圈與 6 的圈、尾巴方向書寫不清時易混。",
    (6, 5): "6 與 5：同上。",
    (0, 6): "0 與 6：0 略橢圓或有一小缺口可能被看成 6。",
    (6, 0): "6 與 0：同上。",
    (0, 8): "0 與 8：0 若有缺口可能被看成 8。",
    (8, 0): "8 與 0：同上。",
    (8, 9): "8 與 9：8 若寫得開口或下半圓像 9、或 9 的圈小，在 28×28 下易混。",
    (9, 8): "9 與 8：同上。",
    (8, 2): "8 與 2：8 斷開或 2 的上弧明顯時，結構在低解析度下可能相近。",
    (2, 8): "2 與 8：同上。",
    (9, 7): "9 與 7：9 的圈小或 7 加橫槓時，頭部形狀可能相似。",
    (9, 5): "9 與 5：9 的圈與 5 的弧線、轉折在書寫模糊時易混。",
    (5, 9): "5 與 9：同上。",
    (4, 6): "4 與 6：4 的開口大或 6 的圈不明顯時，下半部可能混淆。",
    (6, 4): "6 與 4：同上。",
    (2, 3): "2 與 3：2 的上弧與 3 的轉折在書寫風格相近時易混。",
    (3, 2): "3 與 2：同上。",
}


def write_worst10_analysis(worst10, out_dir, model_type):
    """寫入 worst10_analysis.md：表格 + 混淆對統計 + 可能原因。"""
    path = Path(out_dir) / "worst10_analysis.md"
    lines = [
        "# Top 10 Worst 分析",
        "",
        "「Top 10 worst」為**預測錯誤**且**模型 confidence 最低**的 10 筆。",
        "",
        "## 1. 本 run 的 Worst 10 明細",
        "",
        "| Rank | GT | Pred | Confidence |",
        "|------|-----|------|------------|",
    ]
    for i, r in enumerate(worst10, 1):
        lines.append(f"| {i} | {r['gt']} | {r['pred']} | {r['confidence']:.4f} |")
    lines.extend(["", "## 2. 混淆對統計 (GT → Pred)", ""])
    from collections import Counter
    pairs = Counter((r["gt"], r["pred"]) for r in worst10)
    for (gt, pred), count in pairs.most_common():
        reason = CONFUSION_REASONS.get((gt, pred), "視覺或書寫上相近，或邊界樣本。")
        lines.append(f"- **{gt} → {pred}**（{count} 筆）：{reason}")
    lines.extend([
        "",
        "## 3. 可能原因摘要",
        "",
        "- **視覺相似**：4/9、3/5/8、7/1、2/7、5/6、0/6/8 等對在 28×28 下本就易混。",
        "- **書寫模糊**：手寫介於兩類之間 → 模型 confidence 低、易錯。",
        f"- **模型**：本 run 為 **{model_type.upper()}**；若為 MLP 則無局部結構，對細微筆畫較不敏感；CNN 則較能抓局部，但邊界樣本仍可能進 worst 10。",
        "",
        "可對照 **top10_worst_cases.png** 看每張影像的 GT / Pred / confidence。",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_embeddings(model, test_loader, device):
    """Run forward_embedding on test set; return (embeddings, labels) numpy arrays."""
    model.eval()
    embs, labels_list = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            embedding, _ = model.forward_embedding(images)
            embs.append(embedding.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labels_list, axis=0)


def main():
    p = argparse.ArgumentParser(description="Evaluate MNIST model on test set and produce visualizations.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (e.g. outputs/best_mnist_cnn.pt)")
    p.add_argument("--model", type=str, required=True, choices=["mlp", "cnn"])
    p.add_argument("--output_dir", type=str, default=None, help="Defaults to outputs/")
    p.add_argument("--n_sample", type=int, default=25, help="Number of random samples for pred_vs_gt grid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_tsne", action="store_true", help="Skip t-SNE (slow on full test set)")
    p.add_argument("--tsne_sample", type=int, default=2000, help="Subsample size for t-SNE if not full")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir or ROOT / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(args.model, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader = get_mnist_loaders(batch_size=64, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}  Test acc={test_acc:.4f}")
    with open(out_dir / "eval_results.txt", "w") as f:
        f.write(f"checkpoint={args.checkpoint} model={args.model}\n")
        f.write(f"Test loss={test_loss:.4f}  Test acc={test_acc:.4f}\n")

    # 1. Prediction vs Ground Truth: random sample N
    all_rows = collect_predictions(model, test_loader, device)
    sampled = random.sample(all_rows, min(args.n_sample, len(all_rows)))
    plot_grid(sampled, out_dir / "pred_vs_gt_grid.png", "Prediction vs Ground Truth (random sample)")

    # 2. Worst-case: incorrect, sorted by confidence ascending, top 10
    incorrect = [r for r in all_rows if r["gt"] != r["pred"]]
    incorrect.sort(key=lambda r: r["confidence"])
    worst10 = incorrect[:10]
    if worst10:
        plot_grid(worst10, out_dir / "top10_worst_cases.png", "Top 10 worst cases (lowest confidence)")
        write_worst10_analysis(worst10, out_dir, args.model)
    else:
        print("No incorrect predictions; skipping top10_worst_cases.png")

    # 3 & 4. Embedding extraction and visualization
    embeddings, labels_gt = collect_embeddings(model, test_loader, device)

    # PCA (mandatory)
    pca = PCA(n_components=2, random_state=args.seed)
    emb_2d_pca = pca.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(10):
        mask = labels_gt == c
        ax.scatter(emb_2d_pca[mask, 0], emb_2d_pca[mask, 1], label=str(c), alpha=0.6, s=10)
    ax.legend()
    ax.set_title("Embedding (PCA 2D), colored by GT label")
    plt.savefig(out_dir / "embedding_pca.png", dpi=120, bbox_inches="tight")
    plt.close()

    # t-SNE: fixed init so same seed gives same figure every time (init='pca' + random_state)
    if not args.no_tsne:
        n_tsne = min(args.tsne_sample, len(embeddings))
        if n_tsne < len(embeddings):
            idx = np.random.choice(len(embeddings), n_tsne, replace=False)
            emb_sub = embeddings[idx]
            lab_sub = labels_gt[idx]
        else:
            emb_sub, lab_sub = embeddings, labels_gt
        tsne = TSNE(
            n_components=2,
            random_state=args.seed,
            perplexity=30,
            init="pca",
        )
        emb_2d_tsne = tsne.fit_transform(emb_sub)
        fig, ax = plt.subplots(figsize=(8, 6))
        for c in range(10):
            mask = lab_sub == c
            ax.scatter(emb_2d_tsne[mask, 0], emb_2d_tsne[mask, 1], label=str(c), alpha=0.6, s=10)
        ax.legend()
        ax.set_title("Embedding (t-SNE 2D), colored by GT label")
        plt.savefig(out_dir / "embedding_tsne.png", dpi=120, bbox_inches="tight")
        plt.close()

    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
