"""
Compare O->0 confusion between CNN and ResNet on EMNIST36 test set,
and generate side-by-side worst-case figure.

Run from project root:
  python scripts/analyze_o_to_0_compare.py
"""
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.cnn import CNN
from models.resnet import ResNet
from scripts.datasets import get_emnist_digits_uppercase_loaders, EMNIST36_LABEL_NAMES


SEED = 42
O_LABEL = EMNIST36_LABEL_NAMES.index("O")  # 24
ZERO_LABEL = EMNIST36_LABEL_NAMES.index("0")  # 0
OUT_DIR = ROOT / "outputs" / "analysis"

CHECKPOINTS = {
    "cnn": ROOT / "outputs" / "runs" / "20260210_231003_emnist36_cnn" / "best_emnist36_cnn.pt",
    "resnet": ROOT / "outputs" / "runs" / "20260211_085917_emnist36_resnet" / "best_emnist36_resnet.pt",
}


def build_model(name: str, device: torch.device):
    if name == "cnn":
        model = CNN(num_classes=36, dropout=0.0)
    elif name == "resnet":
        model = ResNet(num_classes=36, dropout=0.0)
    else:
        raise ValueError(name)
    return model.to(device)


def collect_o_to_0_cases(model, test_loader, device):
    """Return all O->0 rows with image/confidence/loss and summary stats."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    rows = []
    total_o = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            losses = criterion(logits, labels)
            conf_pred = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

            is_o = labels == O_LABEL
            total_o += int(is_o.sum().item())
            mask = is_o & (preds == ZERO_LABEL)
            idxs = torch.where(mask)[0]
            for i in idxs:
                rows.append(
                    {
                        "image": images[i].cpu().squeeze(0),
                        "confidence": float(conf_pred[i].item()),  # confidence for predicted class (0)
                        "loss": float(losses[i].item()),
                    }
                )
    # sort as worst-case: highest loss first
    rows.sort(key=lambda r: r["loss"], reverse=True)
    return rows, total_o


def plot_compare(cnn_rows, resnet_rows, out_path, k=10):
    """2 rows x k cols grid: CNN top-k O->0 worst, ResNet top-k O->0 worst."""
    k = min(k, len(cnn_rows), len(resnet_rows))
    if k == 0:
        return
    fig, axes = plt.subplots(2, k, figsize=(2 * k, 5))
    for j in range(k):
        r = cnn_rows[j]
        ax = axes[0, j]
        ax.imshow(r["image"].numpy(), cmap="gray")
        ax.set_title(f"conf={r['confidence']:.2f}\nloss={r['loss']:.2f}", fontsize=8)
        ax.axis("off")

        r2 = resnet_rows[j]
        ax2 = axes[1, j]
        ax2.imshow(r2["image"].numpy(), cmap="gray")
        ax2.set_title(f"conf={r2['confidence']:.2f}\nloss={r2['loss']:.2f}", fontsize=8)
        ax2.axis("off")

    axes[0, 0].set_ylabel("CNN\n(O->0 worst)", fontsize=10)
    axes[1, 0].set_ylabel("ResNet\n(O->0 worst)", fontsize=10)
    fig.suptitle("O -> 0 Worst-case Comparison (Top-K by loss)", fontsize=12, weight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    _, _, test_loader = get_emnist_digits_uppercase_loaders(batch_size=64, num_workers=0, seed=SEED)

    report = {}
    model_rows = {}
    for model_name, ckpt_path in CHECKPOINTS.items():
        model = build_model(model_name, device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        rows, total_o = collect_o_to_0_cases(model, test_loader, device)
        o_to_0_count = len(rows)
        o_to_0_rate = (o_to_0_count / total_o * 100.0) if total_o > 0 else 0.0
        report[model_name] = {
            "total_O_samples": int(total_o),
            "O_to_0_count": int(o_to_0_count),
            "O_to_0_rate_pct": float(o_to_0_rate),
            "avg_confidence_pred0": float(np.mean([r["confidence"] for r in rows])) if rows else 0.0,
            "avg_loss": float(np.mean([r["loss"] for r in rows])) if rows else 0.0,
        }
        model_rows[model_name] = rows

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "o_to_0_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    fig_path = OUT_DIR / "o_to_0_worstcase_compare.png"
    plot_compare(model_rows["cnn"], model_rows["resnet"], fig_path, k=10)

    print("Saved:")
    print(json_path)
    print(fig_path)
    print("\nSummary:")
    for name in ("cnn", "resnet"):
        r = report[name]
        print(
            f"{name}: O->0 {r['O_to_0_count']}/{r['total_O_samples']} "
            f"({r['O_to_0_rate_pct']:.2f}%), avg_conf={r['avg_confidence_pred0']:.3f}, avg_loss={r['avg_loss']:.3f}"
        )


if __name__ == "__main__":
    main()
