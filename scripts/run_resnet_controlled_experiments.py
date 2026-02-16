"""
Run two controlled ResNet experiments on EMNIST36:

Run A: Balanced Sampling (WeightedRandomSampler) + CrossEntropy
Run B: Random Sampling + Focal Loss (gamma=2.0, alpha=1.0)

Same architecture/hyperparams as baseline: ResNet, Adam(lr=1e-3), epochs=10, batch_size=10, seed=42.

Deliverables:
- metrics_runA.json / metrics_runB.json
- confusion_runA.png / confusion_runB.png
- embedding_pairs_runA.png / embedding_pairs_runB.png
- printed comparison (O recall, 0 recall, O->0, 0->O, overall acc)
"""
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.resnet import ResNet
from scripts.datasets import EMNIST36_LABEL_NAMES, get_emnist_digits_uppercase_loaders
from scripts.train_utils import evaluate, save_checkpoint, train_one_epoch


SEED = 42
NUM_CLASSES = 36
EPOCHS = 10
BATCH_SIZE = 10
LR = 1e-3

PAIR_LABELS = ["0", "O", "1", "I", "2", "Z", "5", "S"]
PAIR_CLASS_IDS = [EMNIST36_LABEL_NAMES.index(x) for x in PAIR_LABELS]
O_ID = EMNIST36_LABEL_NAMES.index("O")
ZERO_ID = EMNIST36_LABEL_NAMES.index("0")

OUT_BASE = ROOT / "outputs" / "analysis" / "resnet_controlled"
RUNA_DIR = OUT_BASE / "runA_balanced_sampler"
RUNB_DIR = OUT_BASE / "runB_focal_loss"


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce
        return loss.mean()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_resnet(device: torch.device) -> ResNet:
    return ResNet(num_classes=NUM_CLASSES, dropout=0.0).to(device)


def get_label_from_dataset(ds, idx: int) -> int:
    """Fast-ish label retrieval supporting Subset/Concat/EMNISTMemmap."""
    if isinstance(ds, Subset):
        return get_label_from_dataset(ds.dataset, ds.indices[idx])
    if isinstance(ds, ConcatDataset):
        # map global idx to child dataset
        running = 0
        for child in ds.datasets:
            n = len(child)
            if idx < running + n:
                return get_label_from_dataset(child, idx - running)
            running += n
        raise IndexError(idx)
    # EMNISTMemmap has labels + optional target_transform
    if hasattr(ds, "labels"):
        y = int(ds.labels[idx])
        if hasattr(ds, "target_transform") and ds.target_transform is not None:
            y = int(ds.target_transform(y))
        return y
    # fallback
    return int(ds[idx][1])


def make_balanced_sampler(train_ds) -> WeightedRandomSampler:
    labels = np.array([get_label_from_dataset(train_ds, i) for i in range(len(train_ds))], dtype=np.int64)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    counts = np.maximum(counts, 1)
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)


@dataclass
class EvalArtifacts:
    accuracy: float
    loss: float
    per_class_precision: List[float]
    per_class_recall: List[float]
    confusion: np.ndarray
    o_recall: float
    zero_recall: float
    o_to_zero: int
    zero_to_o: int


def collect_eval_rows(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    rows = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            emb, logits = model.forward_embedding(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            conf = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
            for i in range(images.size(0)):
                rows.append(
                    {
                        "embedding": emb[i].cpu().numpy(),
                        "gt": int(labels[i].item()),
                        "pred": int(preds[i].item()),
                        "confidence": float(conf[i].item()),
                    }
                )
    return rows


def evaluate_and_save(model: nn.Module, test_loader: DataLoader, device: torch.device, out_dir: Path, run_tag: str) -> EvalArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    rows = collect_eval_rows(model, test_loader, device)
    gt = np.array([r["gt"] for r in rows], dtype=np.int64)
    pred = np.array([r["pred"] for r in rows], dtype=np.int64)
    embs = np.array([r["embedding"] for r in rows], dtype=np.float32)
    conf = np.array([r["confidence"] for r in rows], dtype=np.float32)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for g, p in zip(gt, pred):
        cm[g, p] += 1

    precision = []
    recall = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision.append(float(p))
        recall.append(float(r))

    o_recall = recall[O_ID]
    zero_recall = recall[ZERO_ID]
    o_to_zero = int(cm[O_ID, ZERO_ID])
    zero_to_o = int(cm[ZERO_ID, O_ID])

    # confusion image
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix ({run_tag})")
    ax.set_xlabel("Pred")
    ax.set_ylabel("GT")
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(EMNIST36_LABEL_NAMES, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(EMNIST36_LABEL_NAMES, fontsize=7)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_{run_tag}.png", dpi=140, bbox_inches="tight")
    plt.close()

    # embedding t-SNE for selected pair classes
    pair_mask = np.isin(gt, PAIR_CLASS_IDS)
    pair_emb = embs[pair_mask]
    pair_gt = gt[pair_mask]
    pair_conf = conf[pair_mask]
    if len(pair_emb) > 1:
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(pair_emb) - 1), init="pca")
        pair_2d = tsne.fit_transform(pair_emb)

        # worst 10 low-confidence in selected classes
        worst_idx = np.argsort(pair_conf)[: min(10, len(pair_conf))]

        fig, ax = plt.subplots(figsize=(9, 7))
        for cls in PAIR_CLASS_IDS:
            m = pair_gt == cls
            ax.scatter(pair_2d[m, 0], pair_2d[m, 1], s=8, alpha=0.65, label=EMNIST36_LABEL_NAMES[cls])
        ax.scatter(
            pair_2d[worst_idx, 0],
            pair_2d[worst_idx, 1],
            marker="*",
            s=220,
            c="none",
            edgecolors="red",
            linewidths=1.8,
            label="Worst10 low-conf",
            zorder=5,
        )
        ax.set_title(f"t-SNE selected pairs ({run_tag})")
        ax.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"embedding_pairs_{run_tag}.png", dpi=150, bbox_inches="tight")
        plt.close()

    metrics = {
        "run_tag": run_tag,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "per_class_precision": {EMNIST36_LABEL_NAMES[i]: precision[i] for i in range(NUM_CLASSES)},
        "per_class_recall": {EMNIST36_LABEL_NAMES[i]: recall[i] for i in range(NUM_CLASSES)},
        "O_recall": float(o_recall),
        "0_recall": float(zero_recall),
        "O_to_0_count": o_to_zero,
        "0_to_O_count": zero_to_o,
    }
    with open(out_dir / f"metrics_{run_tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    np.save(out_dir / f"confusion_{run_tag}.npy", cm)

    return EvalArtifacts(
        accuracy=float(test_acc),
        loss=float(test_loss),
        per_class_precision=precision,
        per_class_recall=recall,
        confusion=cm,
        o_recall=float(o_recall),
        zero_recall=float(zero_recall),
        o_to_zero=o_to_zero,
        zero_to_o=zero_to_o,
    )


def train_one_run(
    run_tag: str,
    use_balanced_sampler: bool,
    use_focal: bool,
    out_dir: Path,
    device: torch.device,
) -> EvalArtifacts:
    set_seed(SEED)
    train_loader, val_loader, test_loader = get_emnist_digits_uppercase_loaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
        augment_train=False,
        seed=SEED,
        use_memmap=True,
    )

    train_ds = train_loader.dataset
    if use_balanced_sampler:
        sampler = make_balanced_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = build_resnet(device)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = FocalLoss(gamma=2.0, alpha=1.0) if use_focal else nn.CrossEntropyLoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
        print(
            f"[{run_tag}] epoch {epoch:2d} | train loss={train_loss:.4f} acc={train_acc:.4f} "
            f"| val loss={val_loss:.4f} acc={val_acc:.4f}",
            flush=True,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, str(out_dir / f"best_{run_tag}.pt"))
        save_checkpoint(model, optimizer, epoch, val_acc, str(out_dir / f"last_{run_tag}.pt"))

    ckpt = torch.load(out_dir / f"best_{run_tag}.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return evaluate_and_save(model, test_loader, device, out_dir, run_tag)


def main():
    device = torch.device("cpu")
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print("Starting controlled experiments on ResNet (CPU)...", flush=True)
    print("Run A: Balanced sampler + CE", flush=True)
    run_a = train_one_run(
        run_tag="runA",
        use_balanced_sampler=True,
        use_focal=False,
        out_dir=RUNA_DIR,
        device=device,
    )

    print("\nRun B: Random sampler + Focal Loss", flush=True)
    run_b = train_one_run(
        run_tag="runB",
        use_balanced_sampler=False,
        use_focal=True,
        out_dir=RUNB_DIR,
        device=device,
    )

    print("\n=== Short comparison (RunB - RunA) ===")
    print(f"Overall accuracy: {run_b.accuracy:.4f} - {run_a.accuracy:.4f} = {run_b.accuracy - run_a.accuracy:+.4f}")
    print(f"O recall change: {run_b.o_recall:.4f} - {run_a.o_recall:.4f} = {run_b.o_recall - run_a.o_recall:+.4f}")
    print(
        f"0 recall change: {run_b.zero_recall:.4f} - {run_a.zero_recall:.4f} = "
        f"{run_b.zero_recall - run_a.zero_recall:+.4f}"
    )
    print(f"O->0 difference: {run_b.o_to_zero} - {run_a.o_to_zero} = {run_b.o_to_zero - run_a.o_to_zero:+d}")
    print(f"0->O difference: {run_b.zero_to_o} - {run_a.zero_to_o} = {run_b.zero_to_o - run_a.zero_to_o:+d}")
    print(f"\nOutputs saved under: {OUT_BASE}")


if __name__ == "__main__":
    main()
