"""
EMNIST ResNet 終極優化：精準打擊 0/O/1/I。
- 極限過採樣：0, O, 1, I 每類 15,000，其餘 5,000。
- CE 權重：僅 0, O, 1, I 權重 2.0，其餘 1.0；label_smoothing=0.1。
- 強效擴增：僅對 0,O,1,I 套用 RandomAffine(degrees=20, translate=(0.15,0.15))。
- 25 Epoch；Center Loss：1-5 warmup，6-15 穩定 0.05，16-25 線性 0.05→0.15。
- 每 epoch 記錄 O/0、I/1 中心點距離與 2x2 混淆誤判率。
- 最佳模型存為 final_precision_model.pt。
"""
import argparse
import csv
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.resnet import ResNet
from scripts.center_loss import CenterLoss
from scripts.datasets import (
    get_emnist36_precision_targeting_loaders,
    EMNIST36_NUM_CLASSES,
)

SEED = 42
NUM_CLASSES = 36
EMBEDDING_DIM = 64
TARGET_CLASSES = (0, 24, 1, 18)  # 0, O, 1, I
TOTAL_EPOCHS = 25
WARMUP_EPOCHS = 5
ALPHA_STABLE_END = 15
ALPHA_START = 0.05
ALPHA_END = 0.15


def get_alpha(epoch):
    """1-5: 0；6-15: 0.05；16-25: 線性 0.05→0.15。"""
    if epoch <= WARMUP_EPOCHS:
        return 0.0
    if epoch <= ALPHA_STABLE_END:
        return ALPHA_START
    progress = (epoch - ALPHA_STABLE_END) / (TOTAL_EPOCHS - ALPHA_STABLE_END)
    return ALPHA_START + progress * (ALPHA_END - ALPHA_START)


def train_one_epoch(model, train_loader, criterion_ce, criterion_center, optimizer_model, optimizer_center, device, alpha):
    model.train()
    if alpha > 0:
        criterion_center.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer_model.zero_grad()
        optimizer_center.zero_grad()
        logits, embeddings = model(x)
        loss_ce = criterion_ce(logits, y)
        loss = loss_ce
        if alpha > 0:
            loss_center = criterion_center(embeddings, y)
            loss = loss + alpha * loss_center
        loss.backward()
        optimizer_model.step()
        if alpha > 0:
            optimizer_center.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate_val(model, loader, criterion_ce, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion_ce(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def get_embeddings_labels_preds(model, loader, device):
    """一次遍歷取得 val embeddings、labels、preds。"""
    model.eval()
    embs, labels_list, preds_list = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb, logits = model.forward_embedding(x)
            embs.append(emb.cpu().numpy())
            labels_list.append(y.numpy())
            preds_list.append(logits.argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(embs, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(preds_list, axis=0),
    )


def centroid_distance_and_confusion(embeddings, labels, preds, idx_a, idx_b):
    """回傳 (類心距離, 兩類整體準確率)。preds 可從 logits.argmax 取得，需先跑完 val。"""
    mask = (labels == idx_a) | (labels == idx_b)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    emb_a = embeddings[labels == idx_a]
    emb_b = embeddings[labels == idx_b]
    if len(emb_a) == 0 or len(emb_b) == 0:
        return float("nan"), float("nan")
    c_a = np.mean(emb_a, axis=0)
    c_b = np.mean(emb_b, axis=0)
    dist = float(np.linalg.norm(c_a - c_b))
    # 兩類整體準確率（在該兩類樣本上）
    correct = ((labels[mask] == preds[mask]).sum())
    acc = correct / mask.sum()
    return dist, float(acc)


def save_checkpoint(model, criterion_center, optimizer_model, optimizer_center, epoch, val_acc, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "center_state_dict": criterion_center.state_dict(),
        "optimizer_model_state_dict": optimizer_model.state_dict(),
        "optimizer_center_state_dict": optimizer_center.state_dict(),
        "val_acc": val_acc,
    }, path)


def main():
    p = argparse.ArgumentParser(description="EMNIST 36 精準打擊 0/O/1/I，存 final_precision_model.pt")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_model", type=float, default=1e-3)
    p.add_argument("--lr_center", type=float, default=0.5)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    out_dir = Path(args.out_dir or ROOT / "outputs")
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_emnist36_precision_final"
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = run_dir / "error.log"

    def log_error(msg, exc=None):
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            if exc is not None:
                traceback.print_exc(file=f)
        print(msg, flush=True)
        if exc is not None:
            traceback.print_exc()

    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    # Test 與 train 同分布：四類 2400（3x）、其餘 800；若不設則 test 每類 800 平衡
    train_loader, val_loader, test_loader = get_emnist36_precision_targeting_loaders(
        target_classes=TARGET_CLASSES,
        target_per_class=15000,
        other_per_class=5000,
        samples_per_class_test=800,
        test_target_per_class=2400,
        test_other_per_class=800,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=0,
        seed=SEED,
    )
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    # CE 權重：0, O(24), 1, I(18) = 2.0，其餘 1.0
    class_weights = torch.ones(NUM_CLASSES, device=device)
    for c in TARGET_CLASSES:
        class_weights[c] = 2.0

    model = ResNet(num_classes=NUM_CLASSES, dropout=0.0, embedding_dim=EMBEDDING_DIM).to(device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    criterion_center = CenterLoss(num_classes=NUM_CLASSES, feat_dim=EMBEDDING_DIM).to(device)
    optimizer_model = Adam(model.parameters(), lr=args.lr_model)
    optimizer_center = Adam(criterion_center.parameters(), lr=args.lr_center)
    scheduler_model = CosineAnnealingLR(optimizer_model, T_max=TOTAL_EPOCHS, eta_min=1e-5)

    print("=== EMNIST 36 精準打擊：0, O, 1, I 過採樣 15000 + CE 權重 2.0 + 強擴增 ===", flush=True)
    print(f"Center Loss: 1-5 warmup, 6-15 alpha=0.05, 16-25 線性 0.05→0.15", flush=True)
    print(f"Train size: {train_size}  Val size: {val_size}  Epochs: {TOTAL_EPOCHS}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(flush=True)

    log_path = run_dir / "experiments.csv"
    metrics_path = run_dir / "o0_i1_metrics.csv"
    best_val_acc = 0.0
    file_exists = log_path.exists()
    metrics_exists = metrics_path.exists()

    for epoch in range(1, TOTAL_EPOCHS + 1):
        alpha = get_alpha(epoch)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion_ce, criterion_center,
            optimizer_model, optimizer_center, device, alpha=alpha,
        )
        val_loss, val_acc = evaluate_val(model, val_loader, criterion_ce, device)
        scheduler_model.step()
        current_lr = scheduler_model.get_last_lr()[0]

        # O/0、I/1 中心點距離與兩類準確率（val set，一次遍歷）
        val_emb, val_labels, val_preds = get_embeddings_labels_preds(model, val_loader, device)
        dist_o0, acc_o0 = centroid_distance_and_confusion(val_emb, val_labels, val_preds, 24, 0)
        dist_i1, acc_i1 = centroid_distance_and_confusion(val_emb, val_labels, val_preds, 18, 1)

        alpha_str = f" alpha={alpha:.3f}" if alpha > 0 else " (CE only)"
        print(
            f"Epoch {epoch:2d}{alpha_str}  lr={current_lr:.6f}  "
            f"train acc={train_acc:.4f}  val acc={val_acc:.4f}  "
            f"O/0 dist={dist_o0:.4f} acc={acc_o0:.4f}  I/1 dist={dist_i1:.4f} acc={acc_i1:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, criterion_center, optimizer_model, optimizer_center,
                epoch, val_acc, str(run_dir / "final_precision_model.pt"),
            )
        save_checkpoint(
            model, criterion_center, optimizer_model, optimizer_center,
            epoch, val_acc, str(run_dir / "last_precision_model.pt"),
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "alpha", "O0_dist", "O0_acc", "I1_dist", "I1_acc"])
                file_exists = True
            w.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}", f"{alpha:.4f}", f"{dist_o0:.4f}", f"{acc_o0:.4f}", f"{dist_i1:.4f}", f"{acc_i1:.4f}"])

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not metrics_exists:
                w.writerow(["epoch", "O0_centroid_dist", "O0_pair_acc", "I1_centroid_dist", "I1_pair_acc"])
                metrics_exists = True
            w.writerow([epoch, f"{dist_o0:.4f}", f"{acc_o0:.4f}", f"{dist_i1:.4f}", f"{acc_i1:.4f}"])

    print(f"\nBest val acc: {best_val_acc:.4f}", flush=True)
    print(f"Checkpoints: {run_dir / 'final_precision_model.pt'} (best), {run_dir / 'last_precision_model.pt'}", flush=True)
    print(f"O/0、I/1 中心距離與準確率: {metrics_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
