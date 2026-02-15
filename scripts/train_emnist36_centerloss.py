"""
EMNIST 36 類 + 平衡抽樣 + ResNet(embedding_dim=128) + Center Loss。
使用 get_emnist36_balanced_loaders(5000, 800)，雙優化器：Adam(model, 1e-3)、Adam(center_loss, 0.5)，
total_loss = CrossEntropy(logits, labels) + 0.1 * CenterLoss(embeddings, labels)。
Checkpoint 含 model 與 criterion_center 的 state_dict。

小量抽樣快速驗證：--samples_per_class_train 5 --samples_per_class_test 5 --epochs 2
（每類只抽 5 張，可快速確認流程是否正常）
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.resnet import ResNet
from scripts.center_loss import CenterLoss
from scripts.datasets import get_emnist36_balanced_loaders, EMNIST36_NUM_CLASSES

SEED = 42
NUM_CLASSES = 36
EMBEDDING_DIM = 128
CENTER_LOSS_WEIGHT = 0.1


def train_one_epoch(model, train_loader, criterion_ce, criterion_center, optimizer_model, optimizer_center, device):
    model.train()
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
        loss_center = criterion_center(embeddings, y)
        loss = loss_ce + CENTER_LOSS_WEIGHT * loss_center
        loss.backward()
        optimizer_model.step()
        optimizer_center.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate_val(model, loader, criterion_ce, device):
    """Validation：只算 CE loss 與 accuracy，用 logits 即可。"""
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


def _model_architecture_summary(model, num_classes=36):
    """回傳目前 ResNet + embedding 架構的文字摘要。"""
    lines = [
        "Input: (B, 1, 28, 28) 灰階",
        "Stem: Conv2d(1,16,3,1,1) -> BN -> ReLU -> MaxPool2d(2)  => (B, 16, 14, 14)",
        "Layer1: 2x BasicBlock(16->16)  => (B, 16, 14, 14)",
        "Layer2: 2x BasicBlock(16->32, stride=2)  => (B, 32, 7, 7)",
        "Layer3: 2x BasicBlock(32->64, stride=2)  => (B, 64, 3, 3)",
        "AdaptiveAvgPool2d(1)  => (B, 64, 1, 1)",
        "Flatten -> Dropout  => (B, 64)",
        "feat: Linear(64, 128) -> BN -> ReLU  => (B, 128) [embedding]",
        f"fc: Linear(128, {num_classes})  => (B, {num_classes}) [logits]",
        "Output: (logits, embeddings)",
    ]
    return "\n".join(lines)


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
    p = argparse.ArgumentParser(description="EMNIST 36 balanced + ResNet 128-dim embedding + Center Loss")
    p.add_argument("--samples_per_class_train", type=int, default=5,
                   help="訓練集每類最多幾筆；預設 5 小量驗證，正式訓練請設 5000")
    p.add_argument("--samples_per_class_test", type=int, default=5,
                   help="測試集每類最多幾筆；預設 5 小量驗證，正式訓練請設 800")
    p.add_argument("--epochs", type=int, default=2, help="預設 2；正式訓練可改 --epochs 15")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_model", type=float, default=1e-3)
    p.add_argument("--lr_center", type=float, default=0.5)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    out_dir = Path(args.out_dir or ROOT / "outputs")
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_emnist36_centerloss"
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_emnist36_balanced_loaders(
        samples_per_class_train=args.samples_per_class_train,
        samples_per_class_test=args.samples_per_class_test,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        augment_train=args.augment,
        num_workers=0,
        seed=SEED,
    )
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    model = ResNet(num_classes=NUM_CLASSES, dropout=0.0, embedding_dim=EMBEDDING_DIM).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_center = CenterLoss(num_classes=NUM_CLASSES, feat_dim=EMBEDDING_DIM).to(device)
    optimizer_model = Adam(model.parameters(), lr=args.lr_model)
    optimizer_center = Adam(criterion_center.parameters(), lr=args.lr_center)

    # 印出目前 model 架構摘要
    print("=== Model 架構 (ResNet + 128-dim embedding) ===", flush=True)
    print(_model_architecture_summary(model, num_classes=NUM_CLASSES), flush=True)
    print("=== Center Loss: num_classes=36, feat_dim=128 ===", flush=True)
    print(flush=True)

    log_path = run_dir / "experiments.csv"
    best_val_acc = 0.0
    file_exists = log_path.exists()

    print("EMNIST 36 balanced + ResNet(embedding_dim=128) + Center Loss", flush=True)
    if args.samples_per_class_train <= 10 or args.samples_per_class_test <= 10:
        print("(小量抽樣模式：每類樣本數 <= 10，僅供流程驗證)", flush=True)
    print(f"Train size: {train_size}  |  Val size: {val_size}", flush=True)
    print(f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}", flush=True)
    print(f"lr_model={args.lr_model}, lr_center={args.lr_center}, center_weight={CENTER_LOSS_WEIGHT}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion_ce, criterion_center,
            optimizer_model, optimizer_center, device,
        )
        val_loss, val_acc = evaluate_val(model, val_loader, criterion_ce, device)
        print(f"Epoch {epoch:2d}  train loss={train_loss:.4f}  train acc={train_acc:.4f}  val loss={val_loss:.4f}  val acc={val_acc:.4f}", flush=True)

        if args.save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, criterion_center, optimizer_model, optimizer_center,
                epoch, val_acc, str(run_dir / "best_emnist36_centerloss.pt"),
            )
        save_checkpoint(
            model, criterion_center, optimizer_model, optimizer_center,
            epoch, val_acc, str(run_dir / "last_emnist36_centerloss.pt"),
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr_model", "lr_center"])
                file_exists = True
            w.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", args.lr_model, args.lr_center])

    print(f"\nBest val acc: {best_val_acc:.4f}", flush=True)
    print(f"Checkpoints in: {run_dir}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
