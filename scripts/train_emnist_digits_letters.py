"""
Phase 1 — Digits + Uppercase Letters (36 classes: 0–9 + A–Z).
Train CNN / ResNet (optional MLP) on clean EMNIST only; no augmentation.
Checkpoints and curves under outputs/runs/<timestamp>_emnist36_<model>/.
Run from project root: python scripts/train_emnist_digits_letters.py --model cnn --epochs 15 --save_best
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

from models.mlp import MLP
from models.cnn import CNN
from models.resnet import ResNet
from scripts.datasets import (
    get_emnist_digits_uppercase_loaders,
    get_emnist36_balanced_loaders,
    EMNIST36_NUM_CLASSES,
)
from scripts.train_utils import train_one_epoch, evaluate, save_checkpoint


SEED = 42


def build_model(model_type, num_classes, dropout=0.0):
    if model_type == "mlp":
        return MLP(input_size=784, hidden_sizes=(256, 128), num_classes=num_classes, dropout=dropout)
    if model_type == "cnn":
        return CNN(num_classes=num_classes, dropout=dropout)
    if model_type == "resnet":
        return ResNet(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model: {model_type}")


def main():
    p = argparse.ArgumentParser(description="EMNIST 36 classes (0–9 + A–Z), Phase 1 clean baseline")
    p.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn", "resnet"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=10, help="MNIST 用 64；EMNIST 36 資料量約 6 倍，預設 64/6≈10")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--augment", action="store_true", help="Optional; Phase 1 typically no augment")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--balanced", action="store_true", help="用平衡抽樣：每類固定張數，與 center loss 訓練相同資料")
    p.add_argument("--samples_per_class_train", type=int, default=5000, help="--balanced 時訓練集每類最多幾筆")
    p.add_argument("--samples_per_class_test", type=int, default=800, help="--balanced 時測試集每類最多幾筆")
    args = p.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    out_dir = Path(args.out_dir or ROOT / "outputs")
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_emnist36_" + args.model
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    if args.balanced:
        train_loader, val_loader, test_loader = get_emnist36_balanced_loaders(
            samples_per_class_train=args.samples_per_class_train,
            samples_per_class_test=args.samples_per_class_test,
            batch_size=args.batch_size,
            num_workers=0,
            augment_train=args.augment,
            seed=SEED,
        )
    else:
        train_loader, val_loader, test_loader = get_emnist_digits_uppercase_loaders(
            batch_size=args.batch_size, num_workers=0, augment_train=args.augment, seed=SEED
        )
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    model = build_model(args.model, EMNIST36_NUM_CLASSES, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    log_path = run_dir / "experiments.csv"
    status_path = out_dir / "emnist36_train_status.txt"
    best_val_acc = 0.0
    file_exists = log_path.exists()

    def write_status(epoch_done, train_acc, val_acc, best_acc, msg=""):
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(status_path, "w", encoding="utf-8") as f:
            f.write(f"batch_size={args.batch_size}  epochs_done={epoch_done}/{args.epochs}  run_dir={run_dir.name}\n")
            f.write(f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  best_val_acc={best_acc:.4f}\n")
            if msg:
                f.write(msg + "\n")

    print("EMNIST Digits + Letters (36 classes: 0–9 + A–Z) — Phase 1 Clean Baseline", flush=True)
    if args.balanced:
        print(f"Data: 平衡抽樣 (train 每類 {args.samples_per_class_train}, test 每類 {args.samples_per_class_test})", flush=True)
    print(f"Model: {args.model.upper()}  |  Train: {train_size}  |  Val: {val_size}", flush=True)
    print(f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, augment={args.augment}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    write_status(0, 0.0, 0.0, 0.0, "status=starting")
    print(flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:2d}  train loss={train_loss:.4f}  train acc={train_acc:.4f}  val loss={val_loss:.4f}  val acc={val_acc:.4f}", flush=True)

        if args.save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_name = f"best_emnist36_{args.model}.pt"
            save_checkpoint(model, optimizer, epoch, val_acc, str(run_dir / ckpt_name))
        save_checkpoint(model, optimizer, epoch, val_acc, str(run_dir / f"last_emnist36_{args.model}.pt"))
        write_status(epoch, train_acc, val_acc, best_val_acc)

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "model", "lr"])
                file_exists = True
            w.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", args.model, args.lr])

    print(f"\nBest val acc: {best_val_acc:.4f}", flush=True)
    print(f"Checkpoints in: {run_dir}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
