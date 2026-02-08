"""
Phase 4：英文字母手寫辨識 (EMNIST Letters, 26 類)。可選 MLP 或 CNN。
從專案根目錄執行：python scripts/train_emnist_letters.py --model cnn --epochs 1
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.mlp import MLP
from models.cnn import CNN
from scripts.datasets import get_emnist_letters_loaders
from scripts.train_utils import train_one_epoch, evaluate, save_checkpoint

# EMNIST Letters 影像為 28x28，與 MNIST 同
INPUT_SIZE = 784
NUM_CLASSES = 26


def main():
    p = argparse.ArgumentParser(description="EMNIST Letters (A-Z)")
    p.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--cpu", action="store_true", help="強制使用 CPU（避開 CUDA/cuDNN 問題）")
    args = p.parse_args()

    out_dir = Path(args.out_dir or ROOT / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_emnist_letters_loaders(
        batch_size=args.batch_size, num_workers=0, augment_train=args.augment
    )

    if args.model == "mlp":
        model = MLP(input_size=INPUT_SIZE, hidden_sizes=(256, 128), num_classes=NUM_CLASSES, dropout=args.dropout).to(device)
    else:
        model = CNN(num_classes=NUM_CLASSES, dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    print(f"EMNIST Letters + {args.model.upper()}")
    print(f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}\n")

    best_val_acc = 0.0
    log_path = out_dir / f"experiments_emnist_{args.model}.csv"
    file_exists = log_path.exists()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:2d}  train loss={train_loss:.4f}  train acc={train_acc:.4f}  val loss={val_loss:.4f}  val acc={val_acc:.4f}")

        if args.save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, str(out_dir / f"best_emnist_{args.model}.pt"))
        elif not args.save_best:
            save_checkpoint(model, optimizer, epoch, val_acc, str(out_dir / f"last_emnist_{args.model}.pt"))

        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "model", "lr"])
                file_exists = True
            w.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", args.model, args.lr])

    print("\nDone.")


if __name__ == "__main__":
    main()
