"""
MNIST + ResNet。支援 CLI 超參、run 資料夾、checkpoint、曲線圖、overfitting 判斷。
- 每 epoch 印出 Train/Val Loss、Acc（對齊 sky segmentation 訓練日誌）。
- 偵測中斷：若存在未跑完的 run（同前綴 *_mnist_resnet），則自動續訓。
從專案根目錄執行：python scripts/train_mnist_resnet.py --epochs 10 --save_best
"""
import argparse
import csv
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.resnet import ResNet
from scripts.datasets import get_mnist_loaders
from scripts.train_utils import train_one_epoch, evaluate, save_checkpoint, load_checkpoint


def plot_curves(history, save_path, overfitting_msg=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="train loss")
    ax1.plot(epochs, history["val_loss"], label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")
    ax2.plot(epochs, history["train_acc"], label="train acc")
    ax2.plot(epochs, history["val_acc"], label="val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Accuracy")
    if overfitting_msg:
        wrapped = textwrap.fill(f"Overfitting: {overfitting_msg}", width=80)
        fig.text(0.5, 0.02, wrapped, ha="center", fontsize=9, transform=fig.transFigure)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def judge_overfitting(history):
    tl, vl = history["train_loss"], history["val_loss"]
    ta, va = history["train_acc"], history["val_acc"]
    last = -1
    msg = []
    if len(tl) < 2:
        return "Too few epochs to judge."
    mid = len(vl) // 2
    if vl[last] > vl[mid] and tl[last] < tl[mid]:
        msg.append("Val loss rises in 2nd half while train loss keeps falling -> overfitting likely.")
    if len(vl) >= 3 and vl[-1] > vl[-2] and vl[-1] > vl[-3]:
        msg.append("Val loss went up in last few epochs -> possible overfitting.")
    gap = ta[last] - va[last]
    if gap > 0.05:
        msg.append(f"Train acc >> val acc (gap {gap:.2%}) -> overfitting.")
    elif gap > 0.02:
        msg.append(f"Train acc slightly above val acc (gap {gap:.2%}) -> mild overfitting.")
    if not msg:
        msg.append("Train/val curves close -> no obvious overfitting.")
    return " ".join(msg)


def _find_incomplete_resnet_run(out_dir, epochs):
    """找最新且未完成（completed_epochs < epochs）的 *_mnist_resnet run。"""
    runs_dir = out_dir / "runs"
    if not runs_dir.exists():
        return None, 0
    candidates = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.endswith("_mnist_resnet")]
    if not candidates:
        return None, 0
    candidates.sort(key=os.path.getmtime, reverse=True)
    for run_dir in candidates:
        csv_path = run_dir / "experiments_resnet.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        completed = len(rows)
        if 0 <= completed < epochs:
            return run_dir, completed
    return None, 0


def main():
    p = argparse.ArgumentParser(description="MNIST ResNet")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--run_name", type=str, default=None, help="指定 run 資料夾名；不指定則自動找未完成 run 續訓或開新 run")
    args = p.parse_args()

    out_dir = Path(args.out_dir or ROOT / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    use_existing = False
    start_epoch = 1
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    run_dir = None

    if args.run_name:
        run_dir = runs_dir / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "experiments_resnet.csv"
        if csv_path.exists():
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            completed = len(rows)
            if 0 <= completed < args.epochs:
                use_existing = True
                start_epoch = completed + 1
                best_val_acc = max(float(r["val_acc"]) for r in rows) if rows else 0.0
                for r in rows:
                    history["train_loss"].append(float(r["train_loss"]))
                    history["train_acc"].append(float(r["train_acc"]))
                    history["val_loss"].append(float(r["val_loss"]))
                    history["val_acc"].append(float(r["val_acc"]))
    else:
        existing_run, completed = _find_incomplete_resnet_run(out_dir, args.epochs)
        if existing_run is not None:
            run_dir = existing_run
            use_existing = True
            start_epoch = completed + 1
            csv_path = run_dir / "experiments_resnet.csv"
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                best_val_acc = max(float(r["val_acc"]) for r in rows)
                for r in rows:
                    history["train_loss"].append(float(r["train_loss"]))
                    history["train_acc"].append(float(r["train_acc"]))
                    history["val_loss"].append(float(r["val_loss"]))
                    history["val_acc"].append(float(r["val_acc"]))
            print("[繼續訓練] 偵測到未完成的 run，將從上次中斷處續訓", flush=True)
            print(f"  目錄: {run_dir.name}", flush=True)
            print(f"  已完成: {completed}/{args.epochs} epochs", flush=True)
            print()
        else:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_mnist_resnet"
            run_dir = runs_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            print("[新訓練] 建立新 run", flush=True)
            print(f"  目錄: {run_dir.name}", flush=True)
            print()

    log_path = run_dir / "experiments_resnet.csv"

    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_mnist_loaders(
        batch_size=args.batch_size, num_workers=0, augment_train=args.augment
    )
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    model = ResNet(num_classes=10, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    if use_existing:
        last_ckpt = run_dir / "last_mnist_resnet.pt"
        ckpt = load_checkpoint(str(last_ckpt), device)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print(f"[載入檢查點] {last_ckpt.name}  (Epoch {ckpt.get('epoch', '?')})", flush=True)
            print(f"  目前最佳 Val Acc: {best_val_acc:.4f}", flush=True)
            print()
        else:
            print("[繼續訓練] 未找到 last_mnist_resnet.pt，從頭開始訓練權重", flush=True)
            print()

    print("MNIST + ResNet")
    print(f"Train data size: {train_size}  |  Val data size: {val_size}")
    print(f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, augment={args.augment}")
    print(f"Checkpoint / curves 存於: {run_dir}")
    print()
    print("=" * 60)
    print("開始訓練...")
    print("=" * 60)
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>8}")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  {epoch:>5} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>10.4f} | {val_acc:>8.4f}", flush=True)

        if args.save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, str(run_dir / "best_mnist_resnet.pt"))
        save_checkpoint(model, optimizer, epoch, val_acc, str(run_dir / "last_mnist_resnet.pt"))

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if epoch == 1 and (not use_existing or start_epoch == 1):
                w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "augment"])
            w.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", args.lr, args.augment])

    print("-" * 60)
    print("=" * 60)
    print("訓練完成！")
    print("=" * 60)
    print(f"  最佳 Val Acc: {best_val_acc:.4f}")
    print(f"  檢查點目錄: {run_dir}")
    print()

    overfitting_msg = judge_overfitting(history)
    plot_curves(history, run_dir / "train_curves.png", overfitting_msg=overfitting_msg)
    print("--- Overfitting ---")
    print(overfitting_msg)
    print("\nDone.")


if __name__ == "__main__":
    main()
