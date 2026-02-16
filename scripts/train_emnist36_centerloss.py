"""
EMNIST 36 類 + 平衡抽樣 + ResNet 64 維 embedding（不升維）+ Center Loss。
- 模型：Backbone 展平 64 維即 embedding，Linear(64, 36) 得 logits。
- 延遲 Center Loss：前 5 個 epoch 僅 CE（alpha=0），第 6-10 epoch alpha=0.05，第 11-20 epoch 從 0.05 漸進到 0.1。
- Learning rate：使用 CosineAnnealingLR scheduler。
- CenterLoss feat_dim=64；optimizer_center 僅在 alpha>0 時 step。
評估：訓練結束後用 evaluate_emnist_digits_letters.py --model resnet_centerloss_64 產出混淆矩陣與
  metrics.json（含 O/0 等），可與先前 ResNet+128 維升維層的結果比較 Test Accuracy 與 O vs 0 誤判次數。
小量驗證：--samples_per_class_train 5 --samples_per_class_test 5 --epochs 2
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
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.resnet import ResNet
from scripts.center_loss import CenterLoss
from scripts.datasets import get_emnist36_balanced_loaders, EMNIST36_NUM_CLASSES

SEED = 42
NUM_CLASSES = 36
EMBEDDING_DIM = 64
WARMUP_EPOCHS = 5  # 前 5 個 epoch 僅 CE
ALPHA_STABLE_EPOCHS = 10  # 第 6-10 epoch alpha=0.05
ALPHA_START = 0.05  # 第 6-10 epoch 的 alpha
ALPHA_END = 0.1  # 第 20 epoch 的 alpha（漸進式增強）


def get_alpha(epoch, warmup_epochs=WARMUP_EPOCHS, alpha_stable_epochs=ALPHA_STABLE_EPOCHS, alpha_start=ALPHA_START, alpha_end=ALPHA_END):
    """
    計算當前 epoch 的 Center Loss alpha：
    - epoch 1-5: alpha=0（僅 CE）
    - epoch 6-10: alpha=0.05（穩定）
    - epoch 11-20: alpha 從 0.05 線性增加到 0.1
    """
    if epoch <= warmup_epochs:
        return 0.0
    elif epoch <= alpha_stable_epochs:
        return alpha_start
    else:
        # epoch 11-20: 線性插值從 alpha_start 到 alpha_end
        progress = (epoch - alpha_stable_epochs) / (20 - alpha_stable_epochs)  # 0.0 -> 1.0
        return alpha_start + progress * (alpha_end - alpha_start)


def train_one_epoch(model, train_loader, criterion_ce, criterion_center, optimizer_model, optimizer_center, device, alpha):
    """alpha: Center Loss 權重，0 表示僅 CE；僅當 alpha > 0 時 backward center loss 並 step optimizer_center。"""
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
    """回傳目前 ResNet + embedding 架構的文字摘要（64 維不升維）。"""
    lines = [
        "Input: (B, 1, 28, 28) 灰階",
        "Stem: Conv2d(1,16,3,1,1) -> BN -> ReLU -> MaxPool2d(2)  => (B, 16, 14, 14)",
        "Layer1: 2x BasicBlock(16->16)  => (B, 16, 14, 14)",
        "Layer2: 2x BasicBlock(16->32, stride=2)  => (B, 32, 7, 7)",
        "Layer3: 2x BasicBlock(32->64, stride=2)  => (B, 64, 3, 3)",
        "AdaptiveAvgPool2d(1)  => (B, 64, 1, 1)",
        "Flatten -> Dropout  => (B, 64) [embedding，不升維]",
        f"fc: Linear(64, {num_classes})  => (B, {num_classes}) [logits]",
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
    p.add_argument("--epochs", type=int, default=20, help="預設 20；小量驗證可改 --epochs 2")
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
    # CosineAnnealingLR scheduler for model optimizer
    scheduler_model = CosineAnnealingLR(optimizer_model, T_max=args.epochs, eta_min=1e-5)

    # 印出目前 model 架構摘要
    print("=== Model 架構 (ResNet + 64-dim embedding，不升維) ===", flush=True)
    print(_model_architecture_summary(model, num_classes=NUM_CLASSES), flush=True)
    print(f"=== Center Loss: num_classes=36, feat_dim={EMBEDDING_DIM} ===", flush=True)
    print(f"=== 延遲 Center Loss: 前 {WARMUP_EPOCHS} 個 epoch 僅 CE，第 6-10 epoch alpha={ALPHA_START}，第 11-20 epoch 漸進到 {ALPHA_END} ===", flush=True)
    print(f"=== Learning Rate: CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-5) ===", flush=True)
    print(flush=True)

    log_path = run_dir / "experiments.csv"
    best_val_acc = 0.0
    file_exists = log_path.exists()

    print("EMNIST 36 balanced + ResNet(64-dim embedding, no projection) + Center Loss (delayed)", flush=True)
    if args.samples_per_class_train <= 10 or args.samples_per_class_test <= 10:
        print("(小量抽樣模式：每類樣本數 <= 10，僅供流程驗證)", flush=True)
    print(f"Train size: {train_size}  |  Val size: {val_size}", flush=True)
    print(f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}", flush=True)
    print(f"lr_model={args.lr_model} (CosineAnnealing), lr_center={args.lr_center}, warmup_epochs={WARMUP_EPOCHS}, alpha={ALPHA_START}->{ALPHA_END}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(flush=True)

    for epoch in range(1, args.epochs + 1):
        alpha = get_alpha(epoch, WARMUP_EPOCHS, ALPHA_STABLE_EPOCHS, ALPHA_START, ALPHA_END)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion_ce, criterion_center,
            optimizer_model, optimizer_center, device, alpha=alpha,
        )
        val_loss, val_acc = evaluate_val(model, val_loader, criterion_ce, device)
        # Step scheduler after each epoch
        scheduler_model.step()
        current_lr = scheduler_model.get_last_lr()[0]
        alpha_str = f" alpha={alpha:.3f}" if alpha > 0 else " (CE only)"
        print(f"Epoch {epoch:2d}{alpha_str}  lr={current_lr:.6f}  train loss={train_loss:.4f}  train acc={train_acc:.4f}  val loss={val_loss:.4f}  val acc={val_acc:.4f}", flush=True)

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
                w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr_model", "lr_center", "alpha"])
                file_exists = True
            w.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}", args.lr_center, f"{alpha:.4f}"])

    print(f"\nBest val acc: {best_val_acc:.4f}", flush=True)
    print(f"Checkpoints in: {run_dir}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
