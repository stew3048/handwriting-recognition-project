"""
分析五個方法（CNN / 一般 ResNet / ResNet+CL v1,v2,v3）的：
1) 從 .npy 讀取的 2x2 混淆 → 每對兩類整體準確率
2) Embedding 分離度（類心距離/類內散布）→ 是否有改善
輸出：單一報告 + 分離度對照表。Run from project root:
  python scripts/analyze_five_methods_embedding.py --base_dir outputs/emnist36_eval_balanced_5versions [--cpu]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.mlp import MLP
from models.cnn import CNN
from models.resnet import ResNet
from models.vit import ViT
from scripts.datasets import (
    get_emnist36_balanced_loaders,
    EMNIST36_NUM_CLASSES,
    EMNIST36_LABEL_NAMES,
)

PAIRS = [
    ("O_0", (24, 0)), ("I_1", (18, 1)), ("Z_2", (35, 2)), ("S_5", (28, 5)),
    ("A_4", (10, 4)), ("G_6", (16, 6)), ("B_8", (11, 8)),
]
PAIRS_DICT = {name: (idx_a, idx_b) for name, (idx_a, idx_b) in PAIRS}

# 五個方法：(顯示名稱, 資料夾名, checkpoint 路徑, model 類型)
METHODS = [
    ("CNN", "cnn", "outputs/runs/20260215_175247_emnist36_cnn/best_emnist36_cnn.pt", "cnn"),
    ("一般 ResNet", "resnet", "outputs/runs/20260215_175247_emnist36_resnet/best_emnist36_resnet.pt", "resnet"),
    ("ResNet+CL v1 (128d)", "resnet_centerloss_v1", "outputs/runs/20260215_163722_emnist36_centerloss/best_emnist36_centerloss.pt", "resnet_centerloss"),
    ("ResNet+CL v2 (64d)", "resnet_centerloss_v2", "outputs/runs/20260216_005651_emnist36_centerloss/best_emnist36_centerloss.pt", "resnet_centerloss_64"),
    ("ResNet+CL v3 (64d)", "resnet_centerloss_v3", "outputs/runs/20260216_214309_emnist36_centerloss/best_emnist36_centerloss.pt", "resnet_centerloss_64"),
]


def build_model(model_type, device):
    if model_type == "mlp":
        model = MLP(input_size=784, hidden_sizes=(256, 128), num_classes=EMNIST36_NUM_CLASSES, dropout=0.0)
    elif model_type == "cnn":
        model = CNN(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0)
    elif model_type == "resnet":
        model = ResNet(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0)
    elif model_type == "resnet_centerloss":
        model = ResNet(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0, embedding_dim=128)
    elif model_type == "resnet_centerloss_64":
        model = ResNet(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0, embedding_dim=64)
    elif model_type == "vit":
        model = ViT(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    return model.to(device)


def get_embeddings_and_labels(model, test_loader, device):
    model.eval()
    embs, labels_list = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            emb, _ = model.forward_embedding(images)
            embs.append(emb.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labels_list, axis=0)


def centroid(x):
    return np.mean(x, axis=0)


def mean_distance_to_centroid(x, c):
    if len(x) == 0:
        return 0.0
    return np.mean(np.linalg.norm(x - c, axis=1))


def separability_for_pair(embeddings, labels, idx_a, idx_b):
    mask_a = labels == idx_a
    mask_b = labels == idx_b
    emb_a = embeddings[mask_a]
    emb_b = embeddings[mask_b]
    if len(emb_a) == 0 or len(emb_b) == 0:
        return None
    c_a = centroid(emb_a)
    c_b = centroid(emb_b)
    within_a = mean_distance_to_centroid(emb_a, c_a)
    within_b = mean_distance_to_centroid(emb_b, c_b)
    between = float(np.linalg.norm(c_a - c_b))
    avg_within = (within_a + within_b) / 2
    ratio = between / avg_within if avg_within > 1e-9 else 0.0
    return {"separability_ratio": ratio, "between": between, "within_avg": avg_within}


def pair_acc_from_2x2(cm):
    """cm: 2x2, row=真實, col=預測。回傳兩類整體準確率。"""
    total = cm.sum()
    if total == 0:
        return 0.0
    return (cm[0, 0] + cm[1, 1]) / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, default="outputs/emnist36_eval_balanced_5versions")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    base = Path(args.base_dir)
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")

    # ----- 1. 從各方法資料夾讀取 .npy，整理成「每對兩類整體準確率」-----
    # result[method_name][pair_name] = acc_pair
    pair_acc_from_npy = {display: {} for display, _, _, _ in METHODS}
    for display, folder, _, _ in METHODS:
        eval_dir = base / folder
        for name, _ in PAIRS:
            p = eval_dir / f"confusion_2x2_{name}.npy"
            if p.exists():
                cm = np.load(p)
                pair_acc_from_npy[display][name] = pair_acc_from_2x2(cm)
            else:
                pair_acc_from_npy[display][name] = None

    # ----- 2. 對每個方法載入模型、取 embedding、算分離度 -----
    _, _, test_loader = get_emnist36_balanced_loaders(
        samples_per_class_train=5000,
        samples_per_class_test=800,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
    )
    # result[method_name][pair_name] = separability_ratio
    separability = {display: {} for display, _, _, _ in METHODS}
    for display, folder, ckpt_path, model_type in METHODS:
        ckpt_path = ROOT / ckpt_path
        if not ckpt_path.exists():
            print(f"Skip {display}: checkpoint not found {ckpt_path}")
            continue
        model = build_model(model_type, device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        embeddings, labels = get_embeddings_and_labels(model, test_loader, device)
        for name, (idx_a, idx_b) in PAIRS:
            res = separability_for_pair(embeddings, labels, idx_a, idx_b)
            separability[display][name] = res["separability_ratio"] if res else None

    # ----- 3. 寫報告 -----
    lines = []
    lines.append("# 五個方法：從 .npy 混淆與 Embedding 分離度分析\n")
    lines.append("評估資料：平衡 test（每類 800），目錄 `" + str(base) + "`。\n")

    # 表一：從 .npy 的每對兩類整體準確率（%）
    lines.append("## 1. 從 .npy 讀取的易混字對「兩類整體準確率」\n")
    headers = ["字對"] + [m[0] for m in METHODS]
    lines.append("| " + " | ".join(headers) + " |\n")
    lines.append("|" + "|".join(["------"] + ["--------"] * len(METHODS)) + "|\n")
    for name, _ in PAIRS:
        row = [name]
        for display, _, _, _ in METHODS:
            v = pair_acc_from_npy[display].get(name)
            row.append(f"{v*100:.2f}%" if v is not None else "-")
        lines.append("| " + " | ".join(row) + " |\n")

    # 表二：Embedding 分離度（越大越拉開）
    lines.append("\n## 2. Embedding 分離度（類心距離 / 平均類內散布）\n")
    lines.append("數值越大表示該字對在 embedding 空間越拉開。\n")
    lines.append("| " + " | ".join(headers) + " |\n")
    lines.append("|" + "|".join(["------"] + ["--------"] * len(METHODS)) + "|\n")
    for name, _ in PAIRS:
        row = [name]
        for display, _, _, _ in METHODS:
            v = separability[display].get(name)
            row.append(f"{v:.3f}" if v is not None else "-")
        lines.append("| " + " | ".join(row) + " |\n")

    # 簡短結論：誰在 O_0 / I_1 分離度最高、是否隨版本改善
    lines.append("\n## 3. 簡要結論（Embedding 是否有改善）\n")
    for name in ["O_0", "I_1"]:
        vals = [(display, separability[display].get(name)) for display, _, _, _ in METHODS]
        vals = [(d, v) for d, v in vals if v is not None]
        if not vals:
            continue
        best = max(vals, key=lambda x: x[1])
        lines.append(f"- **{name}**：分離度最高為 **{best[0]}**（{best[1]:.3f}）。")
        order = sorted(vals, key=lambda x: x[1], reverse=True)
        order_names = [d for d, _ in order]
        lines.append(f"  由高到低：{' → '.join(order_names)}。\n")
    # 整體：v3 是否在大多數 pair 上最好
    lines.append("- **ResNet+CL 三版**：v1(128d) → v2(64d) → v3(64d) 在易混字對上，分離度是否隨版本提升，見上表。\n")
    report_path = base / "embedding_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"報告已寫入: {report_path}")
    print("\n" + "".join(lines))


if __name__ == "__main__":
    main()
