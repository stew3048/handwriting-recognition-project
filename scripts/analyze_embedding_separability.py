"""
分析易混字對在 embedding 空間是否拉開：讀取 eval 目錄的 .npy 混淆矩陣，
並用 checkpoint 取出 test set embedding，計算每對的類心距離、類內散布、分離度。
輸出：報告 Markdown + 每對的 2D 散點圖（僅該兩類）。
使用方式（專案根目錄）：
  python scripts/analyze_embedding_separability.py --eval_dir outputs/runs/20260216_214309_emnist36_centerloss/eval --checkpoint outputs/runs/20260216_214309_emnist36_centerloss/best_emnist36_centerloss.pt --model resnet_centerloss_64 [--cpu]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.resnet import ResNet
from scripts.datasets import (
    get_emnist36_balanced_loaders,
    EMNIST36_NUM_CLASSES,
    EMNIST36_LABEL_NAMES,
)

# 易混字對：(名稱, (class_idx_a, class_idx_b))，與 evaluate 腳本一致
PAIRS = [
    ("O_0", (24, 0)),   # O, 0
    ("I_1", (18, 1)),   # I, 1
    ("Z_2", (35, 2)),   # Z, 2
    ("S_5", (28, 5)),   # S, 5
    ("A_4", (10, 4)),   # A, 4
    ("G_6", (16, 6)),   # G, 6
    ("B_8", (11, 8)),   # B, 8
]


def load_confusion_npy(eval_dir):
    """載入 eval 目錄下所有 confusion_2x2_*.npy 與 confusion_matrix.npy，回傳 dict 與 36x36 matrix。"""
    eval_dir = Path(eval_dir)
    out = {}
    for name, _ in PAIRS:
        p = eval_dir / f"confusion_2x2_{name}.npy"
        if p.exists():
            out[name] = np.load(p)
    cm_full = None
    p_full = eval_dir / "confusion_matrix.npy"
    if p_full.exists():
        cm_full = np.load(p_full)
    return out, cm_full


def summarize_2x2(name, cm, label_names):
    """cm 為 2x2，順序為 [class_a, class_b]：row=真實、col=預測。"""
    a_name, b_name = label_names[PAIRS_DICT[name][0]], label_names[PAIRS_DICT[name][1]]
    # row0=真實A, row1=真實B；col0=預測A, col1=預測B
    acc_a = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    acc_b = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    total = cm.sum()
    acc_pair = (cm[0, 0] + cm[1, 1]) / total if total > 0 else 0
    return {
        "name": name,
        "a": a_name, "b": b_name,
        "cm": cm,
        "acc_a": acc_a, "acc_b": acc_b, "acc_pair": acc_pair,
        "total": total,
    }


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
    # 分離度：類心距離 / 平均類內散布，越大表示越拉開
    ratio = between / avg_within if avg_within > 1e-9 else 0.0
    return {
        "n_a": len(emb_a), "n_b": len(emb_b),
        "within_a": within_a, "within_b": within_b,
        "between": between,
        "separability_ratio": ratio,
        "emb_a": emb_a, "emb_b": emb_b,
        "c_a": c_a, "c_b": c_b,
    }


# 建 name -> (idx_a, idx_b) 方便用
PAIRS_DICT = {name: (idx_a, idx_b) for name, (idx_a, idx_b) in PAIRS}


def plot_pair_2d(emb_a, emb_b, c_a, c_b, path, name, label_a, label_b):
    """兩類的 embedding 做 PCA 到 2D 畫散點，看有沒有拉開。"""
    all_emb = np.vstack([emb_a, emb_b])
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(all_emb)
    n_a = len(emb_a)
    x_a, y_a = xy[:n_a, 0], xy[:n_a, 1]
    x_b, y_b = xy[n_a:, 0], xy[n_a:, 1]
    c_xy = pca.transform(np.vstack([c_a.reshape(1, -1), c_b.reshape(1, -1)]))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_a, y_a, alpha=0.5, s=12, label=label_a, c="C0")
    ax.scatter(x_b, y_b, alpha=0.5, s=12, label=label_b, c="C1")
    ax.scatter(c_xy[0, 0], c_xy[0, 1], marker="*", s=300, c="C0", edgecolors="black", linewidths=1, label=f"{label_a} centroid")
    ax.scatter(c_xy[1, 0], c_xy[1, 1], marker="*", s=300, c="C1", edgecolors="black", linewidths=1, label=f"{label_b} centroid")
    ax.set_title(f"Embedding 2D (PCA) {name}: {label_a} vs {label_b}")
    ax.legend()
    ax.set_aspect("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="分析 .npy 混淆與 embedding 易混字對分離度")
    p.add_argument("--eval_dir", type=str, required=True, help="evaluate 輸出目錄（含 .npy）")
    p.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint .pt")
    p.add_argument("--model", type=str, default="resnet_centerloss_64", choices=["resnet_centerloss_64", "resnet_centerloss"])
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1. 讀取 .npy 混淆矩陣並整理 -----
    pairs_cm, cm_full = load_confusion_npy(eval_dir)
    lines = []
    lines.append("# 易混字對分析：混淆矩陣與 Embedding 分離度\n")
    lines.append("## 1. 從 .npy 讀取的 2×2 混淆矩陣\n")
    for name in [x[0] for x in PAIRS]:
        if name not in pairs_cm:
            continue
        s = summarize_2x2(name, pairs_cm[name], EMNIST36_LABEL_NAMES)
        a, b = s["a"], s["b"]
        cm = s["cm"]
        lines.append(f"### {name}（{a} vs {b}）\n")
        lines.append(f"- 混淆矩陣 (row=真實, col=預測)：\n")
        lines.append(f"  - 真實{a} → 預測{a}: {cm[0,0]}, 預測{b}: {cm[0,1]}\n")
        lines.append(f"  - 真實{b} → 預測{a}: {cm[1,0]}, 預測{b}: {cm[1,1]}\n")
        lines.append(f"- {a} 準確率: {s['acc_a']:.2%}  |  {b} 準確率: {s['acc_b']:.2%}  |  兩類整體: {s['acc_pair']:.2%}\n")
        lines.append("")

    # ----- 2. 載入模型與 test set embedding -----
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0, embedding_dim=64 if "64" in args.model else 128)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _, _, test_loader = get_emnist36_balanced_loaders(
        samples_per_class_train=5000,
        samples_per_class_test=800,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
    )
    embeddings, labels = get_embeddings_and_labels(model, test_loader, device)

    # ----- 3. 每對分離度 + 2D 圖 -----
    lines.append("## 2. Embedding 空間分離度（類心距離 vs 類內散布）\n")
    lines.append("| 字對 | 類A 樣本數 | 類B 樣本數 | 類內散布(A) | 類內散布(B) | 類心距離 | 分離度(距離/平均散布) |\n")
    lines.append("|------|------------|------------|-------------|-------------|----------|------------------------|\n")

    for name, (idx_a, idx_b) in PAIRS:
        res = separability_for_pair(embeddings, labels, idx_a, idx_b)
        if res is None:
            lines.append(f"| {name} | - | - | - | - | - | - |\n")
            continue
        a_name = EMNIST36_LABEL_NAMES[idx_a]
        b_name = EMNIST36_LABEL_NAMES[idx_b]
        ratio = res["separability_ratio"]
        lines.append(
            f"| {name} | {res['n_a']} | {res['n_b']} | {res['within_a']:.3f} | {res['within_b']:.3f} | {res['between']:.3f} | **{ratio:.3f}** |\n"
        )
        # 畫該對的 2D PCA
        plot_pair_2d(
            res["emb_a"], res["emb_b"], res["c_a"], res["c_b"],
            eval_dir / f"embedding_pair_{name}.png",
            name, a_name, b_name,
        )

    lines.append("\n- **分離度** = 類心距離 / 平均類內散布；數值越大表示兩類在 embedding 空間越拉開。\n")

    # 寫出報告
    report_path = eval_dir / "embedding_separability_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"報告已寫入: {report_path}")
    print("每對 2D 圖: embedding_pair_<O_0|I_1|...>.png")
    # 同時印一份到 stdout
    print("\n" + "".join(lines))


if __name__ == "__main__":
    main()
