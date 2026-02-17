"""
只畫 O vs 0 的 embedding 2D（PCA），五個方法各一張，方便肉眼比對「誰真的拉開、誰混在一起」。
與分離度數字可能不同：數字是高維的類心/散布比，圖是 2D 投影。
Run from project root:
  python scripts/plot_o0_embedding_compare_5methods.py --base_dir outputs/emnist36_eval_balanced_5versions [--cpu]
"""
import argparse
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

from models.cnn import CNN
from models.resnet import ResNet
from scripts.datasets import get_emnist36_balanced_loaders, EMNIST36_NUM_CLASSES

IDX_O, IDX_0 = 24, 0
METHODS = [
    ("CNN", "outputs/runs/20260215_175247_emnist36_cnn/best_emnist36_cnn.pt", "cnn"),
    ("ResNet", "outputs/runs/20260215_175247_emnist36_resnet/best_emnist36_resnet.pt", "resnet"),
    ("ResNet+CL v1 (128d)", "outputs/runs/20260215_163722_emnist36_centerloss/best_emnist36_centerloss.pt", "resnet_centerloss"),
    ("ResNet+CL v2 (64d)", "outputs/runs/20260216_005651_emnist36_centerloss/best_emnist36_centerloss.pt", "resnet_centerloss_64"),
    ("ResNet+CL v3 (64d)", "outputs/runs/20260216_214309_emnist36_centerloss/best_emnist36_centerloss.pt", "resnet_centerloss_64"),
]


def build_model(model_type, device):
    if model_type == "cnn":
        return CNN(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0).to(device)
    if model_type == "resnet":
        return ResNet(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0).to(device)
    if model_type == "resnet_centerloss":
        return ResNet(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0, embedding_dim=128).to(device)
    if model_type == "resnet_centerloss_64":
        return ResNet(num_classes=EMNIST36_NUM_CLASSES, dropout=0.0, embedding_dim=64).to(device)
    raise ValueError(model_type)


def get_embeddings(model, test_loader, device):
    model.eval()
    embs, labels_list = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            emb, _ = model.forward_embedding(images)
            embs.append(emb.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labels_list, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, default="outputs/emnist36_eval_balanced_5versions")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    base = Path(args.base_dir)
    out_dir = base / "o0_embedding_2d_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_emnist36_balanced_loaders(
        samples_per_class_train=5000,
        samples_per_class_test=800,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
    )

    for display_name, ckpt_path, model_type in METHODS:
        ckpt_path = ROOT / ckpt_path
        if not ckpt_path.exists():
            print(f"Skip {display_name}: no checkpoint")
            continue
        model = build_model(model_type, device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        embeddings, labels = get_embeddings(model, test_loader, device)

        mask_o = labels == IDX_O
        mask_0 = labels == IDX_0
        emb_o = embeddings[mask_o]
        emb_0 = embeddings[mask_0]
        if len(emb_o) == 0 or len(emb_0) == 0:
            continue
        all_emb = np.vstack([emb_o, emb_0])
        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(all_emb)
        n_o = len(emb_o)
        x_o, y_o = xy[:n_o, 0], xy[:n_o, 1]
        x_0, y_0 = xy[n_o:, 0], xy[n_o:, 1]
        c_o = np.mean(emb_o, axis=0)
        c_0 = np.mean(emb_0, axis=0)
        c_xy = pca.transform(np.vstack([c_o.reshape(1, -1), c_0.reshape(1, -1)]))

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(x_o, y_o, alpha=0.5, s=14, label="O (letter)", c="C0")
        ax.scatter(x_0, y_0, alpha=0.5, s=14, label="0 (digit)", c="C1")
        ax.scatter(c_xy[0, 0], c_xy[0, 1], marker="*", s=280, c="C0", edgecolors="black", linewidths=1, label="O centroid")
        ax.scatter(c_xy[1, 0], c_xy[1, 1], marker="*", s=280, c="C1", edgecolors="black", linewidths=1, label="0 centroid")
        ax.set_title(f"O vs 0 — {display_name}\n(2D PCA on this pair only)")
        ax.legend()
        ax.set_aspect("equal")
        safe_name = display_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        plt.savefig(out_dir / f"O0_{safe_name}.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_dir / f'O0_{safe_name}.png'}")

    print(f"All O vs 0 2D plots in: {out_dir}")


if __name__ == "__main__":
    main()
