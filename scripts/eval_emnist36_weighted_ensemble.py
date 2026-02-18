"""
加權集成模型 (Weighted Ensemble) — EMNIST 36 類

使用與 V3 / 溫和 / V6 評估時「同一份」test（每類 800 筆，get_emnist36_test_loader_raw），
載入三個 .pt 做加權軟投票，輸出整體 / 魔王類別準確率與混淆矩陣。

- 前處理：V3 / V6 在腳本內做 Normalize((0.5,), (0.5,))；Mild 僅 ToTensor。
- 權重：V3=0.4, V6=0.4, Mild=0.2。
- 請在專案根目錄執行：C:\\Users\\yiching\\handwriting-recognition-project
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.resnet import ResNet
from scripts.datasets import get_emnist36_test_loader_raw

NUM_CLASSES = 36
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEMON_PAIRS = [
    ("O", "0", 24, 0),
    ("I", "1", 18, 1),
    ("S", "5", 28, 5),
    ("Z", "2", 35, 2),
    ("A", "4", 10, 4),
    ("G", "6", 16, 6),
    ("B", "8", 11, 8),
]


def _get_state_dict(path_ckpt: str):
    ckpt = torch.load(path_ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt
    return ckpt


def load_model(ckpt_path: str, use_normalize: bool):
    model = ResNet(num_classes=NUM_CLASSES, embedding_dim=64)
    state = _get_state_dict(ckpt_path)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, use_normalize


def _logits_from_model(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def run_ensemble_inference(models_config, loader, device, weights):
    all_labels = []
    probs_v3, probs_v6, probs_mild = [], [], []
    m_v3, norm_v3 = models_config[0]
    m_v6, norm_v6 = models_config[1]
    m_mild, norm_mild = models_config[2]
    m_v3, m_v6, m_mild = m_v3.to(device), m_v6.to(device), m_mild.to(device)
    with torch.no_grad():
        for x, y in loader:
            all_labels.append(y.numpy())
            x_gpu = x.to(device)
            x_n = (x_gpu - 0.5) / 0.5
            p3 = torch.softmax(_logits_from_model(m_v3, x_n), dim=1).cpu().numpy()
            p6 = torch.softmax(_logits_from_model(m_v6, x_n), dim=1).cpu().numpy()
            pm = torch.softmax(_logits_from_model(m_mild, x_gpu), dim=1).cpu().numpy()
            probs_v3.append(p3)
            probs_v6.append(p6)
            probs_mild.append(pm)
    all_labels = np.concatenate(all_labels)
    probs_v3 = np.concatenate(probs_v3, axis=0)
    probs_v6 = np.concatenate(probs_v6, axis=0)
    probs_mild = np.concatenate(probs_mild, axis=0)
    combined = weights[0] * probs_v3 + weights[1] * probs_v6 + weights[2] * probs_mild
    ensemble_preds = combined.argmax(axis=1)
    return ensemble_preds, all_labels, [
        ("V3", probs_v3.argmax(axis=1)),
        ("V6", probs_v6.argmax(axis=1)),
        ("Mild", probs_mild.argmax(axis=1)),
    ]


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean() * 100.0


def pair_accuracy(y_true, y_pred, idx_a, idx_b):
    mask_a, mask_b = y_true == idx_a, y_true == idx_b
    n_a, n_b = mask_a.sum(), mask_b.sum()
    acc_a = (y_pred[mask_a] == idx_a).mean() * 100.0 if n_a else 0.0
    acc_b = (y_pred[mask_b] == idx_b).mean() * 100.0 if n_b else 0.0
    if n_a and n_b:
        return (acc_a + acc_b) / 2.0
    return acc_a if n_a else acc_b


def plot_confusion_matrix(y_true, y_pred, save_path, num_classes=36):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        return
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    labels_str = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]
    ax.set_xticklabels(labels_str)
    ax.set_yticklabels(labels_str)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white", fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Weighted Ensemble)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="EMNIST 36 加權集成 (V3 + V6 + Mild)，同一份 test 每類 800 筆")
    parser.add_argument("--ckpt_v3", type=str, required=True, help="V3 標竿版 .pt")
    parser.add_argument("--ckpt_v6", type=str, required=True, help="V6 終極均衡版 .pt")
    parser.add_argument("--ckpt_mild", type=str, required=True, help="Mild 溫和均衡版 .pt")
    parser.add_argument("--data_dir", type=str, default=None, help="資料目錄，預設專案 data/emnist")
    parser.add_argument("--output_dir", type=str, default=None, help="報表與混淆矩陣輸出目錄")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--w_v3", type=float, default=0.4)
    parser.add_argument("--w_v6", type=float, default=0.4)
    parser.add_argument("--w_mild", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.output_dir or str(ROOT / "outputs" / "emnist36_ensemble")
    os.makedirs(out_dir, exist_ok=True)
    weights = [args.w_v3, args.w_v6, args.w_mild]
    assert abs(sum(weights) - 1.0) < 1e-6, "權重總和須為 1"

    data_dir = args.data_dir or str(ROOT / "data")
    print("使用與 V3/Mild/V6 評估同一份 test（每類 800 筆，get_emnist36_test_loader_raw）...")
    test_loader = get_emnist36_test_loader_raw(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_per_class=800,
    )

    print("載入三個 checkpoint ...")
    m_v3, norm_v3 = load_model(args.ckpt_v3, use_normalize=True)
    m_v6, norm_v6 = load_model(args.ckpt_v6, use_normalize=True)
    m_mild, norm_mild = load_model(args.ckpt_mild, use_normalize=False)

    ensemble_preds, all_labels, single_preds = run_ensemble_inference(
        [(m_v3, norm_v3), (m_v6, norm_v6), (m_mild, norm_mild)],
        test_loader,
        DEVICE,
        weights,
    )

    results = {}
    print("\n========== 集成前後對比 ==========")
    for name, preds in single_preds:
        acc = accuracy(all_labels, preds)
        results[name] = {"accuracy": round(acc, 4)}
        print(f"  {name}: {acc:.2f}%")
    ens_acc = accuracy(all_labels, ensemble_preds)
    results["Ensemble"] = {"accuracy": round(ens_acc, 4)}
    print(f"  Ensemble (weighted soft voting): {ens_acc:.2f}%")

    print("\n========== 魔王類別（兩類整體準確率 %）==========")
    demon = {}
    for name_a, name_b, idx_a, idx_b in DEMON_PAIRS:
        key = f"{name_a}_{name_b}"
        pair_acc_single = {n: pair_accuracy(all_labels, p, idx_a, idx_b) for n, p in single_preds}
        pair_acc_ens = pair_accuracy(all_labels, ensemble_preds, idx_a, idx_b)
        demon[key] = {
            "V3": round(pair_acc_single["V3"], 2),
            "V6": round(pair_acc_single["V6"], 2),
            "Mild": round(pair_acc_single["Mild"], 2),
            "Ensemble": round(pair_acc_ens, 2),
        }
        print(f"  {key}:  V3={demon[key]['V3']:.2f}  V6={demon[key]['V6']:.2f}  Mild={demon[key]['Mild']:.2f}  Ensemble={pair_acc_ens:.2f}")
    results["demon_pairs"] = demon

    cm_path = os.path.join(out_dir, "confusion_matrix_ensemble.png")
    plot_confusion_matrix(all_labels, ensemble_preds, cm_path)
    print(f"\n混淆矩陣已儲存: {cm_path}")

    report_path = os.path.join(out_dir, "ensemble_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"報表已儲存: {report_path}")

    best_single = max(results[n]["accuracy"] for n in ["V3", "V6", "Mild"])
    print("\n========== 結論 ==========")
    if ens_acc >= 92.0:
        print("  聯手衝破 92%！")
    else:
        print(f"  集成後 {ens_acc:.2f}%，最佳單體 {best_single:.2f}%，提升 {ens_acc - best_single:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
