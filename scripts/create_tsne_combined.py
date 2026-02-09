"""
將各模型、各幾何破壞 S3（以及 clean）的 t-SNE 投影圖組合成一張 PNG。
版面：7 列（clean + 6 種 corruption s3）× 3 欄（MLP, CNN, ResNet），存到 outputs/robustness/tsne_combined_s3.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

OUT_BASE = ROOT / "outputs" / "robustness"
SCENARIOS = [
    ("clean", "Clean"),
    ("translation", "Translation S3"),
    ("rotation", "Rotation S3"),
    ("scale", "Scale S3"),
    ("shear", "Shear S3"),
    ("perspective", "Perspective S3"),
    ("elastic", "Elastic S3"),
]
MODELS = ["mlp", "cnn", "resnet"]
MODEL_NAMES = {"mlp": "MLP", "cnn": "CNN", "resnet": "ResNet"}


def get_tsne_path(model, scenario_key):
    if scenario_key == "clean":
        return OUT_BASE / model / "clean" / "tsne.png"
    return OUT_BASE / model / scenario_key / "s3" / "tsne.png"


def main():
    nrows, ncols = len(SCENARIOS), len(MODELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for i, (scenario_key, scenario_label) in enumerate(SCENARIOS):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            path = get_tsne_path(model, scenario_key)
            if not path.exists():
                ax.text(0.5, 0.5, f"Missing\n{path.name}", ha="center", va="center", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            img = imread(path)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(MODEL_NAMES[model], fontsize=12, weight="bold")
            if j == 0:
                ax.set_ylabel(scenario_label, fontsize=11, weight="bold")

    plt.suptitle("t-SNE 2D Embedding: 7 Scenarios × 3 Models (64 samples)", fontsize=14, weight="bold", y=1.002)
    plt.tight_layout()
    out_path = OUT_BASE / "tsne_combined_s3.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
