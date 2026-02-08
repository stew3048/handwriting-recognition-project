"""
STEP 1: Generate geometrically corrupted MNIST images, SAVE TO DISK first, then draw PNGs for human inspection.

Run from project root:
  python scripts/make_corrupt_viz.py

Flow: generate corruptions -> save tensors under data/corrupt_data/ -> load from disk -> draw PNGs under outputs/corrupt_viz/
So STEP 2 can load the same .pt/.npy and what you see in the PNGs is exactly what will be fed to models.

Outputs:
  data/corrupt_data/indices.npy, labels.npy, clean.pt, <type>_s1.pt, _s2.pt, _s3.pt
  outputs/corrupt_viz/<type>_s1.png, _s2.png, _s3.png, compare_<type>.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms

from tools.mnist_corruptions import (
    SEED,
    N_GRID,
    N_COMPARE,
    get_fixed_indices,
    apply_corruption,
    save_grid,
    save_comparison,
)

CORRUPTION_TYPES = ["translation", "rotation", "scale", "shear", "perspective", "elastic"]
# Deterministic seed per type (do not use hash() — not stable across runs)
TYPE_SEED_OFFSET = {t: i for i, t in enumerate(CORRUPTION_TYPES)}
DATA_DIR = ROOT / "data" / "corrupt_data"
OUT_DIR = ROOT / "outputs" / "corrupt_viz"


def main():
    print("STEP 1: MNIST geometric corruption – data generation + visual inspection")
    print("=" * 60)

    # Fixed indices (same across all types and severities)
    grid_indices = get_fixed_indices(SEED, N_GRID)
    compare_indices = grid_indices[:N_COMPARE]
    assert len(compare_indices) == 16, "Need 16 indices for comparison sheet"

    # Load MNIST test (raw [0,1], no normalization)
    data_dir = ROOT / "data" / "mnist"
    data_dir.mkdir(parents=True, exist_ok=True)
    test_ds = MNIST(root=str(data_dir), train=False, download=True, transform=transforms.ToTensor())

    def get_image(idx):
        img, _ = test_ds[idx]
        return img  # (1, 28, 28)

    def get_label(idx):
        _, label = test_ds[idx]
        return label

    # Deterministic RNG per corruption type (stable across runs)
    def get_rng(corruption_type):
        seed = SEED + TYPE_SEED_OFFSET[corruption_type]
        return np.random.default_rng(seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1) Build clean images and labels, then SAVE ----
    clean_list = [get_image(int(i)) for i in grid_indices]
    labels = np.array([get_label(int(i)) for i in grid_indices], dtype=np.int64)
    np.save(DATA_DIR / "indices.npy", grid_indices)
    np.save(DATA_DIR / "labels.npy", labels)
    torch.save(torch.stack(clean_list), DATA_DIR / "clean.pt")
    print(f"  Saved {DATA_DIR / 'indices.npy'}, labels.npy, clean.pt")

    generated_data = ["indices.npy", "labels.npy", "clean.pt"]
    generated_viz = []

    # ---- 2) For each type: generate corruptions -> SAVE .pt -> then draw PNGs from saved data ----
    for ctype in CORRUPTION_TYPES:
        rng = get_rng(ctype)
        list_s1, list_s2, list_s3 = [], [], []
        for i in grid_indices:
            img = get_image(int(i))
            list_s1.append(apply_corruption(img, ctype, 1, rng))
            list_s2.append(apply_corruption(img, ctype, 2, rng))
            list_s3.append(apply_corruption(img, ctype, 3, rng))

        # Save tensors first (64, 1, 28, 28) each
        t1 = torch.stack(list_s1)
        t2 = torch.stack(list_s2)
        t3 = torch.stack(list_s3)
        torch.save(t1, DATA_DIR / f"{ctype}_s1.pt")
        torch.save(t2, DATA_DIR / f"{ctype}_s2.pt")
        torch.save(t3, DATA_DIR / f"{ctype}_s3.pt")
        generated_data.extend([f"{ctype}_s1.pt", f"{ctype}_s2.pt", f"{ctype}_s3.pt"])
        print(f"  Saved {ctype}_s1.pt, _s2.pt, _s3.pt")

        # ---- 3) Produce human-viewable PNGs FROM the saved files (so PNGs = exactly what STEP 2 will load) ----
        clean_loaded = torch.load(DATA_DIR / "clean.pt")
        s1_loaded = torch.load(DATA_DIR / f"{ctype}_s1.pt")
        s2_loaded = torch.load(DATA_DIR / f"{ctype}_s2.pt")
        s3_loaded = torch.load(DATA_DIR / f"{ctype}_s3.pt")
        list_s1_ = [s1_loaded[i] for i in range(len(s1_loaded))]
        list_s2_ = [s2_loaded[i] for i in range(len(s2_loaded))]
        list_s3_ = [s3_loaded[i] for i in range(len(s3_loaded))]

        for severity, images in ((1, list_s1_), (2, list_s2_), (3, list_s3_)):
            title = f"{ctype} – severity {severity}"
            path = OUT_DIR / f"{ctype}_s{severity}.png"
            save_grid(images, title, path)
            generated_viz.append(path)
            print(f"  Saved {path}")

        clean_16 = [clean_loaded[i] for i in range(N_COMPARE)]
        path_compare = OUT_DIR / f"compare_{ctype}.png"
        save_comparison(clean_16, list_s1_[:16], list_s2_[:16], list_s3_[:16], ctype, path_compare)
        generated_viz.append(path_compare)
        print(f"  Saved {path_compare}")

    print()
    print("Summary of generated files:")
    print("-" * 60)
    print("Data (for STEP 2):")
    for name in sorted(generated_data):
        print(f"  {DATA_DIR / name}")
    print("Visualization (for human inspection):")
    for p in sorted(generated_viz):
        print(f"  {p}")
    print("-" * 60)
    print(f"Data: {len(generated_data)} files under {DATA_DIR}")
    print(f"Viz:  {len(generated_viz)} files under {OUT_DIR}")
    print("STEP 1 done. STEP 2 will load from data/corrupt_data/.")


if __name__ == "__main__":
    main()
