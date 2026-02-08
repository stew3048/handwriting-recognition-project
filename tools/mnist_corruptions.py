"""
MNIST geometric corruption generator for robustness analysis (STEP 1).

Corruption types: translation, rotation, scaling, shear, perspective, elastic.
Each has 3 severity levels (mild / medium / strong). All use a fixed RNG seed
and the same fixed test indices for reproducibility.

Severity choices (documented):
- Translation: ±2px / ±4px / ±6px (max shift in x and y)
- Rotation: ±10° / ±20° / ±30°
- Scale: scale factor in (0.9–1.1) / (0.8–1.2) / (0.7–1.3)
- Shear: ±10° / ±20° / ±30° (x-axis shear)
- Perspective: distortion_scale 0.1 / 0.2 / 0.3 (max corner shift = scale * min(W,H))
- Elastic: alpha (magnitude) / sigma (smoothness via kernel). Mild (5, 3), medium (15, 5), strong (30, 7).
  MNIST digits are 28×28; alpha in pixels, sigma as Gaussian kernel size for smoothing the displacement field.
"""
from pathlib import Path
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF

# Reproducibility
SEED = 42
N_GRID = 64
N_COMPARE = 16
MNIST_TEST_SIZE = 10_000

# Severity levels: 1 = mild, 2 = medium, 3 = strong
# Each entry is (mild_param, medium_param, strong_param) or dict
CORRUPTION_PARAMS = {
    "translation": (2, 4, 6),           # max |tx|,|ty| in pixels
    "rotation": (10, 20, 30),          # max |angle| in degrees
    "scale": ((0.9, 1.1), (0.8, 1.2), (0.7, 1.3)),  # (min_scale, max_scale)
    "shear": (10, 20, 30),             # max |shear| in degrees (x-axis)
    "perspective": (0.1, 0.2, 0.3),    # distortion_scale (fraction of min(W,H))
    "elastic": ((5, 3), (15, 5), (30, 7)),  # (alpha_px, sigma for smoothing)
}


def get_fixed_indices(seed: int, n: int, max_size: int = MNIST_TEST_SIZE) -> np.ndarray:
    """Return n fixed test indices in [0, max_size) for reproducibility."""
    rng = np.random.default_rng(seed)
    return rng.choice(max_size, size=min(n, max_size), replace=False)


def _to_tensor_1ch(img) -> torch.Tensor:
    """Ensure image is (1, H, W) float in [0, 1]."""
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img).float()
    else:
        t = img.float()
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t.clamp(0.0, 1.0)


def _from_tensor(t: torch.Tensor) -> np.ndarray:
    """(1,H,W) or (H,W) -> (H,W) numpy for saving."""
    if t.dim() == 3:
        t = t.squeeze(0)
    return t.clamp(0.0, 1.0).numpy()


def apply_corruption(
    image: Union[torch.Tensor, np.ndarray],
    corruption_type: str,
    severity: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Apply one geometric corruption. severity in {1, 2, 3}.
    image: (1, 28, 28) or (28, 28), values [0, 1].
    Returns (1, 28, 28) clipped to [0, 1].
    """
    img = _to_tensor_1ch(image)
    if img.dim() == 2:
        img = img.unsqueeze(0)
    assert img.shape[-2:] == (28, 28), f"Expected 28x28, got {img.shape}"
    H, W = 28, 28

    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")
    idx = severity - 1

    if corruption_type == "translation":
        max_px = CORRUPTION_PARAMS["translation"][idx]
        tx = float(rng.uniform(-max_px, max_px))
        ty = float(rng.uniform(-max_px, max_px))
        out = TVF.affine(img, angle=0, translate=(tx, ty), scale=1.0, shear=0)

    elif corruption_type == "rotation":
        max_deg = CORRUPTION_PARAMS["rotation"][idx]
        angle = float(rng.uniform(-max_deg, max_deg))
        out = TVF.rotate(img, angle)

    elif corruption_type == "scale":
        low, high = CORRUPTION_PARAMS["scale"][idx]
        scale = float(rng.uniform(low, high))
        out = TVF.affine(img, angle=0, translate=(0, 0), scale=scale, shear=0)

    elif corruption_type == "shear":
        max_shear = CORRUPTION_PARAMS["shear"][idx]
        shear = float(rng.uniform(-max_shear, max_shear))
        out = TVF.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=(shear, 0))

    elif corruption_type == "perspective":
        d = CORRUPTION_PARAMS["perspective"][idx]
        max_shift = d * min(W, H)
        startpoints = [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]
        endpoints = [
            [startpoints[i][0] + rng.uniform(-max_shift, max_shift), startpoints[i][1] + rng.uniform(-max_shift, max_shift)]
            for i in range(4)
        ]
        startpoints = [tuple(p) for p in startpoints]
        endpoints = [tuple(p) for p in endpoints]
        out = TVF.perspective(img, startpoints, endpoints)

    elif corruption_type == "elastic":
        alpha, sigma = CORRUPTION_PARAMS["elastic"][idx]
        # Displacement field: smooth random noise * alpha. sigma = kernel half-size for smoothing.
        noise = rng.uniform(-1, 1, (2, H, W)).astype(np.float32)
        k = max(1, int(sigma * 2) | 1)
        pad = k // 2
        padded0 = np.pad(noise[0], pad, mode="edge")
        padded1 = np.pad(noise[1], pad, mode="edge")
        smoothed0 = np.array(
            [[padded0[i : i + k, j : j + k].mean() for j in range(W)] for i in range(H)],
            dtype=np.float32,
        )
        smoothed1 = np.array(
            [[padded1[i : i + k, j : j + k].mean() for j in range(W)] for i in range(H)],
            dtype=np.float32,
        )
        dx = torch.from_numpy(smoothed0 * alpha).float()
        dy = torch.from_numpy(smoothed1 * alpha).float()
        # Grid: normalized coords -1..1. grid[y,x] = (x_new_norm, y_new_norm)
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
        x_new = (xx + dx).clamp(0, W - 1) / (W - 1) * 2 - 1
        y_new = (yy + dy).clamp(0, H - 1) / (H - 1) * 2 - 1
        grid = torch.stack([x_new, y_new], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        img_batch = img.unsqueeze(0)  # (1, 1, H, W)
        out = F.grid_sample(img_batch, grid, mode="bilinear", padding_mode="zeros").squeeze(0)

    else:
        raise ValueError(f"Unknown corruption_type: {corruption_type}")

    out = out.clamp(0.0, 1.0)
    return out if out.dim() == 3 else out.unsqueeze(0)


def save_grid(images: list, title: str, path: Path) -> None:
    """Save an 8×8 grid of images. images: list of (28,28) or (1,28,28) arrays/tensors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    n_cols = 8
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            arr = _from_tensor(images[idx]) if hasattr(images[idx], "clamp") else np.asarray(images[idx]).squeeze()
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def save_comparison(
    clean_list: list,
    s1_list: list,
    s2_list: list,
    s3_list: list,
    corruption_type: str,
    path: Path,
) -> None:
    """4 columns: [clean | severity1 | severity2 | severity3], each column 4×4 grid (16 samples)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(clean_list), len(s1_list), len(s2_list), len(s3_list), 16)
    clean_list = clean_list[:n]
    s1_list, s2_list, s3_list = s1_list[:n], s2_list[:n], s3_list[:n]

    def to_arr(x):
        return _from_tensor(x) if hasattr(x, "clamp") else np.asarray(x).squeeze()

    n_rows, n_cols_per_block = 4, 4
    n_blocks = 4
    fig, axes = plt.subplots(n_rows, n_cols_per_block * n_blocks, figsize=(n_blocks * n_cols_per_block * 1.0, n_rows * 1.0))
    cols = [("Clean", clean_list), ("Severity 1", s1_list), ("Severity 2", s2_list), ("Severity 3", s3_list)]
    for col_idx, (label, imgs) in enumerate(cols):
        for row in range(n_rows):
            for c in range(n_cols_per_block):
                idx = row * n_cols_per_block + c
                ax = axes[row, col_idx * n_cols_per_block + c]
                if idx < len(imgs):
                    ax.imshow(to_arr(imgs[idx]), cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
        axes[0, col_idx * n_cols_per_block].set_title(label, fontsize=10)
    fig.suptitle(f"{corruption_type} – clean vs severity 1–3")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
