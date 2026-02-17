"""
共用資料集與 DataLoader。MNIST、EMNIST Letters，統一 transform（normalize、可選 augmentation）。
EMNIST 36 類可選 use_memmap=True 避免一次載入全部資料導致 MemoryError。
"""
import os
import random
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, EMNIST
from torchvision.datasets.utils import download_and_extract_archive, extract_archive


# 專案 data 目錄（與此檔相對路徑）
def _data_root():
    return Path(__file__).resolve().parent.parent / "data"


def get_mnist_loaders(
    batch_size=64,
    val_ratio=0.1,
    data_dir=None,
    augment_train=False,
    num_workers=0,
):
    """
    MNIST train/val DataLoader。可選 train 時做簡單 augmentation。
    """
    data_dir = data_dir or _data_root() / "mnist"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if augment_train:
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            normalize,
        ])

    full_train = MNIST(root=str(data_dir), train=True, download=True, transform=transform_train)
    n = len(full_train)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    test_ds = MNIST(root=str(data_dir), train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_emnist_letters_loaders(
    batch_size=64,
    val_ratio=0.1,
    data_dir=None,
    augment_train=False,
    num_workers=0,
):
    """
    EMNIST Letters (A–Z, 26 類) train/val/test DataLoader。
    split='letters' 對應 26 類，label 1–26 對應 A–Z。
    """
    data_dir = data_dir or _data_root() / "emnist"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # EMNIST 影像需轉置空間維度 (H,W) 才與一般書寫方向一致；保持 (C,H,W) 給 Conv2d
    def _transpose_and_normalize():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),  # (1,H,W) 空間轉置，不動 channel
            transforms.Normalize((0.5,), (0.5,)),
        ])

    transform_test = _transpose_and_normalize()
    transform_train = _transpose_and_normalize()
    if augment_train:
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    # EMNIST Letters 標籤為 1–26（A–Z），CrossEntropy 需要 0–25
    target_transform = lambda y: y - 1
    full_train = EMNIST(
        root=str(data_dir), split="letters", train=True, download=True,
        transform=transform_train, target_transform=target_transform
    )
    n = len(full_train)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    test_ds = EMNIST(
        root=str(data_dir), split="letters", train=False, download=True,
        transform=transform_test, target_transform=target_transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# --- EMNIST Digits + Uppercase Letters (36 classes: 0–9 + A–Z) ---
EMNIST36_NUM_CLASSES = 36
EMNIST36_LABEL_NAMES = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]  # "0"-"9", "A"-"Z"

EMNIST_GZIP_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
EMNIST_GZIP_MD5 = "58c8d27c78d21e728a6bc7b3cc06412e"


def _ensure_emnist_raw(data_dir):
    """下載並解壓 EMNIST 原始檔到 EMNIST/raw，不載入進記憶體。"""
    data_dir = Path(data_dir)
    raw = data_dir / "EMNIST" / "raw"
    images_digits = raw / "emnist-digits-train-images-idx3-ubyte"
    if images_digits.exists():
        return
    raw.mkdir(parents=True, exist_ok=True)
    download_and_extract_archive(EMNIST_GZIP_URL, download_root=str(raw), md5=EMNIST_GZIP_MD5)
    gzip_folder = raw / "gzip"
    if gzip_folder.exists():
        for g in gzip_folder.glob("*.gz"):
            extract_archive(str(g), str(raw))
        shutil.rmtree(gzip_folder, ignore_errors=True)


def _read_idx_header(path):
    """IDX: magic 4B, ndim 1B (magic%256), dims 4B each. Return (offset, shape)."""
    with open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic % 256
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        offset = 4 + 4 * ndim
    return offset, dims


class EMNISTMemmap(Dataset):
    """EMNIST 用 memory-mapped 讀取，不一次載入全部，避免 MemoryError。"""
    def __init__(self, root, split, train, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        prefix = f"emnist-{split}-{'train' if train else 'test'}"
        raw = Path(root) / "EMNIST" / "raw"
        img_path = raw / f"{prefix}-images-idx3-ubyte"
        lbl_path = raw / f"{prefix}-labels-idx1-ubyte"
        if not img_path.exists():
            _ensure_emnist_raw(root)
        off_img, shape_img = _read_idx_header(img_path)
        off_lbl, shape_lbl = _read_idx_header(lbl_path)
        self.n = shape_img[0]
        self.images = np.memmap(img_path, dtype="uint8", mode="r", offset=off_img, shape=tuple(shape_img))
        self.labels = np.memmap(lbl_path, dtype="uint8", mode="r", offset=off_lbl, shape=shape_lbl)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        from PIL import Image
        x = self.images[idx].copy()  # (H,W) uint8
        y = int(self.labels[idx])
        img = Image.fromarray(x, mode="L")
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            y = self.target_transform(y)
        return img, y


def get_balanced_subset(dataset, samples_per_class, num_classes=36, seed=42):
    """
    從 dataset 中每類取 samples_per_class 個樣本（不足則全取），回傳 Subset。
    假設 dataset[i] 回傳 (img, label)，label 為 0..num_classes-1。
    """
    class_to_indices = [[] for _ in range(num_classes)]
    rng = random.Random(seed)
    for i in range(len(dataset)):
        _, label = dataset[i]
        if 0 <= label < num_classes:
            class_to_indices[label].append(i)
    indices = []
    for c in range(num_classes):
        n = min(samples_per_class, len(class_to_indices[c]))
        indices.extend(rng.sample(class_to_indices[c], n))
    return Subset(dataset, indices)


def get_balanced_subset_precision(
    dataset,
    num_classes=36,
    target_classes=(0, 24, 1, 18),
    target_per_class=15000,
    other_per_class=5000,
    seed=42,
):
    """
    精準打擊：target_classes 每類取 target_per_class 筆（可過採樣），其餘每類 other_per_class。
    target_classes 預設為 0, O(24), 1, I(18)。不足時有幾筆取幾筆；過採樣時用 replacement。
    """
    class_to_indices = [[] for _ in range(num_classes)]
    rng = random.Random(seed)
    for i in range(len(dataset)):
        _, label = dataset[i]
        if 0 <= label < num_classes:
            class_to_indices[label].append(i)
    indices = []
    for c in range(num_classes):
        pool = class_to_indices[c]
        if not pool:
            continue
        if c in target_classes:
            n = target_per_class
            if n <= len(pool):
                indices.extend(rng.sample(pool, n))
            else:
                indices.extend(rng.choices(pool, k=n))
        else:
            n = min(other_per_class, len(pool))
            indices.extend(rng.sample(pool, n))
    return Subset(dataset, indices)


class TransformWrapper(Dataset):
    """對每筆 (img, label) 套用 transform(img)，用於 val 等需統一 ToTensor+Normalize 的場合。"""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


class PerClassAugmentWrapper(Dataset):
    """
    對指定類別套用「動態」擴增，輸入為 PIL，輸出為 Tensor。
    順序嚴格：RandomAffine(PIL) -> ToTensor -> Normalize -> RandomErasing(Tensor)。
    - light_transform / strong_transform：整條鏈（含 ToTensor、Normalize）；strong 末端為 RandomErasing。
    - 非目標類別僅套用 base_transform（ToTensor + Normalize）。
    """

    def __init__(self, subset, target_classes, light_transform, strong_transform, base_transform, p=0.5):
        self.subset = subset
        self.target_classes = set(target_classes)
        self.light_transform = light_transform
        self.strong_transform = strong_transform
        self.base_transform = base_transform
        self.p = p

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if label in self.target_classes:
            if random.random() < self.p:
                img = self.strong_transform(img)
            else:
                img = self.light_transform(img)
        else:
            img = self.base_transform(img)
        return img, label


class UniformAugmentWrapper(Dataset):
    """所有類別統一擴增：以機率 p 套用 augment_transform，否則用 base_transform。"""

    def __init__(self, subset, augment_transform, base_transform, p=0.3):
        self.subset = subset
        self.augment_transform = augment_transform
        self.base_transform = base_transform
        self.p = p

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if random.random() < self.p:
            img = self.augment_transform(img)
        else:
            img = self.base_transform(img)
        return img, label


def get_emnist_digits_uppercase_loaders(
    batch_size=64,
    val_ratio=0.1,
    data_dir=None,
    augment_train=False,
    num_workers=0,
    seed=42,
    use_memmap=True,
):
    """
    EMNIST Digits + Uppercase Letters: 36 classes (0–9 + A–Z).
    use_memmap=True（預設）用 memory-mapped 讀取，避免一次載入導致 MemoryError。
    Letters labels 1–26 映射為 10–35。
    """
    data_dir = data_dir or _data_root() / "emnist"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    def _transpose_and_normalize():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    transform_test = _transpose_and_normalize()
    transform_train = _transpose_and_normalize()
    if augment_train:
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    if use_memmap:
        _ensure_emnist_raw(str(data_dir))
        digits_train = EMNISTMemmap(str(data_dir), "digits", train=True, transform=transform_train)
        digits_test = EMNISTMemmap(str(data_dir), "digits", train=False, transform=transform_test)
        letters_train = EMNISTMemmap(
            str(data_dir), "letters", train=True,
            transform=transform_train,
            target_transform=lambda y: y - 1 + 10,
        )
        letters_test = EMNISTMemmap(
            str(data_dir), "letters", train=False,
            transform=transform_test,
            target_transform=lambda y: y - 1 + 10,
        )
    else:
        digits_train = EMNIST(
            root=str(data_dir), split="digits", train=True, download=True,
            transform=transform_train,
        )
        digits_test = EMNIST(
            root=str(data_dir), split="digits", train=False, download=True,
            transform=transform_test,
        )
        letters_train = EMNIST(
            root=str(data_dir), split="letters", train=True, download=True,
            transform=transform_train,
            target_transform=lambda y: y - 1 + 10,
        )
        letters_test = EMNIST(
            root=str(data_dir), split="letters", train=False, download=True,
            transform=transform_test,
            target_transform=lambda y: y - 1 + 10,
        )

    full_train = ConcatDataset([digits_train, letters_train])
    test_ds = ConcatDataset([digits_test, letters_test])

    n = len(full_train)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_emnist36_balanced_loaders(
    samples_per_class_train=5000,
    samples_per_class_test=800,
    batch_size=64,
    val_ratio=0.1,
    data_dir=None,
    augment_train=False,
    num_workers=0,
    seed=42,
    use_memmap=True,
):
    """
    EMNIST 36 類平衡抽樣：train 每類最多 samples_per_class_train，test 每類最多 samples_per_class_test，
    再對 train 做 random_split 得到 train/val，回傳 train_loader, val_loader, test_loader。
    """
    data_dir = data_dir or _data_root() / "emnist"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    def _transpose_and_normalize():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    transform_test = _transpose_and_normalize()
    transform_train = _transpose_and_normalize()
    if augment_train:
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    if use_memmap:
        _ensure_emnist_raw(str(data_dir))
        digits_train = EMNISTMemmap(str(data_dir), "digits", train=True, transform=transform_train)
        digits_test = EMNISTMemmap(str(data_dir), "digits", train=False, transform=transform_test)
        letters_train = EMNISTMemmap(
            str(data_dir), "letters", train=True,
            transform=transform_train,
            target_transform=lambda y: y - 1 + 10,
        )
        letters_test = EMNISTMemmap(
            str(data_dir), "letters", train=False,
            transform=transform_test,
            target_transform=lambda y: y - 1 + 10,
        )
    else:
        digits_train = EMNIST(
            root=str(data_dir), split="digits", train=True, download=True,
            transform=transform_train,
        )
        digits_test = EMNIST(
            root=str(data_dir), split="digits", train=False, download=True,
            transform=transform_test,
        )
        letters_train = EMNIST(
            root=str(data_dir), split="letters", train=True, download=True,
            transform=transform_train,
            target_transform=lambda y: y - 1 + 10,
        )
        letters_test = EMNIST(
            root=str(data_dir), split="letters", train=False, download=True,
            transform=transform_test,
            target_transform=lambda y: y - 1 + 10,
        )

    full_train = ConcatDataset([digits_train, letters_train])
    test_ds = ConcatDataset([digits_test, letters_test])

    train_subset = get_balanced_subset(full_train, samples_per_class_train, num_classes=36, seed=seed)
    test_subset = get_balanced_subset(test_ds, samples_per_class_test, num_classes=36, seed=seed)

    n = len(train_subset)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_ds, val_ds = random_split(
        train_subset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_emnist36_balanced_refined_loaders(
    target_classes=(0, 24, 1, 18),  # 0, O, 1, I
    target_per_class=15000,
    other_per_class=5000,
    samples_per_class_test=800,
    batch_size=64,
    val_ratio=0.1,
    data_dir=None,
    num_workers=0,
    seed=42,
    use_memmap=True,
):
    """
    溫和均衡版：0, O(24), 1, I(18) 每類 target_per_class（15000），其餘 other_per_class（5000）。
    僅對 0/O/1/I 做擴增：RandomAffine(degrees=10, translate=(0.1, 0.1))，20% 機率套用。
    不做 RandomErasing。
    Test 使用平衡分布：每類 samples_per_class_test 筆（公正衡量）。
    回傳 train_loader, val_loader, test_loader。
    """
    data_dir = data_dir or _data_root() / "emnist"
    data_dir = Path(data_dir)

    def _to_tensor_only():
        """僅做 ToTensor，不做 Normalize（注意：v3 有做 Normalize，此版本與 v3 不一致）"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
        ])

    transform_test = _to_tensor_only()
    transform_train_pil = None  # 保持 PIL，讓 wrapper 內做擴增

    if use_memmap:
        _ensure_emnist_raw(str(data_dir))
        digits_train = EMNISTMemmap(str(data_dir), "digits", train=True, transform=transform_train_pil)
        digits_test = EMNISTMemmap(str(data_dir), "digits", train=False, transform=transform_test)
        letters_train = EMNISTMemmap(
            str(data_dir), "letters", train=True,
            transform=transform_train_pil,
            target_transform=lambda y: y - 1 + 10,
        )
        letters_test = EMNISTMemmap(
            str(data_dir), "letters", train=False,
            transform=transform_test,
            target_transform=lambda y: y - 1 + 10,
        )
    else:
        digits_train = EMNIST(str(data_dir), "digits", train=True, download=True, transform=transform_train_pil)
        digits_test = EMNIST(str(data_dir), "digits", train=False, download=True, transform=transform_test)
        letters_train = EMNIST(
            str(data_dir), "letters", train=True, download=True,
            transform=transform_train_pil,
            target_transform=lambda y: y - 1 + 10,
        )
        letters_test = EMNIST(
            str(data_dir), "letters", train=False, download=True,
            transform=transform_test,
            target_transform=lambda y: y - 1 + 10,
        )

    full_train = ConcatDataset([digits_train, letters_train])
    test_ds = ConcatDataset([digits_test, letters_test])

    train_subset = get_balanced_subset_precision(
        full_train,
        num_classes=36,
        target_classes=target_classes,
        target_per_class=target_per_class,
        other_per_class=other_per_class,
        seed=seed,
    )
    # Test 使用平衡分布：每類 samples_per_class_test 筆（公正衡量）
    test_subset = get_balanced_subset(test_ds, samples_per_class_test, num_classes=36, seed=seed)

    n = len(train_subset)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_part, val_part = random_split(
        train_subset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # 僅對 0/O/1/I 做擴增：RandomAffine(degrees=10, translate=(0.1, 0.1))，20% 機率
    # 所有類別都只做 ToTensor，不做 Normalize（注意：v3 有做 Normalize，此版本與 v3 不一致）
    to_tensor_only = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2)),
    ])
    augment_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        to_tensor_only,  # 擴增後也只做 ToTensor，不做 Normalize
    ])
    # PerClassAugmentWrapper: target_classes 以機率 p 用 strong_transform（擴增），否則用 light_transform（不擴增）
    # 非 target_classes 用 base_transform（僅 ToTensor，不做 Normalize）
    train_ds = PerClassAugmentWrapper(train_part, target_classes, to_tensor_only, augment_transform, to_tensor_only, p=0.2)
    val_ds = TransformWrapper(val_part, to_tensor_only)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_emnist36_precision_targeting_loaders(
    target_classes=(0, 24, 1, 18),
    target_per_class=15000,
    other_per_class=5000,
    samples_per_class_test=800,
    test_target_per_class=None,
    test_other_per_class=None,
    batch_size=64,
    val_ratio=0.1,
    data_dir=None,
    num_workers=0,
    seed=42,
    use_memmap=True,
):
    """
    精準打擊：0, O(24), 1, I(18) 四類每類 target_per_class（3x 過採樣 15000），其餘 other_per_class（5000）。
    魔王類動態擴增：每筆進入前以機率 p=0.5 僅輕微旋轉，50% 機率施加強力 RandomAffine + RandomErasing。
    Test：若 test_target_per_class / test_other_per_class 有給，則 test 也同分布；否則每類 samples_per_class_test 平衡。
    回傳 train_loader, val_loader, test_loader。
    """
    data_dir = data_dir or _data_root() / "emnist"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    def _to_tensor_norm():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    transform_test = _to_tensor_norm()
    # 精準打擊：train 先不轉 Tensor，讓 wrapper 內做 RandomAffine(PIL) -> ToTensor -> Normalize -> RandomErasing(Tensor)
    transform_train_pil = None

    if use_memmap:
        _ensure_emnist_raw(str(data_dir))
        digits_train = EMNISTMemmap(str(data_dir), "digits", train=True, transform=transform_train_pil)
        digits_test = EMNISTMemmap(str(data_dir), "digits", train=False, transform=transform_test)
        letters_train = EMNISTMemmap(
            str(data_dir), "letters", train=True,
            transform=transform_train_pil,
            target_transform=lambda y: y - 1 + 10,
        )
        letters_test = EMNISTMemmap(
            str(data_dir), "letters", train=False,
            transform=transform_test,
            target_transform=lambda y: y - 1 + 10,
        )
    else:
        digits_train = EMNIST(
            root=str(data_dir), split="digits", train=True, download=True,
            transform=transform_train_pil,
        )
        digits_test = EMNIST(
            root=str(data_dir), split="digits", train=False, download=True,
            transform=transform_test,
        )
        letters_train = EMNIST(
            root=str(data_dir), split="letters", train=True, download=True,
            transform=transform_train_pil,
            target_transform=lambda y: y - 1 + 10,
        )
        letters_test = EMNIST(
            root=str(data_dir), split="letters", train=False, download=True,
            transform=transform_test,
            target_transform=lambda y: y - 1 + 10,
        )

    full_train = ConcatDataset([digits_train, letters_train])
    test_ds = ConcatDataset([digits_test, letters_test])

    train_subset = get_balanced_subset_precision(
        full_train,
        num_classes=36,
        target_classes=target_classes,
        target_per_class=target_per_class,
        other_per_class=other_per_class,
        seed=seed,
    )
    if test_target_per_class is not None and test_other_per_class is not None:
        test_subset = get_balanced_subset_precision(
            test_ds,
            num_classes=36,
            target_classes=target_classes,
            target_per_class=test_target_per_class,
            other_per_class=test_other_per_class,
            seed=seed,
        )
    else:
        test_subset = get_balanced_subset(test_ds, samples_per_class_test, num_classes=36, seed=seed)

    n = len(train_subset)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_part, val_part = random_split(
        train_subset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    # 順序：RandomAffine(PIL) -> ToTensor -> Normalize -> RandomErasing(Tensor)，RandomErasing 僅作用於 Tensor
    to_tensor_norm = _to_tensor_norm()
    light_aug = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        to_tensor_norm,
    ])
    strong_aug = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.15, 0.15)),
        to_tensor_norm,
        transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.5),
    ])
    train_ds = PerClassAugmentWrapper(train_part, target_classes, light_aug, strong_aug, to_tensor_norm, p=0.5)
    val_ds = TransformWrapper(val_part, to_tensor_norm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
