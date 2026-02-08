"""
共用資料集與 DataLoader。MNIST、EMNIST Letters，統一 transform（normalize、可選 augmentation）。
"""
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, EMNIST


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
