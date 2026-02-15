"""
Center Loss: 每個樣本的 feature 與其類別中心的 L2 距離平方之平均。
用於 EMNIST36 + ResNet 128 維 embedding 訓練。
"""
import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (B, feat_dim), labels: (B,) 為類別 index 0..num_classes-1。
        回傳 batch 內每個樣本到其類中心的 L2 距離平方之平均。
        """
        centers_batch = self.centers[labels]
        return ((features - centers_batch) ** 2).sum(dim=1).mean()
