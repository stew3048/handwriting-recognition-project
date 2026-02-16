"""
小型 ResNet：適配 MNIST 28x28 灰階，約 4～6 層卷積 + residual，輸出 10 類。
可選 embedding_dim（如 128）時插入 64→embedding_dim→num_classes，forward 回傳 (logits, embeddings)。
"""
import torch
import torch.nn as nn
from typing import Tuple, Union


def _conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class _BasicBlock(nn.Module):
    """Two 3x3 convs with skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    """
    Small ResNet for MNIST: 1ch 28x28 -> 10 classes.
    Stem -> 3 layers of BasicBlocks (16, 32, 64 channels) -> global pool -> FC.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.0, embedding_dim: int = None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.in_ch = 16
        # stem: 28x28x1 -> 14x14x16
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 14x14x16 -> 14x14x16
        self.layer1 = self._make_layer(16, 16, 2, stride=1)
        # 14x14x16 -> 7x7x32
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        # 7x7x32 -> 3x3x64
        self.layer3 = self._make_layer(32, 64, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        if embedding_dim is not None and embedding_dim != 64:
            # 升維：64 -> embedding_dim -> num_classes
            self.feat = nn.Linear(64, embedding_dim)
            self.bn_feat = nn.BatchNorm1d(embedding_dim)
            self.relu_feat = nn.ReLU(inplace=True)
            self.fc = nn.Linear(embedding_dim, num_classes)
        else:
            # embedding_dim is None：只做分類；embedding_dim == 64：不升維，64 維即 embedding
            self.feat = None
            self.bn_feat = None
            self.relu_feat = None
            self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        blocks = [_BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            blocks.append(_BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        if self.embedding_dim is not None:
            if self.embedding_dim == 64:
                embeddings = x
                logits = self.fc(embeddings)
                return logits, embeddings
            embeddings = self.relu_feat(self.bn_feat(self.feat(x)))
            logits = self.fc(embeddings)
            return logits, embeddings
        return self.fc(x)

    def forward_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding, logits). With embedding_dim: 128-dim; else 64-dim before final Linear."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        if self.embedding_dim is not None:
            if self.embedding_dim == 64:
                embedding = x
                logits = self.fc(embedding)
                return embedding, logits
            embedding = self.relu_feat(self.bn_feat(self.feat(x)))
            logits = self.fc(embedding)
            return embedding, logits
        logits = self.fc(x)
        return x, logits
