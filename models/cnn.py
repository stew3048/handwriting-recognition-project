"""
小型 CNN：2~3 層 conv + pool + FC。用於 MNIST（28x28）或同尺寸灰階圖。
"""
import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    簡單 CNN：Conv -> ReLU -> Pool 重複 2~3 層，再 flatten -> FC -> num_classes。
    """

    def __init__(self, num_classes=10, dropout=0.0):
        super().__init__()
        self.num_classes = num_classes
        # 28x28 -> conv 3x3 pad1 -> 28 -> pool 2 -> 14 -> conv -> 7 -> pool -> 3
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 28/2/2/2 = 3 -> 3*3*128 = 1152
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(3 * 3 * 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def forward_embedding(self, x):
        """Return (embedding, logits). Embedding = 256-dim before final Linear. No weight change."""
        x = self.features(x)
        # classifier: [0] Flatten, [1] Dropout, [2] Linear(1152,256), [3] ReLU, [4] Dropout, [5] Linear(256,10)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        embedding = self.classifier[3](x)
        x = self.classifier[4](embedding)
        logits = self.classifier[5](x)
        return embedding, logits
