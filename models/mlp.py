"""
可配置層數與寬度的 MLP。用於 MNIST（784 → 10）或任意 flatten 後的分類。
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    全連接 MLP：input_size -> hidden_sizes[0] -> ... -> num_classes。
    每層後接 ReLU，最後一層無 ReLU。
    """

    def __init__(self, input_size=784, hidden_sizes=(256, 128), num_classes=10, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x):
        # x: (B, C, H, W) -> flatten -> (B, input_size)
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def forward_embedding(self, x):
        """Return (embedding, logits). Embedding = layer before final classifier. No weight change."""
        x = x.view(x.size(0), -1)
        embedding = self.features(x)
        logits = self.classifier(embedding)
        return embedding, logits
