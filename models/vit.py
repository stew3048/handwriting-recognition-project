"""
Vision Transformer (ViT) for EMNIST36：使用 timm 的 ViT-Tiny，適配 28x28 灰階輸入。
輸入：28x28 灰階 → 轉換為 224x224 RGB → ViT-Tiny → 36 類分類。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import timm


class ViT(nn.Module):
    """
    ViT-Tiny wrapper for EMNIST36 classification.
    Uses timm's pretrained ViT-Tiny, adapts 28x28 grayscale to 224x224 RGB.
    """

    def __init__(self, num_classes: int = 36, dropout: float = 0.0, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        # Load pretrained ViT-Tiny (patch16, 224x224 input)
        self.vit = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=0,  # Remove default head, we'll add our own
            drop_rate=dropout,
        )
        # Get embedding dimension from the model
        self.embed_dim = self.vit.embed_dim  # Should be 192 for ViT-Tiny
        
        # Replace classification head for 36 classes
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        # Input preprocessing: 28x28 grayscale -> 224x224 RGB
        # We'll do this in forward to avoid modifying dataset transforms

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert 28x28 grayscale (B, 1, 28, 28) to 224x224 RGB (B, 3, 224, 224).
        Uses bilinear interpolation for resize, repeats channel for RGB.
        """
        # Resize: 28x28 -> 224x224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Convert 1 channel to 3 channels (grayscale -> RGB)
        x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: returns logits only.
        Input: (B, 1, 28, 28) grayscale
        Output: (B, num_classes) logits
        """
        x = self._preprocess_input(x)
        # Get cls token embedding (B, embed_dim)
        cls_token = self.vit(x)
        # Classification head
        logits = self.head(cls_token)
        return logits

    def forward_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: returns (embedding, logits) for evaluation/visualization.
        Input: (B, 1, 28, 28) grayscale
        Output: (embedding, logits) where embedding is cls_token (B, embed_dim)
        """
        x = self._preprocess_input(x)
        # Get cls token embedding (B, embed_dim)
        embedding = self.vit(x)
        # Classification head
        logits = self.head(embedding)
        return embedding, logits
