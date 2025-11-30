"""Conv1DClassifier implementation"""
from typing import Optional

import torch
import torch.nn as nn


class Conv1DClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4, dropout: float = 0.5, base_channels: int = 64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, 128),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        pooled = self.global_pool(feat)
        out = self.classifier(pooled)
        return out


def create_model(cfg: Optional[dict] = None) -> Conv1DClassifier:
    if cfg is None:
        return Conv1DClassifier()
    return Conv1DClassifier(
        in_channels=cfg.get("in_channels", 3),
        num_classes=cfg.get("num_classes", 4),
        dropout=cfg.get("dropout", 0.5),
        base_channels=cfg.get("base_channels", 64)
    )