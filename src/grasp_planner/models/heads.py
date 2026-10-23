"""Shared dense output heads: four 1-channel maps (quality, cos2theta, sin2theta, width)."""

from __future__ import annotations

import torch
from torch import nn

HEAD_NAMES = ("quality", "cos2", "sin2", "width")


class GraspHeads(nn.Module):
    """Four parallel convolutions, each producing one output map without activation
    (the losses regress the raw outputs directly, as in GR-ConvNet)."""

    def __init__(self, in_channels: int, kernel_size: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.convs = nn.ModuleDict({n: nn.Conv2d(in_channels, 1, kernel_size) for n in HEAD_NAMES})

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.dropout(z)
        return {n: conv(z) for n, conv in self.convs.items()}
