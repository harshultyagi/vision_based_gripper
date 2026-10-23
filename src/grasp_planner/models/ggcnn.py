"""GG-CNN baseline (Morrison, Corke & Leitner, RSS 2018), adapted to arbitrary input size.

Original: three strided convolutions (9x9/3, 5x5/2, 3x3/2) with 32/16/8 filters, three mirrored
transposed convolutions, and four 2x2 output convolutions (quality, cos2, sin2, width). The
original operates on 300x300 depth only; here the input channel count is configurable and a
final bilinear resize guarantees the output grid equals the input grid for any size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .heads import GraspHeads


class GGCNN(nn.Module):
    def __init__(self, in_channels: int = 1, filters: tuple[int, int, int] = (32, 16, 8)):
        super().__init__()
        f1, f2, f3 = filters
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, f1, 9, stride=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(f3, f3, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(f3, f2, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(f2, f1, 9, stride=3, padding=3, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.heads = GraspHeads(f1, kernel_size=2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = x.shape[-2:]
        z = self.decoder(self.encoder(x))
        z = F.interpolate(z, size=(h + 1, w + 1), mode="bilinear", align_corners=False)
        return self.heads(z)  # 2x2 heads shrink (h+1, w+1) -> (h, w)
