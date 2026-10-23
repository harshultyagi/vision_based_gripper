"""GR-ConvNet (Kumra, Joshi & Sahin, IROS 2020): generative residual convolutional network.

Layer sizes follow the authors' reference implementation (``GenerativeResnet``). For a
224 x 224 input the spatial sizes are::

    conv1 9x9/1  -> 224   conv2 4x4/2 -> 112   conv3 4x4/2 -> 56
    5 x residual blocks (128 ch) -> 56
    deconv 4x4/2 (pad 1, out-pad 1) -> 113   deconv 4x4/2 (pad 2, out-pad 1) -> 225
    conv 9x9/1 -> 225   2x2 heads -> 224
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .heads import GraspHeads


class ResidualBlock(nn.Module):
    """y = x + BN(conv(ReLU(BN(conv(x))))) with 3x3 convolutions and constant width."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return x + y


class GRConvNet(nn.Module):
    def __init__(self, in_channels: int = 4, channels: int = 32, num_residual: int = 5, dropout: float = 0.1):
        super().__init__()
        c = channels
        self.conv1 = nn.Conv2d(in_channels, c, 9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, 2 * c, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * c)
        self.conv3 = nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * c)
        self.res = nn.Sequential(*[ResidualBlock(4 * c) for _ in range(num_residual)])
        self.conv4 = nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(2 * c)
        self.conv5 = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(c)
        self.conv6 = nn.ConvTranspose2d(c, c, 9, stride=1, padding=4)
        self.heads = GraspHeads(c, kernel_size=2, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = x.shape[-2:]
        z = F.relu(self.bn1(self.conv1(x)))
        z = F.relu(self.bn2(self.conv2(z)))
        z = F.relu(self.bn3(self.conv3(z)))
        z = self.res(z)
        z = F.relu(self.bn4(self.conv4(z)))
        z = F.relu(self.bn5(self.conv5(z)))
        z = self.conv6(z)
        if z.shape[-2:] != (h + 1, w + 1):  # only for input sizes not divisible by 4
            z = F.interpolate(z, size=(h + 1, w + 1), mode="bilinear", align_corners=False)
        return self.heads(z)
