"""Dense regression loss for the four grasp maps.

``L = sum_h SmoothL1(pred_h, target_h)`` over heads h in {quality, cos2, sin2, width}, each a
mean over all pixels in the batch. Smooth L1 (Huber with beta = 1) is quadratic for small
residuals and linear for large ones, so it is less dominated by outliers than MSE. This is
the loss used by GR-ConvNet; the same loss is used for the GG-CNN baseline so that the two
models differ only in architecture.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.heads import HEAD_NAMES


def grasp_loss(pred: dict[str, torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """``pred``: dict of (B, 1, H, W) maps; ``target``: (B, 4, H, W) in HEAD_NAMES order."""
    total = torch.zeros((), device=target.device, dtype=torch.float32)
    parts: dict[str, float] = {}
    for i, name in enumerate(HEAD_NAMES):
        loss = F.smooth_l1_loss(pred[name].float(), target[:, i : i + 1].float())
        parts[name] = float(loss.detach())
        total = total + loss
    return total, parts
