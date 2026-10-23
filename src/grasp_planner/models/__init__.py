"""Model factory."""

from __future__ import annotations

from torch import nn

from .ggcnn import GGCNN
from .grconvnet import GRConvNet
from .heads import HEAD_NAMES

__all__ = ["GGCNN", "GRConvNet", "HEAD_NAMES", "build_model", "count_parameters"]


def build_model(model_cfg: dict, in_channels: int) -> nn.Module:
    """Instantiate a model from the ``model`` section of a config."""
    name = model_cfg["name"].lower()
    if name == "ggcnn":
        return GGCNN(in_channels, tuple(model_cfg.get("filters", (32, 16, 8))))
    if name == "grconvnet":
        return GRConvNet(
            in_channels,
            channels=int(model_cfg.get("channels", 32)),
            num_residual=int(model_cfg.get("num_residual", 5)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"unknown model {model_cfg['name']!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
