"""Seeding, device selection, logging and environment helpers."""

from __future__ import annotations

import logging
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds any third-party code using the legacy global RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_usable() -> bool:
    """True only if CUDA is available *and* at least one device is visible. (With
    ``CUDA_VISIBLE_DEVICES=""`` some builds report ``is_available()`` True but 0 devices.)"""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def select_device(requested: str = "auto") -> torch.device:
    """``auto`` -> CUDA if usable, else CPU. Raises if CUDA is requested but unusable."""
    if requested == "auto":
        return torch.device("cuda" if cuda_usable() else "cpu")
    dev = torch.device(requested)
    if dev.type == "cuda" and not cuda_usable():
        raise RuntimeError("CUDA requested but no CUDA device is usable")
    return dev


def environment_info(device: torch.device) -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cpu": platform.processor(),
    }
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 2**30, 2))
    return info


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers, force=True
    )
