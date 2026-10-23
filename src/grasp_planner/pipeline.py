"""Glue shared by the scripts: config -> samples, datasets, model and checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import REPO_ROOT, resolve_dataset_root
from .data.cornell import CornellSample, discover_samples
from .data.dataset import CornellGraspDataset, PreprocessSpec
from .data.splits import load_split
from .data.transforms import AugmentSpec
from .inference.decode import DecodeSpec
from .models import build_model

CACHE_DIR = REPO_ROOT / "outputs" / "cache" / "depth_inpainted"


def split_samples(cfg: dict, dataset_root: Path | None = None) -> dict[str, list[CornellSample]]:
    """Parse the dataset and group samples by the split file named in ``cfg['data']['split_file']``."""
    root = resolve_dataset_root(dataset_root)
    samples, _ = discover_samples(root)
    by_id = {s.sample_id: s for s in samples}
    split = load_split(REPO_ROOT / cfg["data"]["split_file"])
    out: dict[str, list[CornellSample]] = {}
    for name in ("train", "val", "test"):
        missing = [i for i in split["splits"][name] if i not in by_id]
        if missing:
            raise KeyError(f"split '{name}' references scenes not found/valid in the dataset: {missing[:10]}")
        out[name] = [by_id[i] for i in split["splits"][name]]
    return out


def augment_spec(cfg: dict) -> AugmentSpec | None:
    a = cfg["data"].get("augment")
    if not a or not a.get("enabled", False):
        return None
    return AugmentSpec(
        rotation_deg=float(a.get("rotation_deg", 0.0)),
        zoom_min=float(a.get("zoom_min", 1.0)),
        translate_px=float(a.get("translate_px", 0.0)),
        hflip_prob=float(a.get("hflip_prob", 0.0)),
    )


def make_dataset(
    cfg: dict, samples: list[CornellSample], train: bool, cache_dir: Path | None = CACHE_DIR
) -> CornellGraspDataset:
    return CornellGraspDataset(
        samples,
        PreprocessSpec.from_config(cfg),
        augment=augment_spec(cfg) if train else None,
        seed=int(cfg["experiment"]["seed"]),
        cache_dir=cache_dir,
    )


def decode_spec(cfg: dict) -> DecodeSpec:
    return DecodeSpec.from_config(cfg.get("decode", {}), float(cfg["data"]["max_width"]))


def make_model(cfg: dict) -> torch.nn.Module:
    return build_model(cfg["model"], PreprocessSpec.from_config(cfg).in_channels)


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model from the config stored inside a checkpoint and load its weights."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = make_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
