"""PyTorch dataset: Cornell scenes -> network inputs and dense grasp target maps.

Target maps (all of shape ``(S, S)`` on the network-input grid)
---------------------------------------------------------------
* ``quality``  Q: 1 inside the *centre third* of every positive rectangle, 0 elsewhere.
  The centre-third region (opening/3 along the closing direction x full jaw size) follows
  GG-CNN: only pixels near the rectangle centre are labelled as good grasp centres.
* ``cos2``  cos(2 theta) and ``sin2``  sin(2 theta) inside the same region, 0 elsewhere.
  Doubling the angle makes the encoding continuous across the theta ~ theta + pi symmetry.
* ``width``  opening in *output* pixels divided by ``max_width``, inside the region.

Where rectangles overlap, later rectangles overwrite earlier ones (file order), as in the
GG-CNN reference implementation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .cornell import CornellSample, load_depth, load_rgb
from .grasp import Grasp, rect_to_grasp
from .transforms import (
    AugmentSpec,
    CropSpec,
    apply_affine_points,
    inpaint_depth,
    sample_affine,
    warp_image,
)

MODALITY_CHANNELS = {"rgb": 3, "depth": 1, "rgbd": 4}


@dataclass(frozen=True)
class PreprocessSpec:
    """Deterministic preprocessing settings shared by training, evaluation and inference."""

    output_size: int = 224
    crop: CropSpec = CropSpec(320.0, 250.0, 320.0)
    modality: str = "rgbd"
    max_width: float = 150.0  # normaliser for the width map, in output pixels

    @property
    def in_channels(self) -> int:
        return MODALITY_CHANNELS[self.modality]

    @staticmethod
    def from_config(cfg: dict) -> PreprocessSpec:
        d = cfg["data"]
        c = d["crop"]
        return PreprocessSpec(
            output_size=int(d["output_size"]),
            crop=CropSpec(float(c["cx"]), float(c["cy"]), float(c["size"])),
            modality=d["modality"],
            max_width=float(d["max_width"]),
        )


def normalise_rgb(rgb: np.ndarray) -> np.ndarray:
    """uint8 (H, W, 3) -> float32 (3, H, W) in [0, 1] with the per-image mean removed."""
    x = rgb.astype(np.float32) / 255.0
    x -= x.mean()
    return x.transpose(2, 0, 1)


def normalise_depth(depth: np.ndarray) -> np.ndarray:
    """float depth (H, W) in metres -> (1, H, W): per-image mean removed, clipped to [-1, 1]."""
    d = depth - depth.mean()
    return np.clip(d, -1.0, 1.0)[None].astype(np.float32)


def build_input(rgb: np.ndarray | None, depth: np.ndarray | None, modality: str) -> np.ndarray:
    """Stack normalised channels in the order [depth, r, g, b] (depth first, as GR-ConvNet)."""
    parts = []
    if modality in ("depth", "rgbd"):
        if depth is None:
            raise ValueError(f"modality '{modality}' needs a depth image")
        parts.append(normalise_depth(depth))
    if modality in ("rgb", "rgbd"):
        if rgb is None:
            raise ValueError(f"modality '{modality}' needs an RGB image")
        parts.append(normalise_rgb(rgb))
    return np.concatenate(parts, axis=0)


def draw_grasp_targets(grasps: list[Grasp], size: int, max_width: float) -> np.ndarray:
    """Rasterise grasps into the (4, size, size) target stack [quality, cos2, sin2, width]."""
    out = np.zeros((4, size, size), dtype=np.float32)
    for g in grasps:
        centre_region = Grasp(g.x, g.y, g.theta, g.width / 3.0, g.jaw_size)
        poly = np.round(centre_region.polygon()).astype(np.int32)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        m = mask.astype(bool)
        if not m.any():
            continue
        out[0][m] = 1.0
        out[1][m] = np.cos(2 * g.theta)
        out[2][m] = np.sin(2 * g.theta)
        out[3][m] = min(g.width, max_width) / max_width
    return out


class DepthCache:
    """Disk cache of inpainted depth images (derived data; lives under ``outputs/cache``)."""

    def __init__(self, cache_dir: Path | None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, sample: CornellSample) -> np.ndarray:
        if sample.depth_path is None:
            raise FileNotFoundError(f"sample {sample.sample_id} has no depth image")
        if self.cache_dir:
            # key on the absolute source path + size + mtime so different dataset copies (or test
            # fixtures with the same scene id) can never collide or serve stale data
            st = sample.depth_path.stat()
            key = hashlib.sha1(
                f"{sample.depth_path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()
            ).hexdigest()[:12]
            f = self.cache_dir / f"pcd{sample.sample_id}_{key}.npy"
            if f.exists():
                return np.load(f)
        d = inpaint_depth(load_depth(sample.depth_path))
        if self.cache_dir:
            np.save(f, d)
        return d


class CornellGraspDataset(Dataset):
    """Torch dataset over a list of :class:`CornellSample` (one split).

    ``augment`` is only passed for the training split. Each item is a dict with
    ``x`` (C, S, S) input, ``y`` (4, S, S) targets, ``M`` (2, 3) original->input affine and
    ``index`` (position in ``samples``).
    """

    def __init__(
        self,
        samples: list[CornellSample],
        spec: PreprocessSpec,
        augment: AugmentSpec | None = None,
        seed: int = 0,
        cache_dir: Path | None = None,
    ):
        self.samples = samples
        self.spec = spec
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.depth_cache = DepthCache(cache_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """Make augmentation depend on (seed, epoch, index) so runs are reproducible
        regardless of the number of DataLoader workers."""
        self.epoch = epoch

    def load_raw(self, i: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        s = self.samples[i]
        rgb = load_rgb(s.rgb_path) if self.spec.modality in ("rgb", "rgbd") else None
        depth = self.depth_cache.get(s) if self.spec.modality in ("depth", "rgbd") else None
        return rgb, depth

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        rng = np.random.default_rng([self.seed, self.epoch, i]) if self.augment else None
        M = sample_affine(self.spec.crop, self.spec.output_size, self.augment, rng)
        rgb, depth = self.load_raw(i)
        S = self.spec.output_size
        rgb_w = warp_image(rgb, M, S) if rgb is not None else None
        depth_w = warp_image(depth, M, S) if depth is not None else None
        x = build_input(rgb_w, depth_w, self.spec.modality)
        grasps = [rect_to_grasp(apply_affine_points(M, r)) for r in s.pos]
        y = draw_grasp_targets(grasps, S, self.spec.max_width)
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "M": torch.from_numpy(M.astype(np.float64)),
            "index": i,
        }
