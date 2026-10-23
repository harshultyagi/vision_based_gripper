"""High-level inference: image(s) in, ranked grasp candidates out (original-image pixels)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ..data.cornell import IMAGE_HEIGHT, IMAGE_WIDTH, mask_invalid_depth
from ..data.dataset import PreprocessSpec, build_input
from ..data.grasp import Grasp
from ..data.transforms import CropSpec, build_affine, inpaint_depth, warp_image
from ..evaluation.runner import predict_maps
from ..pipeline import decode_spec, load_checkpoint_model
from ..utils import select_device
from .decode import DecodeSpec, decode_grasps, grasp_to_original


@dataclass
class Prediction:
    candidates: list[Grasp]  # ranked, original-image pixel coordinates
    maps: dict[str, np.ndarray]  # post-network maps on the input grid
    M: np.ndarray  # original -> input affine
    input_size: int

    @property
    def best(self) -> Grasp | None:
        return self.candidates[0] if self.candidates else None


def crop_for_image(spec: PreprocessSpec, width: int, height: int) -> CropSpec:
    """Training crop for 640x480 Cornell images; for other sizes the crop centre and side are
    scaled proportionally (side by the smaller scale factor) so the same relative region is used."""
    if (width, height) == (IMAGE_WIDTH, IMAGE_HEIGHT):
        return spec.crop
    sx, sy = width / IMAGE_WIDTH, height / IMAGE_HEIGHT
    return CropSpec(spec.crop.cx * sx, spec.crop.cy * sy, spec.crop.size * min(sx, sy))


class GraspPredictor:
    """Loads a trained checkpoint and predicts grasps for single images."""

    def __init__(self, checkpoint: str | Path, device: torch.device | None = None):
        self.device = device or select_device("auto")
        self.model, ckpt = load_checkpoint_model(Path(checkpoint), self.device)
        self.cfg = ckpt["config"]
        self.checkpoint_epoch = ckpt.get("epoch")
        self.spec = PreprocessSpec.from_config(self.cfg)
        self.decode_spec: DecodeSpec = decode_spec(self.cfg)

    @property
    def modality(self) -> str:
        return self.spec.modality

    def predict(
        self, rgb: np.ndarray | None, depth: np.ndarray | None = None, max_candidates: int | None = None
    ) -> Prediction:
        """``rgb``: uint8 (H, W, 3); ``depth``: float (H, W) metres, NaN/<=0 = invalid."""
        ref = rgb if rgb is not None else depth
        if ref is None:
            raise ValueError("no image given")
        if self.modality in ("rgbd", "depth") and depth is None:
            raise ValueError(f"this model ({self.modality}) needs a depth image")
        if self.modality in ("rgbd", "rgb") and rgb is None:
            raise ValueError(f"this model ({self.modality}) needs an RGB image")
        h, w = ref.shape[:2]
        if depth is not None and rgb is not None and depth.shape[:2] != rgb.shape[:2]:
            raise ValueError(f"RGB {rgb.shape[:2]} and depth {depth.shape[:2]} sizes differ")
        S = self.spec.output_size
        M = build_affine(crop_for_image(self.spec, w, h), S)
        rgb_w = warp_image(rgb, M, S) if rgb is not None and self.modality != "depth" else None
        depth_w = None
        if depth is not None and self.modality != "rgb":
            depth_w = warp_image(inpaint_depth(mask_invalid_depth(depth)), M, S)
        x = torch.from_numpy(build_input(rgb_w, depth_w, self.modality))[None].to(self.device)
        maps = predict_maps(self.model, x)[0]
        dspec = (
            self.decode_spec
            if max_candidates is None
            else replace(self.decode_spec, max_candidates=max_candidates)
        )
        cands = [grasp_to_original(g, M) for g in decode_grasps(*maps, dspec)]
        names = ("quality", "cos2", "sin2", "width")
        return Prediction(cands, {n: maps[i] for i, n in enumerate(names)}, M, S)
