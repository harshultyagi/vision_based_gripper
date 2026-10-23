"""Geometric preprocessing and augmentation for images and grasp rectangles.

Every spatial operation (crop, resize, rotation, zoom, translation jitter, horizontal flip)
is composed into **one** 2x3 affine matrix ``M`` that maps original image pixel coordinates
``(x, y, 1)`` to network-input pixel coordinates. The image is resampled once with
``cv2.warpAffine(img, M)`` and the grasp rectangle *vertices* are transformed by the same
``M``; grasp parameters are then recomputed from the transformed vertices. Because the
vertices are transformed exactly like the pixels, angle, width and centre stay consistent by
construction (including the angle sign flip under mirroring). Predictions are mapped back to
the original image with ``invert_affine(M)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CropSpec:
    """Fixed crop window in the *original* image: centre (cx, cy) and square side length."""

    cx: float
    cy: float
    size: float


@dataclass(frozen=True)
class AugmentSpec:
    """Random augmentation ranges (training only)."""

    rotation_deg: float = 0.0  # rotation sampled uniformly in [-rotation_deg, rotation_deg]
    zoom_min: float = 1.0  # crop side multiplied by a factor in [zoom_min, 1] (zooming in)
    translate_px: float = 0.0  # crop-centre jitter in original pixels, per axis
    hflip_prob: float = 0.0


def build_affine(
    crop: CropSpec,
    output_size: int,
    rotation_rad: float = 0.0,
    zoom: float = 1.0,
    shift: tuple[float, float] = (0.0, 0.0),
    hflip: bool = False,
) -> np.ndarray:
    """Compose the affine map original-pixels -> output-pixels.

    Steps, applied to a point ``p``:
      1. translate so the (jittered) crop centre is the origin;
      2. rotate by ``rotation_rad`` counter-clockwise *on screen*;
      3. scale by ``output_size / (crop.size * zoom)``;
      4. optionally mirror horizontally (x -> -x);
      5. translate the origin to the output centre ``((S-1)/2, (S-1)/2)``.

    On-screen counter-clockwise rotation in a y-down frame is the matrix
    ``[[cos a, sin a], [-sin a, cos a]]``.
    """
    cx, cy = crop.cx + shift[0], crop.cy + shift[1]
    s = output_size / (crop.size * zoom)
    c, sn = np.cos(rotation_rad), np.sin(rotation_rad)
    rot = np.array([[c, sn], [-sn, c]])
    lin = s * rot
    if hflip:
        lin = np.diag([-1.0, 1.0]) @ lin
    half = (output_size - 1) / 2.0
    t = np.array([half, half]) - lin @ np.array([cx, cy])
    return np.hstack([lin, t[:, None]])


def sample_affine(
    crop: CropSpec, output_size: int, aug: AugmentSpec | None, rng: np.random.Generator | None
) -> np.ndarray:
    """Draw a random augmentation (or the identity crop if ``aug``/``rng`` is None)."""
    if aug is None or rng is None:
        return build_affine(crop, output_size)
    rot = np.deg2rad(rng.uniform(-aug.rotation_deg, aug.rotation_deg)) if aug.rotation_deg else 0.0
    zoom = rng.uniform(aug.zoom_min, 1.0) if aug.zoom_min < 1.0 else 1.0
    shift = tuple(rng.uniform(-aug.translate_px, aug.translate_px, size=2)) if aug.translate_px else (0, 0)
    flip = bool(rng.random() < aug.hflip_prob)
    return build_affine(crop, output_size, rot, zoom, shift, flip)


def apply_affine_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to points of shape (..., 2)."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ M[:, :2].T + M[:, 2]


def invert_affine(M: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 affine matrix."""
    return cv2.invertAffineTransform(np.asarray(M, dtype=np.float64))


def warp_image(
    img: np.ndarray, M: np.ndarray, output_size: int, interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Resample ``img`` into the output frame; pixels from outside the image replicate the border."""
    return cv2.warpAffine(
        img, M, (output_size, output_size), flags=interpolation, borderMode=cv2.BORDER_REPLICATE
    )


def inpaint_depth(depth: np.ndarray) -> np.ndarray:
    """Fill NaN depth pixels with OpenCV Navier-Stokes inpainting (``cv2.INPAINT_NS``), the
    same approach as the GG-CNN reference code.

    The depth is scaled to [0, 1] for inpainting (OpenCV requires 8/32-bit single channel with a
    bounded range) and scaled back afterwards. A 1-pixel border is added so missing values on
    the image edge are filled from the interior.
    """
    d = np.pad(depth, 1, mode="constant", constant_values=np.nan).astype(np.float32)
    mask = (~np.isfinite(d)).astype(np.uint8)
    if not mask.any():
        return depth.astype(np.float32)
    valid = d[mask == 0]
    if valid.size == 0:
        raise ValueError("depth image has no valid pixels")
    scale = float(np.abs(valid).max())
    d = np.where(mask == 1, 0.0, d / scale).astype(np.float32)
    d = cv2.inpaint(d, mask, 1, cv2.INPAINT_NS)
    return (d[1:-1, 1:-1] * scale).astype(np.float32)
