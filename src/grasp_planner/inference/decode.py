"""Decode dense network outputs into ranked grasp candidates.

Pipeline (all in network-input pixel coordinates, then optionally mapped to the original image):

1. **Post-process maps.** ``theta = 0.5 * atan2(sin2, cos2)``; ``width = width_map * max_width``;
   quality and width are smoothed with a Gaussian (sigma ``q_sigma``) to suppress
   single-pixel spikes, as in GG-CNN / GR-ConvNet.
2. **Peak extraction.** Local maxima of the smoothed quality map inside a
   ``(2 * min_distance + 1)`` window whose value is >= ``q_threshold``; sorted by quality.
3. **Validity checks.** Width finite and in ``[min_width, max_width]``; centre inside the
   image; ``theta`` finite.
4. **Non-maximum suppression.** Walk the peaks in decreasing quality and drop any candidate
   whose rectangle overlaps an already accepted one with IoU > ``nms_iou`` *and* whose angle
   is within ``nms_angle_deg``; this removes near-duplicates while keeping distinct grasps that
   share a location but differ in orientation.
5. **Ranking.** The remaining candidates are ranked by quality score; the first is the
   *selected* grasp. The score is the network's quality output, not a calibrated success
   probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from ..data.grasp import Grasp, angle_difference, rect_to_grasp
from ..data.transforms import apply_affine_points, invert_affine


@dataclass(frozen=True)
class DecodeSpec:
    q_sigma: float = 2.0
    q_threshold: float = 0.1
    min_distance: int = 5
    max_peaks: int = 50
    min_width: float = 2.0
    nms_iou: float = 0.3
    nms_angle_deg: float = 30.0
    max_candidates: int = 10
    max_width: float = 150.0

    @staticmethod
    def from_config(cfg: dict, max_width: float) -> DecodeSpec:
        return DecodeSpec(**{**cfg, "max_width": max_width})


def post_process(
    quality: np.ndarray, cos2: np.ndarray, sin2: np.ndarray, width: np.ndarray, spec: DecodeSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw (H, W) maps -> (smoothed quality, angle in radians, width in pixels)."""
    q = gaussian_filter(quality.astype(np.float64), spec.q_sigma)
    ang = 0.5 * np.arctan2(sin2, cos2)
    w = gaussian_filter(width.astype(np.float64) * spec.max_width, 1.0)
    return q, ang, w


def find_peaks(q: np.ndarray, spec: DecodeSpec) -> list[tuple[int, int]]:
    """Local maxima (row, col) of ``q`` above threshold, highest first."""
    size = 2 * spec.min_distance + 1
    is_peak = (q == maximum_filter(q, size=size, mode="nearest")) & (q >= spec.q_threshold)
    rows, cols = np.nonzero(is_peak)
    order = np.argsort(-q[rows, cols], kind="stable")
    return [(int(rows[i]), int(cols[i])) for i in order[: spec.max_peaks]]


def is_valid(g: Grasp, shape: tuple[int, int], spec: DecodeSpec) -> bool:
    h, w = shape
    return (
        np.isfinite([g.x, g.y, g.theta, g.width, g.score]).all()
        and spec.min_width <= g.width <= spec.max_width
        and 0 <= g.x <= w - 1
        and 0 <= g.y <= h - 1
    )


def nms(cands: list[Grasp], spec: DecodeSpec) -> list[Grasp]:
    keep: list[Grasp] = []
    for g in cands:  # already sorted by score
        dup = any(
            angle_difference(g.theta, k.theta) < np.deg2rad(spec.nms_angle_deg) and g.iou(k) > spec.nms_iou
            for k in keep
        )
        if not dup:
            keep.append(g)
        if len(keep) >= spec.max_candidates:
            break
    return keep


def decode_grasps(
    quality: np.ndarray, cos2: np.ndarray, sin2: np.ndarray, width: np.ndarray, spec: DecodeSpec
) -> list[Grasp]:
    """Full decode: maps (H, W) -> ranked candidates in input-pixel coordinates."""
    q, ang, w = post_process(quality, cos2, sin2, width, spec)
    cands = []
    for r, c in find_peaks(q, spec):
        g = Grasp(float(c), float(r), float(ang[r, c]), float(w[r, c]), float(w[r, c]) / 2.0, float(q[r, c]))
        if is_valid(g, q.shape, spec):
            cands.append(g)
    return nms(cands, spec)


def grasp_to_original(g: Grasp, M: np.ndarray) -> Grasp:
    """Map a grasp from network-input coordinates back to the original image using the
    inverse of the preprocessing affine ``M``. Done on the rectangle vertices, so rotation,
    scale and mirroring are all undone consistently. The score is carried over."""
    pts = apply_affine_points(invert_affine(M), g.polygon())
    o = rect_to_grasp(pts)
    return Grasp(o.x, o.y, o.theta, o.width, o.jaw_size, g.score)
