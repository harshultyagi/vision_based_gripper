"""Planar grasp geometry: grasp rectangles, the (center, angle, width) representation,
contact points, rectangle IoU and angle differences.

Conventions (used everywhere in the project)
--------------------------------------------
* Pixel coordinates ``(x, y)``: origin at the top-left corner, ``x`` = column (to the right),
  ``y`` = row (downwards).
* ``theta``: angle of the gripper *closing direction* (the line joining the two jaws), measured
  counter-clockwise from the +x axis **as seen on screen**. Because image ``y`` points down,
  ``theta = atan2(-dy, dx)``. Wrapped to ``[-pi/2, pi/2)`` because a parallel-jaw grasp is
  unchanged by a rotation of ``pi``.
* ``width``: gripper opening in pixels = distance between the two jaw contact points.
* ``jaw_size``: extent of each jaw plate perpendicular to the closing direction (pixels).

Cornell rectangles are four vertices ``p0, p1, p2, p3``. The edge ``p0 -> p1`` runs along the
gripper closing direction (its length is the opening), ``p1 -> p2`` is a jaw plate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

HALF_PI = np.pi / 2


def wrap_angle(theta: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (radians) to ``[-pi/2, pi/2)``, the canonical range of a grasp angle."""
    return (theta + HALF_PI) % np.pi - HALF_PI


def angle_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two grasp angles, respecting the pi symmetry.

    Result is in ``[0, pi/2]``. Example: 80 deg vs -80 deg -> 20 deg.
    """
    d = abs(a - b) % np.pi
    return float(min(d, np.pi - d))


@dataclass(frozen=True)
class Grasp:
    """A planar parallel-jaw grasp in pixel coordinates.

    Attributes:
        x, y: grasp centre (pixels).
        theta: closing-direction angle in radians, ``[-pi/2, pi/2)``.
        width: gripper opening (pixels).
        jaw_size: jaw plate size (pixels); for predictions this is set to ``width / 2``
            following the GG-CNN / GR-ConvNet evaluation convention.
        score: model quality score (only meaningful for predictions).
    """

    x: float
    y: float
    theta: float
    width: float
    jaw_size: float
    score: float = float("nan")

    @property
    def center(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def direction(self) -> np.ndarray:
        """Unit vector (dx, dy) along the closing direction, in image (y-down) coordinates."""
        return np.array([np.cos(self.theta), -np.sin(self.theta)])

    def contact_points(self) -> np.ndarray:
        """The two jaw contact points ``[[x1, y1], [x2, y2]]`` = centre -/+ (width/2) * direction."""
        c = np.array([self.x, self.y])
        h = 0.5 * self.width * self.direction
        return np.stack([c - h, c + h])

    def polygon(self) -> np.ndarray:
        """Rectangle vertices (4, 2) in the Cornell vertex order p0..p3, as (x, y)."""
        d = self.direction
        n = np.array([-d[1], d[0]])  # perpendicular (jaw direction)
        c = np.array([self.x, self.y])
        hw, hj = 0.5 * self.width, 0.5 * self.jaw_size
        p0 = c - hw * d - hj * n
        p1 = c + hw * d - hj * n
        p2 = c + hw * d + hj * n
        p3 = c - hw * d + hj * n
        return np.stack([p0, p1, p2, p3])

    def iou(self, other: Grasp) -> float:
        return polygon_iou(self.polygon(), other.polygon())


def rect_to_grasp(points: np.ndarray) -> Grasp:
    """Convert a Cornell rectangle (4, 2) array of (x, y) vertices into a :class:`Grasp`."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError(f"expected (4, 2) rectangle, got {pts.shape}")
    dx, dy = pts[1] - pts[0]
    theta = wrap_angle(float(np.arctan2(-dy, dx)))
    width = float(np.hypot(dx, dy))
    jaw_size = float(np.linalg.norm(pts[2] - pts[1]))
    cx, cy = pts.mean(axis=0)
    return Grasp(float(cx), float(cy), float(theta), width, jaw_size)


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Exact intersection-over-union of two convex polygons given as (N, 2) arrays."""
    pa, pb = Polygon(a), Polygon(b)
    if not pa.is_valid or not pb.is_valid:
        pa, pb = pa.buffer(0), pb.buffer(0)
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0


def rectangle_geometry_errors(points: np.ndarray, tol_deg: float = 10.0) -> list[str]:
    """Return a list of problems with a Cornell rectangle (empty list = valid).

    Checks: finite coordinates, non-degenerate sides, approximately right-angled corners
    (within ``tol_deg``). Cornell rectangles were hand-labelled so corners are not exact.
    """
    pts = np.asarray(points, dtype=np.float64)
    errors: list[str] = []
    if pts.shape != (4, 2):
        return [f"shape {pts.shape} != (4, 2)"]
    if not np.all(np.isfinite(pts)):
        return ["non-finite coordinate (NaN/inf)"]
    for i in range(4):
        e1 = pts[(i + 1) % 4] - pts[i]
        e0 = pts[i] - pts[i - 1]
        n0, n1 = np.linalg.norm(e0), np.linalg.norm(e1)
        if n0 < 1e-6 or n1 < 1e-6:
            errors.append("degenerate (zero-length) side")
            break
        cosang = abs(float(np.dot(e0, e1) / (n0 * n1)))
        if cosang > np.sin(np.deg2rad(tol_deg)):
            errors.append(f"corner {i} deviates from 90 deg by more than {tol_deg} deg")
            break
    return errors


def grasp_matches(pred: Grasp, gt: Grasp, iou_thresh: float = 0.25, angle_thresh_deg: float = 30.0) -> bool:
    """Cornell rectangle metric: IoU > ``iou_thresh`` and angle difference < ``angle_thresh_deg``."""
    if angle_difference(pred.theta, gt.theta) >= np.deg2rad(angle_thresh_deg):
        return False
    return pred.iou(gt) > iou_thresh
