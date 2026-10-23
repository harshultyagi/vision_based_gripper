"""Cornell evaluation metrics.

Per-image metrics (aggregated as the mean over images):

* ``top1_success``: 1 if the top-ranked predicted grasp matches *any* positive ground-truth
  rectangle under the rectangle metric (IoU > 0.25 and angle difference < 30 deg, angles
  compared modulo pi), else 0. Images with no prediction count as failures.
* ``topk_success``: 1 if any of the first k candidates matches any GT rectangle.
* ``center_error_px``: Euclidean distance from the top prediction's centre to the *nearest*
  GT rectangle centre (original-image pixels).
* ``angle_error_deg``: angle difference to that same nearest-centre GT rectangle.
* ``width_error_px``: |predicted opening - GT opening| for that same GT rectangle.
* ``best_iou``: max IoU between the top prediction and any GT rectangle.

Pairing the error metrics with the nearest-centre GT rectangle is a choice made here (there is
no single "correct" GT for a multi-annotation image); it is documented in
``docs/evaluation_protocol.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..data.grasp import Grasp, angle_difference, grasp_matches


@dataclass
class ImageResult:
    sample_id: str
    num_candidates: int
    top1_success: bool
    topk_success: bool
    best_iou: float
    center_error_px: float
    angle_error_deg: float
    width_error_px: float
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_image(
    sample_id: str,
    preds: list[Grasp],
    gts: list[Grasp],
    k: int = 5,
    iou_thresh: float = 0.25,
    angle_thresh_deg: float = 30.0,
) -> ImageResult:
    if not preds:
        nan = float("nan")
        return ImageResult(sample_id, 0, False, False, 0.0, nan, nan, nan, nan)
    top = preds[0]

    def matches(p: Grasp) -> bool:
        return any(grasp_matches(p, g, iou_thresh, angle_thresh_deg) for g in gts)

    top1 = matches(top)
    topk = top1 or any(matches(p) for p in preds[1:k])
    dists = [np.hypot(top.x - g.x, top.y - g.y) for g in gts]
    j = int(np.argmin(dists))
    near = gts[j]
    return ImageResult(
        sample_id=sample_id,
        num_candidates=len(preds),
        top1_success=bool(top1),
        topk_success=bool(topk),
        best_iou=float(max(top.iou(g) for g in gts)),
        center_error_px=float(dists[j]),
        angle_error_deg=float(np.rad2deg(angle_difference(top.theta, near.theta))),
        width_error_px=float(abs(top.width - near.width)),
        score=float(top.score),
    )


def summarise(results: list[ImageResult], k: int) -> dict:
    """Aggregate per-image results (mean over images; errors use images with a prediction)."""
    n = len(results)
    have = [r for r in results if r.num_candidates > 0]

    def stat(name: str) -> dict:
        v = np.array([getattr(r, name) for r in have], dtype=np.float64)
        if v.size == 0:
            return {"mean": None, "median": None}
        return {"mean": float(v.mean()), "median": float(np.median(v))}

    return {
        "num_images": n,
        "num_images_with_prediction": len(have),
        "top1_accuracy": float(np.mean([r.top1_success for r in results])) if n else None,
        f"top{k}_accuracy": float(np.mean([r.topk_success for r in results])) if n else None,
        "best_iou": stat("best_iou"),
        "center_error_px": stat("center_error_px"),
        "angle_error_deg": stat("angle_error_deg"),
        "width_error_px": stat("width_error_px"),
        "mean_num_candidates": float(np.mean([r.num_candidates for r in results])) if n else None,
    }
