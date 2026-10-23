"""Create the reproducible split files and choose the fixed crop window from TRAINING data only.

Usage:
    python scripts/make_splits.py [--seed 42]

Writes:
    experiments/splits/image_wise_seed<seed>.json
    experiments/splits/block_wise_seed<seed>.json
    experiments/results/crop_analysis.json

The crop window (centre and side, original pixels) is chosen once from the image-wise TRAIN
split: the centre is the median of the positive grasp centres, and the side is the smallest
multiple of 16 px for which >= 99% of training positive-rectangle vertices fall inside. The
same crop is then used for every split and both strategies. No validation or test annotation
is used to choose it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_planner.config import REPO_ROOT, resolve_dataset_root  # noqa: E402
from grasp_planner.data.cornell import discover_samples  # noqa: E402
from grasp_planner.data.grasp import rect_to_grasp  # noqa: E402
from grasp_planner.data.splits import make_split, save_split  # noqa: E402
from grasp_planner.utils import setup_logging  # noqa: E402

RATIOS = (0.7, 0.15, 0.15)


def coverage(samples, cx: float, cy: float, side: float) -> dict:
    """Fraction of positive-rectangle vertices / whole rectangles / scenes inside the window."""
    h = side / 2
    v_in, r_in, n_v, n_r, scene_ok = 0, 0, 0, 0, 0
    for s in samples:
        inside = (np.abs(s.pos[..., 0] - cx) <= h) & (np.abs(s.pos[..., 1] - cy) <= h)  # (N, 4)
        v_in += int(inside.sum())
        n_v += inside.size
        r_in += int(inside.all(axis=1).sum())
        n_r += inside.shape[0]
        scene_ok += int(inside.all(axis=1).any())
    return {
        "vertex_fraction": v_in / n_v,
        "rectangle_fraction": r_in / n_r,
        "scenes_with_at_least_one_full_rectangle": scene_ok / len(samples),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-size", type=int, default=20)
    ap.add_argument("--gap", type=int, default=2)
    args = ap.parse_args()
    setup_logging()
    samples, _ = discover_samples(resolve_dataset_root())
    ids = [s.sample_id for s in samples]
    split_dir = REPO_ROOT / "experiments" / "splits"

    img = make_split(ids, "image", RATIOS, args.seed)
    blk = make_split(ids, "block", RATIOS, args.seed, block_size=args.block_size, gap=args.gap)
    save_split(img, split_dir / f"image_wise_seed{args.seed}.json")
    save_split(blk, split_dir / f"block_wise_seed{args.seed}.json")
    print("image-wise counts:", img["counts"])
    print("block-wise counts:", blk["counts"])

    # ---- crop window from image-wise TRAIN split only
    train_ids = set(img["splits"]["train"])
    train = [s for s in samples if s.sample_id in train_ids]
    centres = np.array([[g.x, g.y] for s in train for g in map(rect_to_grasp, s.pos)])
    cx, cy = (float(round(v)) for v in np.median(centres, axis=0))
    table = {}
    chosen = None
    for side in range(160, 481, 16):
        c = coverage(train, cx, cy, side)
        table[side] = c
        if chosen is None and c["vertex_fraction"] >= 0.99:
            chosen = side
    if chosen is None:
        chosen = 480
    # keep the window inside the 640 x 480 image where possible
    cy = float(min(max(cy, chosen / 2), 480 - chosen / 2))
    cx = float(min(max(cx, chosen / 2), 640 - chosen / 2))
    analysis = {
        "source": "image-wise train split only",
        "median_train_grasp_centre": [cx, cy],
        "chosen_crop": {"cx": cx, "cy": cy, "size": chosen},
        "rule": "smallest multiple of 16 px with >= 99% of train positive vertices inside",
        "train_coverage_by_side": table,
        "chosen_train_coverage": coverage(train, cx, cy, chosen),
    }
    out = REPO_ROOT / "experiments" / "results" / "crop_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=1), encoding="utf-8")
    print("chosen crop:", analysis["chosen_crop"], analysis["chosen_train_coverage"])


if __name__ == "__main__":
    main()
