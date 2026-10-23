"""Predict grasps for one image with a trained checkpoint.

Usage:
    python scripts/predict.py --checkpoint outputs/runs/<name>/checkpoints/best.pt \
        --rgb path/to/pcd0123r.png [--depth path/to/pcd0123d.tiff] [--gt path/to/pcd0123cpos.txt] \
        [--num-candidates 5] [--out outputs/predictions/pcd0123.png]

Prints the ranked candidates (centre, angle, width, score, contact points) as JSON and saves
an annotated image plus a figure of the raw output maps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_planner.config import REPO_ROOT  # noqa: E402
from grasp_planner.data.cornell import load_depth, load_rgb, parse_rectangle_file  # noqa: E402
from grasp_planner.data.transforms import warp_image  # noqa: E402
from grasp_planner.inference.predictor import GraspPredictor  # noqa: E402
from grasp_planner.visualization.plots import render_prediction, save_maps_figure  # noqa: E402


def grasp_record(rank: int, g) -> dict:
    cps = g.contact_points()
    return {
        "rank": rank,
        "center_xy_px": [round(g.x, 2), round(g.y, 2)],
        "angle_deg": round(float(np.rad2deg(g.theta)), 2),
        "width_px": round(g.width, 2),
        "score": round(g.score, 4),
        "contact_points_xy_px": [[round(float(v), 2) for v in p] for p in cps],
    }


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--rgb", default=None)
    ap.add_argument("--depth", default=None)
    ap.add_argument("--gt", default=None, help="optional cpos.txt to overlay ground truth")
    ap.add_argument("--num-candidates", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    pred = GraspPredictor(args.checkpoint, torch.device(args.device) if args.device else None)
    rgb = load_rgb(Path(args.rgb)) if args.rgb else None
    depth = load_depth(Path(args.depth)) if args.depth else None
    result = pred.predict(rgb, depth, max_candidates=args.num_candidates)
    if not result.candidates:
        print("No valid grasp candidate above the quality threshold.")
    records = [grasp_record(i + 1, g) for i, g in enumerate(result.candidates)]
    print(
        json.dumps(
            {"model": pred.cfg["experiment"]["name"], "modality": pred.modality, "candidates": records},
            indent=1,
        )
    )

    stem = Path(args.rgb or args.depth).stem
    out = Path(args.out) if args.out else REPO_ROOT / "outputs" / "predictions" / f"{stem}_pred.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    gt = parse_rectangle_file(Path(args.gt)).rects if args.gt else None
    base = (
        rgb
        if rgb is not None
        else cv2.cvtColor(
            cv2.normalize(np.nan_to_num(depth), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2RGB,
        )
    )
    img = render_prediction(base, result.candidates, gt, None, args.num_candidates)
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    crop_rgb = warp_image(rgb, result.M, result.input_size) if rgb is not None else None
    save_maps_figure(crop_rgb, result.maps, out.with_name(out.stem + "_maps.png"), title=stem)
    print("saved", out)


if __name__ == "__main__":
    main()
