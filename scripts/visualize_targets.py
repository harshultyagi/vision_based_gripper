"""Diagnostic figure: network inputs and target maps for a few TRAINING scenes, with and without
augmentation, with the transformed GT rectangles overlaid. Used to verify that augmentation
keeps images and labels aligned.

Usage:
    python scripts/visualize_targets.py --config configs/experiment_configs/grconvnet_rgbd_image.yaml
Writes docs/figures/training_targets.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_planner.config import REPO_ROOT, load_config  # noqa: E402
from grasp_planner.data.cornell import load_rgb  # noqa: E402
from grasp_planner.data.transforms import apply_affine_points, warp_image  # noqa: E402
from grasp_planner.pipeline import make_dataset, split_samples  # noqa: E402
from grasp_planner.visualization.plots import GT_POS, draw_rect_points  # noqa: E402


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment_configs/grconvnet_rgbd_image.yaml")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train = split_samples(cfg)["train"][: args.n]
    rows = []
    for aug in (False, True):
        ds = make_dataset(cfg, train, train=aug)
        ds.set_epoch(1)
        for i in range(len(train)):
            rows.append((aug, ds, i))
    fig, axes = plt.subplots(len(rows), 5, figsize=(16, 3.2 * len(rows)))
    for r, (aug, ds, i) in enumerate(rows):
        item = ds[i]
        M = item["M"].numpy()
        S = ds.spec.output_size
        rgb = warp_image(load_rgb(ds.samples[i].rgb_path), M, S)
        for rect in ds.samples[i].pos:
            draw_rect_points(rgb, apply_affine_points(M, rect), GT_POS, 1)
        y = item["y"].numpy()
        ang = np.where(y[0] > 0, np.rad2deg(0.5 * np.arctan2(y[2], y[1])), np.nan)
        panels = [
            (rgb, f"pcd{ds.samples[i].sample_id} {'augmented' if aug else 'no aug'}", None, None),
            (item["x"][0].numpy(), "depth input (normalised)", "gray", None),
            (y[0], "quality target", "viridis", (0, 1)),
            (ang, "angle target [deg]", "hsv", (-90, 90)),
            (np.where(y[0] > 0, y[3] * ds.spec.max_width, np.nan), "width target [px]", "magma", None),
        ]
        for ax, (img, title, cmap, lim) in zip(axes[r], panels, strict=True):
            im = ax.imshow(img, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            if cmap and title != "depth input (normalised)":
                fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = REPO_ROOT / "docs" / "figures" / "training_targets.png"
    fig.savefig(out, dpi=80)
    print("wrote", out)


if __name__ == "__main__":
    main()
