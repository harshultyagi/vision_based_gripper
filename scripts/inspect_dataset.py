"""Inspect the local Cornell dataset: structure, annotation statistics, exclusions, depth
validity, and diagnostic visualisations. Read-only with respect to the dataset.

Usage:
    python scripts/inspect_dataset.py [--dataset-root PATH]

Writes:
    experiments/results/dataset_stats.json
    experiments/results/dataset_exclusions.json
    docs/figures/dataset_examples.png, dataset_histograms.png, dataset_depth_example.png
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_planner.config import REPO_ROOT, resolve_dataset_root  # noqa: E402
from grasp_planner.data.cornell import discover_samples, load_depth, load_rgb  # noqa: E402
from grasp_planner.data.grasp import rect_to_grasp  # noqa: E402
from grasp_planner.data.transforms import inpaint_depth  # noqa: E402
from grasp_planner.utils import setup_logging  # noqa: E402


def describe(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return {
        "count": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "p5": float(np.percentile(v, 5)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--depth-sample", type=int, default=0, help="check depth of every Nth scene (0 = all)")
    args = ap.parse_args()
    setup_logging()
    root = resolve_dataset_root(args.dataset_root)
    out_dir = REPO_ROOT / "experiments" / "results"
    fig_dir = REPO_ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- structure
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    file_kinds: dict[str, collections.Counter] = {}
    for f in folders:
        c = collections.Counter()
        for p in f.iterdir():
            name = p.name
            if name.endswith("cpos.txt"):
                c["cpos.txt"] += 1
            elif name.endswith("cneg.txt"):
                c["cneg.txt"] += 1
            elif name.endswith("r.png"):
                c["r.png"] += 1
            elif name.endswith("d.tiff"):
                c["d.tiff"] += 1
            elif name.endswith(".txt"):
                c["pointcloud.txt"] += 1
            else:
                c["other:" + p.suffix] += 1
        file_kinds[f.name] = dict(c)

    samples, exclusions = discover_samples(root)

    # ---- annotation statistics
    pos_counts = np.array([len(s.pos) for s in samples])
    neg_counts = np.array([len(s.neg) for s in samples])
    grasps = [rect_to_grasp(r) for s in samples for r in s.pos]
    widths = np.array([g.width for g in grasps])
    jaws = np.array([g.jaw_size for g in grasps])
    angles = np.rad2deg([g.theta for g in grasps])
    cx = np.array([g.x for g in grasps])
    cy = np.array([g.y for g in grasps])
    all_pts = np.concatenate([s.pos.reshape(-1, 2) for s in samples])
    rect_issues = [{"sample_id": s.sample_id, "issue": i} for s in samples for i in s.issues]

    # ---- image sizes and depth validity
    import cv2

    sizes = collections.Counter()
    depth_info = {"checked": 0, "nan_fraction": [], "min_m": [], "max_m": [], "dtype": set(), "shape": set()}
    step = args.depth_sample or 1
    for i, s in enumerate(samples):
        im = cv2.imread(str(s.rgb_path), cv2.IMREAD_UNCHANGED)
        sizes[f"{im.shape[1]}x{im.shape[0]}x{im.shape[2] if im.ndim == 3 else 1}"] += 1
        if s.depth_path is not None and i % step == 0:
            import tifffile

            raw = tifffile.imread(str(s.depth_path))
            d = load_depth(s.depth_path)
            depth_info["checked"] += 1
            depth_info["dtype"].add(str(raw.dtype))
            depth_info["shape"].add(str(raw.shape))
            depth_info["nan_fraction"].append(float(np.isnan(d).mean()))
            depth_info["min_m"].append(float(np.nanmin(d)))
            depth_info["max_m"].append(float(np.nanmax(d)))
    depth_summary = {
        "checked_scenes": depth_info["checked"],
        "dtype": sorted(depth_info["dtype"]),
        "shape": sorted(depth_info["shape"]),
        "invalid_pixel_fraction": describe(depth_info["nan_fraction"]),
        "min_depth": describe(depth_info["min_m"]),
        "max_depth": describe(depth_info["max_m"]),
    }

    backgrounds = sorted((root / "backgrounds").glob("*.png")) if (root / "backgrounds").exists() else []
    stats = {
        "dataset_root_name": root.name,
        "folders": file_kinds,
        "num_scene_folders": sum(1 for f in folders if f.name.isdigit()),
        "num_background_images": len(backgrounds),
        "num_usable_scenes": len(samples),
        "num_excluded_scenes": len(exclusions),
        "num_scenes_with_depth": sum(s.depth_path is not None for s in samples),
        "image_sizes": dict(sizes),
        "scene_id_range": [samples[0].sample_id, samples[-1].sample_id],
        "positive_rectangles": {"total": int(pos_counts.sum()), "per_scene": describe(pos_counts)},
        "negative_rectangles": {"total": int(neg_counts.sum()), "per_scene": describe(neg_counts)},
        "rejected_rectangles": len(rect_issues),
        "positive_grasp_width_px": describe(widths),
        "positive_grasp_jaw_size_px": describe(jaws),
        "positive_grasp_angle_deg": describe(angles),
        "positive_grasp_center_x": describe(cx),
        "positive_grasp_center_y": describe(cy),
        "positive_vertex_bbox": {
            "x_min": float(all_pts[:, 0].min()),
            "x_max": float(all_pts[:, 0].max()),
            "y_min": float(all_pts[:, 1].min()),
            "y_max": float(all_pts[:, 1].max()),
        },
        "depth": depth_summary,
        "object_identity": "not available: the local copy has no object-ID mapping file (z.txt)",
    }
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "dataset_exclusions.json").write_text(
        json.dumps({"excluded_scenes": exclusions, "rejected_rectangles": rect_issues}, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                k: stats[k]
                for k in (
                    "num_usable_scenes",
                    "num_excluded_scenes",
                    "rejected_rectangles",
                    "image_sizes",
                    "positive_rectangles",
                    "negative_rectangles",
                )
            },
            indent=1,
        )
    )
    print("depth:", json.dumps(depth_summary, indent=1))

    # ---- figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from grasp_planner.visualization.plots import GT_NEG, GT_POS, draw_rect_points

    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(samples), 8, replace=False))
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4))
    for ax, i in zip(axes.ravel(), pick, strict=True):
        s = samples[i]
        img = load_rgb(s.rgb_path)
        for r in s.neg:
            draw_rect_points(img, r, GT_NEG, 1)
        for r in s.pos:
            draw_rect_points(img, r, GT_POS, 2)
        ax.imshow(img)
        ax.set_title(f"pcd{s.sample_id}: {len(s.pos)} pos / {len(s.neg)} neg", fontsize=9)
        ax.axis("off")
    fig.suptitle("Cornell annotations (green = positive, red = negative rectangles)")
    fig.tight_layout()
    fig.savefig(fig_dir / "dataset_examples.png", dpi=90)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.4))
    axes[0].hist(widths, 50)
    axes[0].set(title="opening width (px)", xlabel="px")
    axes[1].hist(jaws, 50)
    axes[1].set(title="jaw size (px)", xlabel="px")
    axes[2].hist(angles, 36)
    axes[2].set(title="angle theta (deg)", xlabel="deg")
    axes[3].hist(pos_counts, np.arange(0, pos_counts.max() + 2) - 0.5)
    axes[3].set(title="positive rectangles per scene")
    fig.tight_layout()
    fig.savefig(fig_dir / "dataset_histograms.png", dpi=100)
    plt.close(fig)

    s = samples[pick[0]]
    if s.depth_path is not None:
        d = load_depth(s.depth_path)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        axes[0].imshow(load_rgb(s.rgb_path))
        axes[0].set_title(f"pcd{s.sample_id} RGB")
        im = axes[1].imshow(d, cmap="viridis")
        axes[1].set_title(f"raw depth ({np.isnan(d).mean() * 100:.1f}% invalid = white)")
        fig.colorbar(im, ax=axes[1], fraction=0.035)
        im = axes[2].imshow(inpaint_depth(d), cmap="viridis")
        axes[2].set_title("after inpainting")
        fig.colorbar(im, ax=axes[2], fraction=0.035)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "dataset_depth_example.png", dpi=90)
        plt.close(fig)
    print("wrote", out_dir / "dataset_stats.json")


if __name__ == "__main__":
    main()
