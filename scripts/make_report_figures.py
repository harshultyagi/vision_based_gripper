"""Figures for the technical report, generated from a trained checkpoint and saved results.

Usage:
    python scripts/make_report_figures.py --run outputs/runs/<name> --split test [--decode-id 0123]

Writes into docs/figures/:
    decode_pipeline_<run>.png    step-by-step decoding of one scene of the split
    qualitative_<run>_<split>.png  grid of GT vs prediction overlays (from per-image results CSV)
    background_<run>.png         predictions on background-only images (no object, no GT)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_planner.config import REPO_ROOT, resolve_dataset_root  # noqa: E402
from grasp_planner.data.cornell import load_depth, load_rgb  # noqa: E402
from grasp_planner.data.transforms import apply_affine_points, warp_image  # noqa: E402
from grasp_planner.inference.decode import find_peaks, is_valid, nms, post_process  # noqa: E402
from grasp_planner.inference.predictor import GraspPredictor  # noqa: E402
from grasp_planner.pipeline import split_samples  # noqa: E402
from grasp_planner.visualization.plots import (  # noqa: E402
    BEST,
    CAND,
    GT_POS,
    draw_grasp,
    draw_rect_points,
    render_prediction,
)

FIG = REPO_ROOT / "docs" / "figures"


def decode_figure(pred: GraspPredictor, sample, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from grasp_planner.data.grasp import Grasp

    rgb = load_rgb(sample.rgb_path)
    depth = load_depth(sample.depth_path) if pred.modality != "rgb" else None
    res = pred.predict(rgb, depth)
    m, spec = res.maps, pred.decode_spec
    q, ang, w = post_process(m["quality"], m["cos2"], m["sin2"], m["width"], spec)
    peaks = find_peaks(q, spec)
    raw = [
        Grasp(float(c), float(r), float(ang[r, c]), float(w[r, c]), float(w[r, c]) / 2, float(q[r, c]))
        for r, c in peaks
    ]
    valid = [g for g in raw if is_valid(g, q.shape, spec)]
    kept = nms(valid, spec)
    crop = warp_image(rgb, res.M, res.input_size)
    cand_img = crop.copy()
    for g in valid:
        draw_grasp(cand_img, g, (160, 160, 160), 1, 1)
    for i, g in reversed(list(enumerate(kept))):
        draw_grasp(cand_img, g, BEST if i == 0 else CAND, 2 if i == 0 else 1, 3 if i == 0 else 2)
    final = render_prediction(rgb, res.candidates, sample.pos, None, num_candidates=5)
    crop_gt = crop.copy()
    for r in sample.pos:
        draw_rect_points(crop_gt, apply_affine_points(res.M, r), GT_POS, 1)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    axes[0].imshow(crop_gt)
    axes[0].set_title(f"(a) 224x224 input crop + GT (pcd{sample.sample_id})")
    im = axes[1].imshow(m["quality"], cmap="viridis")
    axes[1].set_title("(b) raw quality map")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    im = axes[2].imshow(q, cmap="viridis")
    pr = np.array(peaks) if peaks else np.zeros((0, 2))
    axes[2].scatter(pr[:, 1], pr[:, 0], s=14, c="red")
    axes[2].set_title(f"(c) smoothed quality, {len(peaks)} local maxima >= {spec.q_threshold}")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].imshow(cand_img)
    axes[3].set_title(
        f"(d) {len(valid)} valid (grey) -> {len(kept)} after NMS\n(blue = rank 1, amber = others)"
    )
    axes[4].imshow(final)
    axes[4].set_title("(e) original image: selected (blue), contacts (magenta)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=85)
    plt.close(fig)
    lines = [
        f"{i + 1}: score {g.score:.3f} centre ({g.x:.1f},{g.y:.1f}) angle {np.rad2deg(g.theta):.1f} "
        f"width {g.width:.1f}"
        for i, g in enumerate(res.candidates[:5])
    ]
    print("decode figure candidates (original px):\n  " + "\n  ".join(lines))


def qualitative_grid(pred: GraspPredictor, samples, per_image_csv: Path, out: Path, n_each: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with per_image_csv.open(encoding="utf-8") as f:
        rows = {r["sample_id"]: r for r in csv.DictReader(f)}
    by_id = {s.sample_id: s for s in samples}
    rng = np.random.default_rng(0)
    succ = [i for i, r in rows.items() if r["top1_success"] == "True"]
    fail = [i for i, r in rows.items() if r["top1_success"] != "True"]
    pick = [("success", i) for i in rng.choice(succ, min(n_each, len(succ)), replace=False)]
    pick += [("failure", i) for i in rng.choice(fail, min(n_each, len(fail)), replace=False)]
    cols = 4
    nrows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(4.4 * cols, 3.6 * nrows))
    for ax, (tag, sid) in zip(np.atleast_1d(axes).ravel(), pick, strict=False):
        s = by_id[sid]
        rgb = load_rgb(s.rgb_path)
        res = pred.predict(rgb, load_depth(s.depth_path) if pred.modality != "rgb" else None)
        img = render_prediction(rgb, res.candidates, s.pos, None, num_candidates=3)
        x0, y0, x1, y1 = crop_box(pred)
        ax.imshow(img[y0:y1, x0:x1])
        r = rows[sid]
        ax.set_title(
            f"{tag}: pcd{sid}  IoU {float(r['best_iou']):.2f}  dθ {float(r['angle_error_deg']):.0f}°",
            fontsize=9,
            color="green" if tag == "success" else "red",
        )
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(pick) :]:
        ax.axis("off")
    fig.suptitle(
        "Green: GT positives · blue: selected grasp · amber: ranks 2-3 · magenta: contact points "
        "(shown inside the network's crop window)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=85)
    plt.close(fig)


def crop_box(pred: GraspPredictor) -> tuple[int, int, int, int]:
    c = pred.spec.crop
    h = c.size / 2
    return int(max(c.cx - h, 0)), int(max(c.cy - h, 0)), int(min(c.cx + h, 640)), int(min(c.cy + h, 480))


def background_figure(pred: GraspPredictor, out: Path) -> None:
    """Background images have no object and no depth file: only an RGB model can run on them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg = sorted((resolve_dataset_root() / "backgrounds").glob("*.png"))[:4]
    if not bg or pred.modality != "rgb":
        print("background figure skipped (needs an RGB-only model and background images)")
        return
    fig, axes = plt.subplots(1, len(bg), figsize=(4.4 * len(bg), 3.6))
    for ax, p in zip(np.atleast_1d(axes), bg, strict=True):
        rgb = load_rgb(p)
        res = pred.predict(rgb, None)
        img = render_prediction(rgb, res.candidates, None, None, num_candidates=3)
        x0, y0, x1, y1 = crop_box(pred)
        ax.imshow(img[y0:y1, x0:x1])
        best = f"top score {res.best.score:.2f}" if res.best else "no candidate"
        ax.set_title(f"{p.name}: {best}", fontsize=9)
        ax.axis("off")
    fig.suptitle("External-style check: Cornell background images (no object, never used in training)")
    fig.tight_layout()
    fig.savefig(out, dpi=85)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--decode-id", default=None, help="scene id for the decoding figure (default: first)")
    args = ap.parse_args()
    run = Path(args.run)
    run = run if run.is_absolute() else REPO_ROOT / run
    pred = GraspPredictor(run / "checkpoints" / "best.pt")
    name = pred.cfg["experiment"]["name"]
    samples = split_samples(pred.cfg)[args.split]
    s = next((x for x in samples if x.sample_id == args.decode_id), samples[0])
    FIG.mkdir(parents=True, exist_ok=True)
    if pred.modality != "rgb":
        decode_figure(pred, s, FIG / f"decode_pipeline_{name}.png")
    csv_path = REPO_ROOT / "experiments" / "results" / f"{name}__{args.split}_per_image.csv"
    if csv_path.exists():
        qualitative_grid(pred, samples, csv_path, FIG / f"qualitative_{name}_{args.split}.png")
    else:
        print("no per-image CSV yet:", csv_path)
    background_figure(pred, FIG / f"background_{name}.png")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
