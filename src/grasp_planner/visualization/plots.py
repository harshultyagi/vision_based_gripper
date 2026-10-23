"""Drawing helpers for grasps, candidates, ground truth and training curves."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..data.grasp import Grasp  # noqa: E402

# RGB colours
GT_POS = (40, 200, 40)
GT_NEG = (230, 60, 60)
CAND = (255, 190, 0)
BEST = (0, 120, 255)
CONTACT = (255, 0, 255)


def draw_grasp(
    img: np.ndarray, g: Grasp, color, thickness: int = 2, jaw_thickness: int | None = None
) -> None:
    """Draw a grasp rectangle in place. The two jaw edges (p1-p2 and p3-p0) are drawn thicker
    so the gripper plates are distinguishable from the opening edges."""
    p = np.round(g.polygon()).astype(np.int32)
    jt = jaw_thickness or thickness + 1
    cv2.line(img, tuple(p[0]), tuple(p[1]), color, thickness, cv2.LINE_AA)
    cv2.line(img, tuple(p[2]), tuple(p[3]), color, thickness, cv2.LINE_AA)
    cv2.line(img, tuple(p[1]), tuple(p[2]), color, jt, cv2.LINE_AA)
    cv2.line(img, tuple(p[3]), tuple(p[0]), color, jt, cv2.LINE_AA)


def draw_rect_points(img: np.ndarray, pts: np.ndarray, color, thickness: int = 1) -> None:
    cv2.polylines(img, [np.round(pts).astype(np.int32)], True, color, thickness, cv2.LINE_AA)


def render_prediction(
    rgb: np.ndarray,
    candidates: list[Grasp],
    gt_pos: np.ndarray | None = None,
    gt_neg: np.ndarray | None = None,
    num_candidates: int = 5,
    show_scores: bool = True,
) -> np.ndarray:
    """Return an annotated copy of ``rgb``: GT positives (green, thin), GT negatives (red, thin,
    optional), candidates 2..k (amber, with rank:score), selected grasp (blue, thick) and its two
    contact points (magenta dots)."""
    img = rgb.copy()
    if gt_neg is not None:
        for r in gt_neg:
            draw_rect_points(img, r, GT_NEG, 1)
    if gt_pos is not None:
        for r in gt_pos:
            draw_rect_points(img, r, GT_POS, 1)
    shown = candidates[:num_candidates]
    for rank, g in reversed(list(enumerate(shown))):
        if rank == 0:
            continue
        draw_grasp(img, g, CAND, 1, 2)
        if show_scores:
            cv2.putText(
                img,
                f"{rank + 1}:{g.score:.2f}",
                (int(g.x) + 4, int(g.y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                CAND,
                1,
                cv2.LINE_AA,
            )
    if shown:
        b = shown[0]
        draw_grasp(img, b, BEST, 2, 4)
        for cp in b.contact_points():
            cv2.circle(img, tuple(np.round(cp).astype(int)), 4, CONTACT, -1, cv2.LINE_AA)
        if show_scores:
            cv2.putText(
                img,
                f"1:{b.score:.2f}",
                (int(b.x) + 4, int(b.y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                BEST,
                1,
                cv2.LINE_AA,
            )
    return img


def save_maps_figure(
    rgb_input: np.ndarray | None, maps: dict[str, np.ndarray], path: Path, title: str = ""
) -> None:
    """Figure of the network outputs: input crop, quality, angle (deg), width."""
    q = maps["quality"]
    ang = np.rad2deg(0.5 * np.arctan2(maps["sin2"], maps["cos2"]))
    panels = [
        ("quality", q, "viridis", (0, 1)),
        ("angle [deg]", ang, "hsv", (-90, 90)),
        ("width (normalised)", maps["width"], "magma", None),
    ]
    n = len(panels) + (rgb_input is not None)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.3))
    i = 0
    if rgb_input is not None:
        axes[0].imshow(rgb_input)
        axes[0].set_title("network input (RGB part)")
        i = 1
    for ax, (name, m, cmap, lim) in zip(axes[i:], panels, strict=True):
        im = ax.imshow(m, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_history(histories: dict[str, list[dict]], path: Path) -> None:
    """Training/validation loss and validation accuracy curves for one or more runs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, rows in histories.items():
        ep = [int(r["epoch"]) for r in rows]
        (l1,) = axes[0].plot(ep, [float(r["train_loss"]) for r in rows], label=f"{name} train")
        axes[0].plot(
            ep, [float(r["val_loss"]) for r in rows], "--", color=l1.get_color(), label=f"{name} val"
        )
        axes[1].plot(ep, [float(r["val_top1"]) for r in rows], color=l1.get_color(), label=name)
    axes[0].set(xlabel="epoch", ylabel="loss (sum of 4 Smooth-L1 terms)", title="Loss", yscale="log")
    axes[1].set(
        xlabel="epoch", ylabel="val top-1 rectangle accuracy", title="Validation accuracy", ylim=(0, 1)
    )
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
