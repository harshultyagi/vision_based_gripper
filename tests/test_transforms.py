"""Augmentation correctness: grasp parameters must transform exactly like the pixels."""

import cv2
import numpy as np
import pytest

from grasp_planner.data.grasp import Grasp, angle_difference, rect_to_grasp
from grasp_planner.data.transforms import (
    AugmentSpec,
    CropSpec,
    apply_affine_points,
    build_affine,
    inpaint_depth,
    invert_affine,
    sample_affine,
    warp_image,
)

CROP = CropSpec(320.0, 240.0, 320.0)
S = 160  # scale factor 0.5


def transformed(g: Grasp, M: np.ndarray) -> Grasp:
    return rect_to_grasp(apply_affine_points(M, g.polygon()))


def test_identity_crop_scales_width_and_keeps_angle():
    g = Grasp(320.0, 240.0, np.deg2rad(30), 40.0, 10.0)
    t = transformed(g, build_affine(CROP, S))
    assert (t.x, t.y) == pytest.approx(((S - 1) / 2, (S - 1) / 2))
    assert t.width == pytest.approx(20.0)
    assert t.jaw_size == pytest.approx(5.0)
    assert angle_difference(t.theta, g.theta) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("rot", [10.0, 90.0, -135.0])
def test_rotation_adds_angle(rot):
    g = Grasp(330.0, 250.0, np.deg2rad(20), 40.0, 10.0)
    t = transformed(g, build_affine(CROP, S, rotation_rad=np.deg2rad(rot)))
    assert np.rad2deg(angle_difference(t.theta, g.theta + np.deg2rad(rot))) == pytest.approx(0, abs=1e-6)
    assert t.width == pytest.approx(20.0)


def test_hflip_negates_angle():
    g = Grasp(320.0, 240.0, np.deg2rad(25), 40.0, 10.0)
    t = transformed(g, build_affine(CROP, S, hflip=True))
    assert np.rad2deg(angle_difference(t.theta, -g.theta)) == pytest.approx(0, abs=1e-6)


def test_zoom_scales_width():
    g = Grasp(320.0, 240.0, 0.0, 40.0, 10.0)
    t = transformed(g, build_affine(CROP, S, zoom=0.8))
    assert t.width == pytest.approx(40 * S / (320 * 0.8))


def test_inverse_roundtrip():
    rng = np.random.default_rng(0)
    M = sample_affine(CROP, S, AugmentSpec(180, 0.8, 20, 0.5), rng)
    pts = rng.uniform(0, 640, (10, 2))
    back = apply_affine_points(invert_affine(M), apply_affine_points(M, pts))
    assert back == pytest.approx(pts)


@pytest.mark.parametrize("seed", range(5))
def test_pixels_and_vertices_stay_aligned(seed):
    """Paint a grasp rectangle into an image, warp both, and check the warped mask's centroid and
    principal orientation agree with the transformed grasp."""
    rng = np.random.default_rng(seed)
    g = Grasp(330.0, 245.0, rng.uniform(-1.5, 1.5), 120.0, 30.0)
    img = np.zeros((480, 640), np.float32)
    cv2.fillPoly(img, [np.round(g.polygon()).astype(np.int32)], 1.0)
    M = sample_affine(CROP, S, AugmentSpec(180, 0.8, 10, 0.5), rng)
    w = warp_image(img, M, S)
    t = transformed(g, M)
    ys, xs = np.nonzero(w > 0.5)
    assert xs.mean() == pytest.approx(t.x, abs=1.0)
    assert ys.mean() == pytest.approx(t.y, abs=1.0)
    # principal axis of the warped blob = closing direction (long side of the 120x30 rectangle)
    cov = np.cov(np.stack([xs - xs.mean(), ys - ys.mean()]))
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, np.argmax(evals)]
    blob_theta = np.arctan2(-vy, vx)
    assert np.rad2deg(angle_difference(blob_theta, t.theta)) < 2.0


def test_inpaint_fills_nans():
    d = np.full((50, 60), 0.7, np.float32)
    d[10:20, 10:20] = np.nan
    d[0, :] = np.nan
    out = inpaint_depth(d)
    assert np.isfinite(out).all()
    assert out[15, 15] == pytest.approx(0.7, abs=1e-3)
