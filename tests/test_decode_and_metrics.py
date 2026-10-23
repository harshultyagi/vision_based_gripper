"""Candidate decoding, NMS, ranking, coordinate mapping and evaluation metrics."""

import numpy as np
import pytest

from grasp_planner.data.dataset import draw_grasp_targets
from grasp_planner.data.grasp import Grasp, angle_difference
from grasp_planner.data.transforms import CropSpec, build_affine
from grasp_planner.evaluation.metrics import evaluate_image, summarise
from grasp_planner.inference.decode import DecodeSpec, decode_grasps, grasp_to_original, nms

SPEC = DecodeSpec(q_sigma=1.0, q_threshold=0.2, min_distance=3, max_width=150.0)


def test_decode_recovers_drawn_grasps_and_ranks_them():
    a = Grasp(30.0, 30.0, np.deg2rad(40), 30.0, 12.0)
    b = Grasp(80.0, 70.0, np.deg2rad(-60), 45.0, 12.0)
    t = draw_grasp_targets([a], 112, 150.0)
    tb = draw_grasp_targets([b], 112, 150.0)
    t[0] = np.maximum(t[0], 0.6 * tb[0])  # b has lower quality than a
    for i in (1, 2, 3):
        t[i] = np.where(tb[0] > 0, tb[i], t[i])
    cands = decode_grasps(*t, SPEC)
    assert len(cands) >= 2
    best, second = cands[0], cands[1]
    assert best.score > second.score
    assert np.hypot(best.x - a.x, best.y - a.y) < 2.5
    assert np.rad2deg(angle_difference(best.theta, a.theta)) < 1
    assert best.width == pytest.approx(30, abs=2)
    assert np.hypot(second.x - b.x, second.y - b.y) < 2.5


def test_decode_empty_when_below_threshold():
    z = np.zeros((64, 64))
    assert decode_grasps(z + 0.05, z + 1, z, z + 0.2, SPEC) == []


def test_nms_removes_duplicates_but_keeps_other_orientations():
    g1 = Grasp(50, 50, 0.0, 40, 20, score=0.9)
    dup = Grasp(52, 50, np.deg2rad(5), 40, 20, score=0.8)
    perp = Grasp(50, 50, np.deg2rad(90), 40, 20, score=0.7)
    kept = nms([g1, dup, perp], SPEC)
    assert kept == [g1, perp]


def test_grasp_to_original_inverts_preprocessing():
    crop = CropSpec(320.0, 250.0, 320.0)
    M = build_affine(crop, 224, rotation_rad=0.4, zoom=0.9, hflip=True)
    from grasp_planner.data.grasp import rect_to_grasp
    from grasp_planner.data.transforms import apply_affine_points

    orig = Grasp(300.0, 260.0, np.deg2rad(15), 50.0, 25.0, score=0.7)
    in_net = rect_to_grasp(apply_affine_points(M, orig.polygon()))
    back = grasp_to_original(Grasp(in_net.x, in_net.y, in_net.theta, in_net.width, in_net.jaw_size, 0.7), M)
    assert (back.x, back.y, back.width) == pytest.approx((300, 260, 50))
    assert angle_difference(back.theta, orig.theta) == pytest.approx(0, abs=1e-9)
    assert back.score == 0.7


def test_metrics_per_image():
    gts = [Grasp(100, 100, 0.0, 40, 20), Grasp(200, 100, np.deg2rad(90), 30, 15)]
    good = Grasp(102, 101, np.deg2rad(10), 42, 21, score=0.9)
    bad = Grasp(150, 150, np.deg2rad(45), 40, 20, score=0.95)
    r = evaluate_image("x", [good], gts)
    assert r.top1_success and r.topk_success
    assert r.center_error_px == pytest.approx(np.hypot(2, 1))
    assert r.angle_error_deg == pytest.approx(10)
    assert r.width_error_px == pytest.approx(2)
    r2 = evaluate_image("y", [bad, good], gts, k=2)
    assert not r2.top1_success and r2.topk_success
    r3 = evaluate_image("z", [], gts)
    assert not r3.top1_success and r3.num_candidates == 0
    s = summarise([r, r2, r3], k=2)
    assert s["top1_accuracy"] == pytest.approx(1 / 3)
    assert s["top2_accuracy"] == pytest.approx(2 / 3)
    assert s["num_images_with_prediction"] == 2
