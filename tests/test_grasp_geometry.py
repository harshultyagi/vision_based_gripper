"""Grasp representation, angle conventions, contact points, IoU and the Cornell criterion."""

import numpy as np
import pytest

from grasp_planner.data.grasp import (
    Grasp,
    angle_difference,
    grasp_matches,
    polygon_iou,
    rect_to_grasp,
    rectangle_geometry_errors,
    wrap_angle,
)


def test_rect_to_grasp_axis_aligned():
    # p0 -> p1 runs along +x: closing direction horizontal, theta = 0, opening 40, jaw 10
    pts = np.array([[100, 95], [140, 95], [140, 105], [100, 105]], float)
    g = rect_to_grasp(pts)
    assert (g.x, g.y) == pytest.approx((120, 100))
    assert g.theta == pytest.approx(0.0)
    assert g.width == pytest.approx(40)
    assert g.jaw_size == pytest.approx(10)


def test_rect_to_grasp_angle_is_counter_clockwise_on_screen():
    # p1 is up-right of p0 on screen (smaller y) -> positive angle (+45 deg)
    pts = np.array([[0, 10], [10, 0], [12, 2], [2, 12]], float)
    assert np.rad2deg(rect_to_grasp(pts).theta) == pytest.approx(45.0)
    # p1 down-right of p0 -> -45 deg
    pts2 = np.array([[0, 0], [10, 10], [8, 12], [-2, 2]], float)
    assert np.rad2deg(rect_to_grasp(pts2).theta) == pytest.approx(-45.0)


def test_vertical_grasp_wraps_to_minus_90():
    pts = np.array([[0, 0], [0, 20], [5, 20], [5, 0]], float)  # p0 -> p1 points down
    assert np.rad2deg(rect_to_grasp(pts).theta) == pytest.approx(-90.0)


@pytest.mark.parametrize("deg", [-89.0, -45.0, 0.0, 30.0, 89.0])
def test_polygon_roundtrip(deg):
    g = Grasp(50.0, 60.0, np.deg2rad(deg), 30.0, 12.0)
    g2 = rect_to_grasp(g.polygon())
    assert (g2.x, g2.y, g2.width, g2.jaw_size) == pytest.approx((50, 60, 30, 12))
    assert angle_difference(g2.theta, g.theta) == pytest.approx(0.0, abs=1e-9)


def test_wrap_angle_range():
    for a in np.linspace(-10, 10, 101):
        w = wrap_angle(a)
        assert -np.pi / 2 <= w < np.pi / 2
        assert angle_difference(w, a) == pytest.approx(0.0, abs=1e-9)


def test_angle_difference_symmetry_cases():
    d = np.deg2rad
    assert np.rad2deg(angle_difference(d(80), d(-80))) == pytest.approx(20)
    assert np.rad2deg(angle_difference(d(10), d(190))) == pytest.approx(0, abs=1e-9)
    assert np.rad2deg(angle_difference(d(0), d(90))) == pytest.approx(90)
    assert np.rad2deg(angle_difference(d(-30), d(30))) == pytest.approx(60)


def test_contact_points():
    g = Grasp(100.0, 100.0, np.deg2rad(90), 40.0, 10.0)  # vertical closing direction
    cps = g.contact_points()
    assert cps[0] == pytest.approx([100, 120])  # centre - 20 * (0, -1)
    assert cps[1] == pytest.approx([100, 80])
    assert np.linalg.norm(cps[1] - cps[0]) == pytest.approx(40)


def test_iou_hand_checked():
    a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert polygon_iou(a, a) == pytest.approx(1.0)
    b = a + [5, 0]  # half overlap: inter 50, union 150
    assert polygon_iou(a, b) == pytest.approx(1 / 3)
    c = a + [20, 0]
    assert polygon_iou(a, c) == 0.0
    # 10x10 vs the same square rotated 45 deg about its centre:
    # intersection = regular octagon area = 2 * (1 + sqrt2) * s^2 with s = 10 / (1 + sqrt2)
    g = Grasp(5, 5, np.deg2rad(45), 10, 10)
    s = 10 / (1 + np.sqrt(2))
    inter = 2 * (1 + np.sqrt(2)) * s**2
    assert polygon_iou(a, g.polygon()) == pytest.approx(inter / (200 - inter))


def test_cornell_criterion():
    gt = Grasp(50, 50, 0.0, 40, 20)
    assert grasp_matches(Grasp(50, 50, np.deg2rad(29), 40, 20), gt)
    assert not grasp_matches(Grasp(50, 50, np.deg2rad(31), 40, 20), gt)
    assert grasp_matches(Grasp(50, 50, np.deg2rad(179), 40, 20), gt)  # 180-deg symmetry
    assert not grasp_matches(Grasp(90, 50, 0.0, 40, 20), gt)  # IoU = 0
    # IoU exactly 1/3 > 0.25 when shifted by half the opening
    assert grasp_matches(Grasp(70, 50, 0.0, 40, 20), gt)


def test_rectangle_validation():
    good = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], float)
    assert rectangle_geometry_errors(good) == []
    assert rectangle_geometry_errors(np.array([[0, 0], [np.nan, 0], [10, 5], [0, 5]]))
    skew = np.array([[0, 0], [10, 0], [15, 5], [0, 5]], float)
    assert rectangle_geometry_errors(skew)
    degenerate = np.array([[0, 0], [0, 0], [10, 5], [0, 5]], float)
    assert rectangle_geometry_errors(degenerate)
