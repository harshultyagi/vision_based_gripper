"""Target-map construction, model I/O shapes, loss values and finite gradients."""

import numpy as np
import pytest
import torch

from grasp_planner.data.dataset import build_input, draw_grasp_targets
from grasp_planner.data.grasp import Grasp
from grasp_planner.models import GGCNN, GRConvNet, build_model, count_parameters
from grasp_planner.training.losses import grasp_loss


def test_draw_targets_values():
    g = Grasp(32.0, 32.0, np.deg2rad(30), 30.0, 12.0)
    t = draw_grasp_targets([g], 64, 150.0)
    assert t.shape == (4, 64, 64)
    m = t[0] > 0
    assert m[32, 32] and not m[0, 0]
    assert t[1, 32, 32] == pytest.approx(np.cos(np.deg2rad(60)), abs=1e-6)
    assert t[2, 32, 32] == pytest.approx(np.sin(np.deg2rad(60)), abs=1e-6)
    assert t[3, 32, 32] == pytest.approx(30 / 150)
    # centre-third region: ~ (30/3) x 12 = 120 px
    assert 80 < m.sum() < 180
    assert (t[1:][:, ~m] == 0).all()


def test_build_input_channel_order():
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 0] = 255
    depth = np.linspace(0.5, 0.9, 64, dtype=np.float32).reshape(8, 8)
    x = build_input(rgb, depth, "rgbd")
    assert x.shape == (4, 8, 8)
    assert x[0].mean() == pytest.approx(0, abs=1e-6)  # depth first, mean removed
    assert build_input(rgb, None, "rgb").shape == (3, 8, 8)
    with pytest.raises(ValueError):
        build_input(rgb, None, "rgbd")


@pytest.mark.parametrize("cls,cin", [(GGCNN, 1), (GGCNN, 4), (GRConvNet, 4), (GRConvNet, 3)])
@pytest.mark.parametrize("size", [224, 100])
def test_output_shapes(cls, cin, size):
    model = cls(cin)
    out = model(torch.randn(2, cin, size, size))
    assert set(out) == {"quality", "cos2", "sin2", "width"}
    for v in out.values():
        assert v.shape == (2, 1, size, size)


def test_parameter_counts_reasonable():
    assert 50_000 < count_parameters(GGCNN(4)) < 200_000
    assert 1_500_000 < count_parameters(GRConvNet(4)) < 2_500_000  # paper reports ~1.9M


def test_loss_zero_for_perfect_prediction_and_finite_gradients():
    model = build_model({"name": "grconvnet", "channels": 8, "num_residual": 1, "dropout": 0.0}, 4)
    x = torch.randn(2, 4, 64, 64)
    y = torch.zeros(2, 4, 64, 64)
    y[:, 0, 20:30, 20:30] = 1
    perfect = {n: y[:, i : i + 1] for i, n in enumerate(["quality", "cos2", "sin2", "width"])}
    assert float(grasp_loss(perfect, y)[0]) == 0.0
    loss, parts = grasp_loss(model(x), y)
    assert torch.isfinite(loss) and set(parts) == {"quality", "cos2", "sin2", "width"}
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_smooth_l1_hand_value():
    y = torch.zeros(1, 4, 2, 2)
    pred = {n: torch.zeros(1, 1, 2, 2) for n in ["quality", "cos2", "sin2", "width"]}
    pred["quality"][..., 0, 0] = 0.5  # 0.5 * 0.5^2 = 0.125 on 1 of 4 pixels
    pred["width"][..., 0, 0] = 3.0  # 3 - 0.5 = 2.5 on 1 of 4 pixels
    loss, parts = grasp_loss(pred, y)
    assert parts["quality"] == pytest.approx(0.125 / 4)
    assert parts["width"] == pytest.approx(2.5 / 4)
    assert float(loss) == pytest.approx((0.125 + 2.5) / 4)
