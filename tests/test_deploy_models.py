"""The committed slim checkpoints (models/*.pt) must exist, load on CPU, match the app config, and
produce candidates through GraspPredictor (no dataset needed)."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from grasp_planner.inference.predictor import GraspPredictor

REPO = Path(__file__).resolve().parents[1]


def test_app_config_points_to_committed_models():
    cfg = yaml.safe_load((REPO / "app" / "app_config.yaml").read_text())
    for key in ("rgbd_checkpoint", "rgb_checkpoint"):
        assert cfg[key].startswith("models/") and (REPO / cfg[key]).exists(), key


@pytest.mark.parametrize("name,modality", [("grconvnet_rgbd_image", "rgbd"), ("grconvnet_rgb_image", "rgb")])
def test_slim_checkpoint_loads_and_predicts_on_cpu(name, modality):
    path = REPO / "models" / f"{name}.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert "optimizer" not in ck and ck["config"]["experiment"]["name"] == name
    p = GraspPredictor(path, torch.device("cpu"))
    assert p.modality == modality
    rgb = np.full((480, 640, 3), 220, np.uint8)
    rgb[230:300, 260:380] = 30  # a dark bar on a light table
    depth = np.full((480, 640), 0.7, np.float32) if modality == "rgbd" else None
    out = p.predict(rgb, depth, max_candidates=3)
    for g in out.candidates:
        assert np.isfinite([g.x, g.y, g.theta, g.width, g.score]).all()
