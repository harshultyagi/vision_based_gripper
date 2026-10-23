"""The Gradio app must surface a clear error (never a placeholder) when no model can be loaded,
and must return a real prediction when a checkpoint is available (synthetic checkpoint here)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
gr = pytest.importorskip("gradio")

from conftest import minimal_config  # noqa: E402

from app import App  # noqa: E402
from grasp_planner.pipeline import make_model  # noqa: E402


def test_missing_checkpoint_reports_error_and_blocks_prediction(tmp_path):
    app = App(tmp_path / "nope.pt", None)
    assert not app.models
    assert "No model loaded" in app.status() and "nope.pt" in app.status()
    from PIL import Image

    with pytest.raises(gr.Error):
        app.run(Image.new("RGB", (640, 480)), None, 5)


def test_rgb_model_returns_real_candidates(tmp_path):
    cfg = minimal_config("unused.json", modality="rgb")
    model = make_model(cfg)
    ckpt = tmp_path / "rgb.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": 0}, ckpt)
    app = App(None, ckpt)
    from PIL import Image

    rgb = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    annotated, summary, rows, files = app.run(Image.fromarray(rgb), None, 3)
    assert annotated.shape == (480, 640, 3)
    assert len(rows) <= 3 and all(len(r) == 8 for r in rows)
    assert all(Path(f).exists() for f in files)
    # an untrained model may legitimately produce no candidate; either way the text says so
    assert ("Selected grasp" in summary) == bool(rows)
    assert rows or "no candidate" in summary
