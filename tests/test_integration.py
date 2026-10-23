"""Integration tests: a few real training steps, checkpoint save/reload/resume, and end-to-end
inference through GraspPredictor -- all on SYNTHETIC scenes (no performance claims)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import minimal_config

from grasp_planner.data.cornell import discover_samples, load_depth, load_rgb
from grasp_planner.data.splits import make_split, save_split
from grasp_planner.inference.predictor import GraspPredictor
from grasp_planner.pipeline import decode_spec, make_dataset, make_model, split_samples
from grasp_planner.training.trainer import Trainer

pytestmark = pytest.mark.integration


def _setup(synthetic_root: Path, tmp_path: Path, model: str = "grconvnet"):
    samples, _ = discover_samples(synthetic_root)
    split = make_split([s.sample_id for s in samples], "image", (0.5, 0.25, 0.25), 0)
    split_file = tmp_path / "split.json"
    save_split(split, split_file)
    cfg = minimal_config(str(split_file), model=model)
    parts = split_samples(cfg, synthetic_root)
    return cfg, parts


@pytest.mark.parametrize("model", ["grconvnet", "ggcnn"])
def test_train_checkpoint_resume_and_predict(synthetic_root, tmp_path, model):
    torch.manual_seed(0)
    cfg, parts = _setup(synthetic_root, tmp_path, model)
    assert not set(s.sample_id for s in parts["train"]) & set(s.sample_id for s in parts["test"])
    train_set = make_dataset(cfg, parts["train"], train=True, cache_dir=tmp_path / "cache")
    val_set = make_dataset(cfg, parts["val"], train=False, cache_dir=tmp_path / "cache")
    item = train_set[0]
    assert item["x"].shape == (4, 64, 64) and item["y"].shape == (4, 64, 64)
    assert item["y"][0].sum() > 0  # augmented grasp still inside the crop

    run = tmp_path / "run"
    tr = Trainer(cfg, make_model(cfg), train_set, val_set, decode_spec(cfg), run, torch.device("cpu"))
    before = [p.detach().clone() for p in tr.model.parameters()]
    res = tr.fit()
    assert res["completed"] and (run / "history.csv").exists()
    assert any(not torch.equal(a, b) for a, b in zip(before, tr.model.parameters(), strict=True))
    best = run / "checkpoints" / "best.pt"
    assert best.exists() and (run / "checkpoints" / "last.pt").exists()

    # reload: identical outputs
    pred = GraspPredictor(best, torch.device("cpu"))
    x = val_set[0]["x"][None]
    ref_model = tr.model.eval()
    ck = torch.load(best, weights_only=False)
    ref_model.load_state_dict(ck["model"])
    with torch.no_grad():
        a, b = ref_model(x), pred.model(x)
    for k in a:
        assert torch.allclose(a[k], b[k])

    # resume continues from the next epoch
    cfg2 = {**cfg, "train": {**cfg["train"], "epochs": 2}}
    tr2 = Trainer(cfg2, make_model(cfg2), train_set, val_set, decode_spec(cfg2), run, torch.device("cpu"))
    tr2.resume(run / "checkpoints" / "last.pt")
    assert tr2.start_epoch == 1
    tr2.fit()

    # end-to-end inference on a synthetic test scene
    s = parts["test"][0]
    out = pred.predict(load_rgb(s.rgb_path), load_depth(s.depth_path), max_candidates=3)
    assert len(out.candidates) <= 3
    for g in out.candidates:
        assert np.isfinite([g.x, g.y, g.theta, g.width, g.score]).all()
        assert g.contact_points().shape == (2, 2)
    assert out.maps["quality"].shape == (64, 64)


def test_predictor_errors_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        GraspPredictor(tmp_path / "missing.pt", torch.device("cpu"))
