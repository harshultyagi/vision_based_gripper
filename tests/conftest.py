"""Shared pytest fixtures. Synthetic fixtures are clearly synthetic and never used for
performance claims; tests marked ``dataset`` use the real local Cornell copy if configured."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_planner.config import ConfigError, resolve_dataset_root  # noqa: E402
from grasp_planner.data.cornell import CornellSample  # noqa: E402
from grasp_planner.data.grasp import Grasp  # noqa: E402


@pytest.fixture(scope="session")
def cornell_root() -> Path:
    try:
        return resolve_dataset_root()
    except ConfigError as exc:
        pytest.skip(f"Cornell dataset not configured: {exc}")


def write_synthetic_scene(folder: Path, sid: str, grasps: list[Grasp], seed: int = 0) -> CornellSample:
    """Write a fake 640x480 scene (a dark bar on a light table) with a matching depth tiff
    and cpos/cneg files in the Cornell format."""
    import tifffile

    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    rgb = np.full((480, 640, 3), 220, np.uint8) + rng.integers(0, 10, (480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 0.70, np.float32)
    for g in grasps:
        poly = np.round(Grasp(g.x, g.y, g.theta + np.pi / 2, g.jaw_size * 3, g.width * 0.6).polygon()).astype(
            np.int32
        )
        cv2.fillPoly(rgb, [poly], (30, 30, 40))
        mask = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        depth[mask == 1] = 0.66
    depth[:5, :] = np.nan  # some invalid pixels, like real sensor data
    cv2.imwrite(str(folder / f"pcd{sid}r.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    tifffile.imwrite(str(folder / f"pcd{sid}d.tiff"), depth)
    lines = []
    for g in grasps:
        lines += [f"{x:.3f} {y:.3f}" for x, y in g.polygon()]
    (folder / f"pcd{sid}cpos.txt").write_text("\n".join(lines) + "\n")
    (folder / f"pcd{sid}cneg.txt").write_text("")
    pos = np.stack([g.polygon() for g in grasps])
    return CornellSample(
        sid,
        folder.name,
        folder / f"pcd{sid}r.png",
        folder / f"pcd{sid}d.tiff",
        folder / f"pcd{sid}cpos.txt",
        folder / f"pcd{sid}cneg.txt",
        pos,
        np.zeros((0, 4, 2)),
    )


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    root = tmp_path / "fake_cornell"
    for k in range(12):
        sid = f"{100 + k:04d}"
        g = Grasp(
            float(rng.uniform(280, 360)),
            float(rng.uniform(210, 290)),
            float(rng.uniform(-1.5, 1.5)),
            float(rng.uniform(30, 60)),
            20.0,
        )
        write_synthetic_scene(root / "01", sid, [g], seed=k)
    return root


def minimal_config(split_file: str, model: str = "grconvnet", modality: str = "rgbd") -> dict:
    return {
        "experiment": {"name": "pytest", "seed": 0},
        "data": {
            "split_file": split_file,
            "modality": modality,
            "output_size": 64,
            "crop": {"cx": 320.0, "cy": 250.0, "size": 320.0},
            "max_width": 150.0,
            "num_workers": 0,
            "augment": {
                "enabled": True,
                "rotation_deg": 30.0,
                "zoom_min": 0.9,
                "translate_px": 5.0,
                "hflip_prob": 0.5,
            },
        },
        "model": {"name": model, "channels": 8, "num_residual": 1, "dropout": 0.0},
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "val_batch_size": 2,
            "optimizer": "adam",
            "lr": 1e-3,
            "scheduler": "none",
            "amp": False,
            "grad_clip": 1.0,
            "early_stopping_patience": 0,
        },
        "decode": {
            "q_sigma": 1.0,
            "q_threshold": 0.0,
            "min_distance": 3,
            "max_peaks": 10,
            "min_width": 0.0,
            "nms_iou": 0.3,
            "nms_angle_deg": 30.0,
            "max_candidates": 5,
        },
        "eval": {"k": 5},
    }
