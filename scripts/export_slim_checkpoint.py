"""Write a small inference-only copy of a training checkpoint (weights + config + provenance,
no optimiser/scheduler state) and verify it gives identical outputs.

Usage:
    python scripts/export_slim_checkpoint.py --run outputs/runs/grconvnet_rgbd_image
Writes models/<run>.pt (committed to Git with the owner's approval; used by the app and the
Hugging Face Space).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_planner.config import REPO_ROOT  # noqa: E402
from grasp_planner.data.dataset import PreprocessSpec  # noqa: E402
from grasp_planner.pipeline import load_checkpoint_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    run = run if run.is_absolute() else REPO_ROOT / run
    src = run / "checkpoints" / "best.pt"
    full = torch.load(src, map_location="cpu", weights_only=False)
    slim = {
        "model": full["model"],
        "config": full["config"],
        "epoch": full["epoch"],
        "source_checkpoint": str(src.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256_16": hashlib.sha256(src.read_bytes()).hexdigest()[:16],
    }
    out = REPO_ROOT / "models" / f"{run.name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out)

    # verify: identical outputs from the full and the slim checkpoint
    cpu = torch.device("cpu")
    m_full, _ = load_checkpoint_model(src, cpu)
    m_slim, _ = load_checkpoint_model(out, cpu)
    c = PreprocessSpec.from_config(full["config"]).in_channels
    x = torch.randn(2, c, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a, b = m_full(x), m_slim(x)
    assert all(torch.equal(a[k], b[k]) for k in a), "slim checkpoint outputs differ"
    print(
        f"wrote {out.relative_to(REPO_ROOT)} ({out.stat().st_size / 2**20:.1f} MB), epoch {slim['epoch']}, "
        f"source sha {slim['source_sha256_16']}, outputs identical"
    )


if __name__ == "__main__":
    main()
