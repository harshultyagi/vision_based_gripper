"""Assemble a ready-to-upload Hugging Face Space (Gradio SDK) in outputs/hf_space/.

Usage:
    python scripts/build_space.py [--examples 0101 0400 0618 0811]

The folder contains the app, the grasp_planner package, the slim checkpoints from models/,
a CPU-only requirements.txt and a README with the Space metadata. Example scenes (RGB + depth)
are copied from the local Cornell dataset into the Space only; they are not committed to
GitHub. Upload instructions: docs/deployment.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_planner.config import REPO_ROOT, resolve_dataset_root  # noqa: E402

OUT = REPO_ROOT / "outputs" / "hf_space"
SPACE_ID = "btechbillu/Vission-Based-grasper"
GITHUB = "https://github.com/harshultyagi/vision_based_gripper"

README = """---
title: Vision-Based Grasp Planner
emoji: 🦾
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Planar parallel-jaw grasp detection trained on Cornell
---

# Vision-Based Grasp Planner

Upload a top-down image of a single object (optionally with its depth TIFF). The app predicts
ranked planar grasps for a parallel-jaw gripper: centre, angle, opening width, score and the two
contact points.

* Model: GR-ConvNet trained on the Cornell Grasping Dataset. The RGB-D model is used when a
  depth file is given (88.6 % top-1 on the held-out image-wise test split); otherwise the
  RGB-only model (90.2 %). Rectangle metric: IoU > 25 % and angle < 30°.
* Runs on CPU here, so a prediction takes a moment.
* Limitations: single-object Cornell-style scenes, 2-D image-plane grasps only, no robot
  validation. Scores are not probabilities.

Code, training details and the full report: {github}
"""

APP_SHIM = '''"""Hugging Face Space entry point: runs app/app.py listening on all interfaces."""

import runpy
import sys

sys.argv = ["app/app.py", "--host", "0.0.0.0", "--port", "7860"]
runpy.run_path("app/app.py", run_name="__main__")
'''

REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.11.0+cpu
numpy>=1.26
scipy>=1.11
opencv-python-headless>=4.9
pillow>=10.0
tifffile>=2024.1.0
pyyaml>=6.0
shapely>=2.0
matplotlib>=3.8
gradio=={gradio_version}
"""


def main() -> None:
    import gradio

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--examples",
        nargs="*",
        default=["0101", "0400", "0618", "0811"],
        help="test-split scene ids to include as clickable examples",
    )
    args = ap.parse_args()

    split = json.loads((REPO_ROOT / "experiments/splits/image_wise_seed42.json").read_text())
    not_test = [i for i in args.examples if i not in split["splits"]["test"]]
    if not_test:
        raise SystemExit(f"examples must come from the image-wise test split; not test: {not_test}")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "app" / "examples").mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    shutil.copytree(REPO_ROOT / "src", OUT / "src", ignore=ignore)
    shutil.copytree(REPO_ROOT / "models", OUT / "models")
    for f in ("app/app.py", "app/app_config.yaml", "LICENSE"):
        shutil.copy(REPO_ROOT / f, OUT / f)
    (OUT / "app.py").write_text(APP_SHIM, encoding="utf-8")
    (OUT / "README.md").write_text(
        README.format(gradio_version=gradio.__version__, github=GITHUB), encoding="utf-8"
    )
    (OUT / "requirements.txt").write_text(
        REQUIREMENTS.format(gradio_version=gradio.__version__), encoding="utf-8"
    )
    (OUT / ".gitattributes").write_text(
        "*.pt filter=lfs diff=lfs merge=lfs -text\n*.tiff filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )

    root = resolve_dataset_root()
    for sid in args.examples:
        found = list(root.glob(f"*/pcd{sid}r.png"))
        if not found:
            raise SystemExit(f"example scene {sid} not found in the dataset")
        png = found[0]
        shutil.copy(png, OUT / "app" / "examples" / png.name)
        tiff = png.with_name(f"pcd{sid}d.tiff")
        if tiff.exists():
            shutil.copy(tiff, OUT / "app" / "examples" / tiff.name)

    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 2**20
    print(f"Space folder ready: {OUT} ({size:.1f} MB) for https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
