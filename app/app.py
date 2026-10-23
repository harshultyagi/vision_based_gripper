"""Gradio web app: upload an image (and optionally a depth TIFF), get ranked grasp candidates.

Launch:
    python app/app.py [--checkpoint PATH] [--rgb-checkpoint PATH] [--port 7860]

Default checkpoints are read from ``app/app_config.yaml`` (paths relative to the repository).
If an RGB-D model is configured and a depth file is uploaded, it is used; otherwise the RGB-only
model is used. If no usable model can be loaded, the app shows an error instead of a result.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grasp_planner.inference.predictor import GraspPredictor  # noqa: E402
from grasp_planner.visualization.plots import render_prediction  # noqa: E402

CONVENTIONS_MD = """
**Conventions.** Pixel coordinates `(x, y)` have their origin at the **top-left** corner; `x` grows
to the right (column) and `y` grows **downwards** (row). The grasp angle `θ` is the direction of the
gripper's closing line, measured **counter-clockwise from the +x axis as seen on screen**, in
`[-90°, 90°)` (a parallel-jaw grasp is identical after a 180° turn). `width` is the gripper opening
in pixels. Contact points are `centre ∓ (width/2)·(cos θ, −sin θ)`.

**Limitations.** The model was trained on the Cornell Grasping Dataset: **one object** on a
light table, seen from above, 640×480 images. It does not handle clutter or choose between several
objects, and it predicts a **2-D image-plane grasp** only (no 3-D pose, no camera-to-robot
calibration, no collision checking). The score is the network's quality output, **not** a
calibrated probability of physical success; no prediction here has been validated on a robot.
"""


class App:
    def __init__(self, rgbd_ckpt: Path | None, rgb_ckpt: Path | None):
        self.models: dict[str, GraspPredictor] = {}
        self.errors: list[str] = []
        for name, path in (("rgbd", rgbd_ckpt), ("rgb", rgb_ckpt)):
            if path is None:
                continue
            try:
                self.models[name] = GraspPredictor(path)
            except Exception as exc:  # surfaced in the UI, never replaced by a fake result
                self.errors.append(f"Could not load {name} model from {path}: {exc}")

    def status(self) -> str:
        lines = [
            f"- **{k}** model: `{v.cfg['experiment']['name']}` (epoch {v.checkpoint_epoch}, "
            f"device {v.device})"
            for k, v in self.models.items()
        ]
        lines += [f"- ⚠️ {e}" for e in self.errors]
        if not self.models:
            lines.append(
                "- ❌ **No model loaded — predictions are disabled.** Train a model and set the "
                "checkpoint path in `app/app_config.yaml` or pass `--checkpoint`."
            )
        return "\n".join(lines)

    def run(self, image, depth_file, num_candidates: int):
        import gradio as gr

        if not self.models:
            raise gr.Error("No trained model is loaded. See the status panel for details.")
        if image is None:
            raise gr.Error("Please upload an RGB image.")
        rgb = np.asarray(image.convert("RGB"))
        depth = None
        if depth_file is not None:
            import tifffile

            depth = tifffile.imread(depth_file if isinstance(depth_file, str) else depth_file.name).astype(
                np.float32
            )
            if depth.shape[:2] != rgb.shape[:2]:
                raise gr.Error(f"Depth size {depth.shape[:2]} does not match image size {rgb.shape[:2]}.")
        if depth is not None and "rgbd" in self.models:
            key = "rgbd"
        elif "rgb" in self.models:
            key = "rgb"
        else:
            raise gr.Error("Only an RGB-D model is loaded: please also upload the matching depth TIFF.")
        pred = self.models[key]
        result = pred.predict(rgb, depth if key == "rgbd" else None, max_candidates=int(num_candidates))
        annotated = render_prediction(rgb, result.candidates, num_candidates=int(num_candidates))
        rows = []
        for i, g in enumerate(result.candidates):
            (x1, y1), (x2, y2) = g.contact_points()
            rows.append(
                [
                    i + 1,
                    round(g.score, 3),
                    round(g.x, 1),
                    round(g.y, 1),
                    round(float(np.rad2deg(g.theta)), 1),
                    round(g.width, 1),
                    f"({x1:.1f}, {y1:.1f})",
                    f"({x2:.1f}, {y2:.1f})",
                ]
            )
        if result.best is None:
            summary = f"Model `{key}`: no candidate passed the validity checks / quality threshold."
        else:
            b = result.best
            (x1, y1), (x2, y2) = b.contact_points()
            summary = (
                f"### Selected grasp (model `{pred.cfg['experiment']['name']}`, input `{key}`)\n"
                f"- centre: **({b.x:.1f}, {b.y:.1f}) px**\n- angle θ: **{np.rad2deg(b.theta):.1f}°**\n"
                f"- opening width: **{b.width:.1f} px**\n"
                f"- quality score: **{b.score:.3f}** (not a probability)\n"
                f"- contact points: **({x1:.1f}, {y1:.1f})** and **({x2:.1f}, {y2:.1f})** px"
            )
        out_dir = Path(tempfile.mkdtemp(prefix="grasp_"))
        from PIL import Image

        png = out_dir / "grasp_prediction.png"
        Image.fromarray(annotated).save(png)
        js = out_dir / "grasp_prediction.json"
        js.write_text(
            json.dumps(
                {
                    "model": pred.cfg["experiment"]["name"],
                    "input": key,
                    "candidates": [
                        dict(
                            zip(
                                [
                                    "rank",
                                    "score",
                                    "x",
                                    "y",
                                    "angle_deg",
                                    "width_px",
                                    "contact_1",
                                    "contact_2",
                                ],
                                r,
                                strict=True,
                            )
                        )
                        for r in rows
                    ],
                },
                indent=1,
            )
        )
        return annotated, summary, rows, [str(png), str(js)]


def build_ui(app: App):
    import gradio as gr

    with gr.Blocks(title="Grasp Planner") as ui:
        gr.Markdown("# Vision-based grasp planner (Cornell, planar parallel-jaw)")
        gr.Markdown(app.status())
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="RGB image (Cornell-style, single object)")
                depth = gr.File(
                    label="Optional depth TIFF (same size, metres) — enables the RGB-D model",
                    file_types=[".tiff", ".tif"],
                )
                k = gr.Slider(1, 10, value=5, step=1, label="Number of candidates to display")
                btn = gr.Button("Predict grasps", variant="primary")
            with gr.Column():
                out_img = gr.Image(
                    label="Candidates (amber, rank:score) · selected (blue) · contacts (magenta)"
                )
                summary = gr.Markdown()
        table = gr.Dataframe(
            headers=["rank", "score", "x", "y", "angle_deg", "width_px", "contact_1", "contact_2"],
            label="Ranked candidates (original-image pixels)",
        )
        files = gr.File(label="Download annotated image + JSON", file_count="multiple")
        gr.Markdown(CONVENTIONS_MD)
        btn.click(app.run, [image, depth, k], [out_img, summary, table, files])
        examples_dir = REPO / "app" / "examples"
        if examples_dir.exists():
            # pcdXXXXr.png, optionally with its pcdXXXXd.tiff depth (enables the RGB-D model)
            rows = []
            for png in sorted(examples_dir.glob("*.png")):
                tiff = png.with_name(png.name.replace("r.png", "d.tiff"))
                rows.append([str(png), str(tiff) if tiff.exists() else None])
            if rows:
                gr.Examples(rows, inputs=[image, depth], label="Example Cornell test-split scenes")
    return ui


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None, help="RGB-D checkpoint (overrides app_config.yaml)")
    ap.add_argument("--rgb-checkpoint", default=None, help="RGB-only checkpoint (overrides app_config.yaml)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 inside a container / HF Space")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg_path = REPO / "app" / "app_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    def resolve(p):
        return None if not p else (Path(p) if Path(p).is_absolute() else REPO / p)

    app = App(
        resolve(args.checkpoint or cfg.get("rgbd_checkpoint")),
        resolve(args.rgb_checkpoint or cfg.get("rgb_checkpoint")),
    )
    print(app.status())
    build_ui(app).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
