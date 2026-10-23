# Vision-Based Robotic Grasp Planner

Given an image of an object, this project predicts **planar grasps for a parallel-jaw gripper**:
grasp centre, orientation, gripper opening, a quality score and the two contact points. It
generates several ranked candidates and selects the best one. Two dense convolutional networks
are trained and evaluated on the **Cornell Grasping Dataset**: GG-CNN (baseline) and GR-ConvNet
(main model).

> **Scope.** These are 2-D image-plane grasps on single-object Cornell scenes. There is no 3-D
> robot pose, no camera calibration, no collision checking, and **no physical validation**. A
> prediction that passes the Cornell metric is geometrically plausible, not proven to succeed on
> a robot.

📘 **Full technical report / textbook:** [`docs/technical_report.md`](docs/technical_report.md)

## Features
* Dataset: Cornell Grasping Dataset, Kaggle copy used here:
  https://www.kaggle.com/datasets/oneoneliu/cornell-grasp
* Validated Cornell loader: every malformed annotation is reported, none dropped silently.
* Augmentation where images and grasp labels share one affine transform, so they stay aligned
  by construction.
* Reproducible image-wise and contiguous-block (approximately object-disjoint) splits.
* GG-CNN and GR-ConvNet trained under identical conditions. Mixed precision, cosine LR, early
  stopping, resumable checkpoints.
* Decoding: Gaussian smoothing → local maxima → validity checks → angle-aware NMS → ranking →
  contact points.
* Cornell rectangle-metric evaluation, with confidence intervals and paired significance tests.
* CLI predictor and a Gradio web app.
* 63 pytest tests. A report checker verifies every code snippet and tagged number in the report.

## Architecture overview

```
RGB (+depth) 640x480 ─► fixed crop 288 px ─► 224x224 ─► GR-ConvNet / GG-CNN
   ─► maps: quality, cos2θ, sin2θ, width ─► peaks + NMS ─► ranked grasps (original pixels)
   ─► selected grasp: centre, θ, width, score, contact points
```

## Results (measured; held-out test split, evaluated once after model selection)

| Model | Input | Split (test images) | Top-1 | Top-5 | Latency fwd / fwd+decode |
|---|---|---|---|---|---|
| **GR-ConvNet (final)** | RGB-D | image-wise (132) | **88.6 %** (95 % CI 82.1–93.0) | 95.5 % | 3.7 / 5.8 ms |
| GG-CNN | RGB-D | image-wise (132) | 84.8 % | 93.2 % | 1.2 / 3.1 ms |
| GR-ConvNet | RGB | image-wise (132) | 90.2 % | 97.7 % | 3.9 / 5.8 ms |
| GR-ConvNet | RGB-D | block-wise (116) | 75.9 % | 90.5 % | — |
| GG-CNN | RGB-D | block-wise (116) | 78.4 % | 91.4 % | — |

Metric: the top-ranked grasp has IoU > 25 % and angle difference < 30° with at least one labelled
grasp, per image. Latency: RTX 4060 Laptop GPU, batch size 1. Differences between the models are
**not statistically significant** at these test-set sizes (paired McNemar p ≥ 0.33). Sources:
`experiments/results/*__test.json`, `model_comparison.md`, `statistics.json`. Published numbers
(e.g. 97.7 % for GR-ConvNet) were obtained under a different protocol; see report §9.8.

## Dataset requirements and setup

The Cornell Grasping Dataset is **not included**. The loader expects:

```
<dataset_root>/01 … 10/pcdXXXXr.png, pcdXXXXd.tiff, pcdXXXXcpos.txt, pcdXXXXcneg.txt
```

The `d.tiff` depth images are the ones produced by the GG-CNN repository's Cornell depth
conversion script. Point the project at your copy, without editing any code:

```bash
copy configs\local.example.yaml configs\local.yaml
```

Then set `dataset_root` in `configs/local.yaml`, or set the environment variable
`GRASP_DATASET_ROOT`. Details are in [`docs/dataset_documentation.md`](docs/dataset_documentation.md).
The dataset folder is only ever read.

## Installation

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Use the PyTorch index that matches your driver (see pytorch.org). A plain `pip install torch` on
Windows installs a CPU-only build. Check the GPU with
`.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`.
On Linux/macOS use `.venv/bin/python`.

## Usage

Prepare the data (deterministic):

```bash
.venv\Scripts\python.exe scripts\inspect_dataset.py
```

```bash
.venv\Scripts\python.exe scripts\make_splits.py --seed 42
```

Run the tests:

```bash
.venv\Scripts\python.exe -m pytest -q
```

Train (about 10 minutes on an RTX 4060 Laptop GPU; add `--smoke` for a 1-minute pipeline check,
`--resume` to continue):

```bash
.venv\Scripts\python.exe scripts\train.py --config configs\experiment_configs\grconvnet_rgbd_image.yaml
```

Evaluate (validation for model selection; `--split test` only once selection is final):

```bash
.venv\Scripts\python.exe scripts\evaluate.py --run outputs\runs\grconvnet_rgbd_image --split val
```

```bash
.venv\Scripts\python.exe scripts\summarize_results.py
```

Predict on one image (prints candidates as JSON and saves an annotated PNG plus a maps figure):

```bash
.venv\Scripts\python.exe scripts\predict.py --checkpoint outputs\runs\grconvnet_rgbd_image\checkpoints\best.pt --rgb path\to\pcd0101r.png --depth path\to\pcd0101d.tiff
```

Web app (http://127.0.0.1:7860). It uses the checkpoints named in `app/app_config.yaml`: RGB-D
when a depth TIFF is uploaded, otherwise the RGB-only model:

```bash
.venv\Scripts\python.exe app\app.py
```

The app uses the slim checkpoints committed in `models/` (7.3 MB each: weights + config, with
the training state removed), so it works straight after cloning. Full training checkpoints stay in
`outputs/` and are not in Git.

## Online demo (Hugging Face Space)

Planned at **https://huggingface.co/spaces/btechbillu/Vission-Based-grasper**. It runs on free CPU
hardware and offers four clickable test-split scenes. The Space is **not published yet**. Build it
with `scripts/build_space.py` and upload it as described in
[`docs/deployment.md`](docs/deployment.md). The built folder was tested locally on CPU.

## Presentation

A 7-slide summary deck (methodology, training choices, results, links):
[`docs/presentation/Vision_Based_Grasp_Planner.pptx`](docs/presentation/Vision_Based_Grasp_Planner.pptx).

## Repository layout

```
configs/      default.yaml, experiment_configs/, local.example.yaml
src/grasp_planner/  data/ models/ training/ evaluation/ inference/ visualization/ utils/
scripts/      inspect_dataset, make_splits, train, run_experiments, evaluate, predict, ...
app/          Gradio app
tests/        pytest suite
experiments/  splits/, results/ (committed metrics)
docs/         technical_report.md and supporting docs, figures/
outputs/      runs, logs, cache (git-ignored)
```

## Limitations
* Cornell only: single objects on a white board, one camera setup, 885 scenes.
* No true object-wise split (the object-ID file is missing locally). The block split is an
  approximation.
* One seed per configuration; test sets of 116–132 images give confidence intervals of about
  ±6 points.
* The quality score is not a calibrated success probability.
* No 3-D pose, calibration, collision checking or robot experiments.

## Status and next steps
All acceptance criteria are met; see [`PROJECT_STATUS.md`](PROJECT_STATUS.md). Planned next:
a true object-wise split once Cornell's `z.txt` is available, multi-seed or cross-validation
runs, and 3-D back-projection of the selected grasp. See the report, §11.2.

## License
MIT (code). The Cornell dataset has its own terms.
