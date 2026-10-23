# PROJECT_STATUS — hand-off record

_Last updated: 2026-10-23, end of session 1 (final check re-run: 60 tests pass, lint clean, report checker clean)._

## Current state
**All 13 acceptance criteria are met.** The pipeline is complete: data → splits → five fully
trained models → one-time test evaluation → CLI + Gradio app → report. Nothing is half-finished.

## Completed milestones
1. Inspection: environment, GPU, dataset (885 scenes, all usable; 1 NaN rectangle rejected).
2. Data pipeline + splits (`experiments/splits/*.json`) + crop chosen from train only.
3. Models: GG-CNN (70,196 params), GR-ConvNet (1,900,900 params).
4. Smoke tests, then five full background runs (all finished, logs end in `TRAINING FINISHED`).
5. Validation comparison → selection frozen and committed → single test evaluation.
6. `scripts/predict.py` and `app/app.py` verified with real checkpoints.
7. Tests and lint pass. Clean-clone reproduction checked (see below).
8. Report (`docs/technical_report.md`) and all docs written; `scripts/check_report.py` passes.

## Architecture and config
* Input: RGB-D (depth, R, G, B), fixed crop (304, 275, side 288) → 224 × 224.
* Model: GR-ConvNet; outputs quality, cos 2θ, sin 2θ, width maps; Smooth-L1 loss.
* Training: Adam 1e-3, cosine to 1e-5, batch 8, ≤ 100 epochs, AMP, patience 30, seed 42.
* All settings: `configs/default.yaml` + `configs/experiment_configs/*.yaml`.

## Dataset path
Not stored in the repo. It is set in `configs/local.yaml` (git-ignored; template
`configs/local.example.yaml`) or the env var `GRASP_DATASET_ROOT`. This machine has a
`configs/local.yaml` pointing at the local Cornell copy.

## Best checkpoint
`outputs/runs/grconvnet_rgbd_image/checkpoints/best.pt` (epoch 85, SHA-256 prefix
`23fda02a829da78c`, 22.9 MB, **not in Git**). RGB-only model:
`outputs/runs/grconvnet_rgb_image/checkpoints/best.pt` (epoch 49).
Load with:
```python
from grasp_planner.inference.predictor import GraspPredictor
p = GraspPredictor("outputs/runs/grconvnet_rgbd_image/checkpoints/best.pt")
result = p.predict(rgb_uint8_hw3, depth_metres_hw)   # result.candidates, result.best
```
Back these files up if they matter: they exist only on this laptop.

## Latest measured results (test split, once)
| Run | Test top-1 | Test top-5 |
|---|---|---|
| grconvnet_rgbd_image (final) | 88.6 % | 95.5 % |
| ggcnn_rgbd_image | 84.8 % | 93.2 % |
| grconvnet_rgb_image | 90.2 % | 97.7 % |
| grconvnet_rgbd_block | 75.9 % | 90.5 % |
| ggcnn_rgbd_block | 78.4 % | 91.4 % |
No pairwise difference is statistically significant (`experiments/results/statistics.json`).
**The test split has now been used. Do not tune anything on it.** Any new experiment must be
selected on validation, and the test numbers above must be reported as already seen.

## Tests
* `python -m pytest -q` → 63 passed (unit, integration on synthetic scenes, real-dataset
  discovery, app error handling).
* `python -m ruff check src scripts tests app` → clean.
* `python scripts/check_report.py` → all code snippets and tagged numbers match.
* Clean-environment check (2026-10-23): fresh `git clone` → new venv → README install commands →
  `configs/local.yaml` from the template → CUDA available → **60 passed** → `make_splits.py` reproduced
  both split files identically (only `created_utc` differs) → `predict.py` with the trained checkpoint
  gave the same candidates → `check_report.py` passed. This check found and fixed a `.gitignore` bug
  that had hidden `src/grasp_planner/data/`.

## Known issues / caveats
* No true object-wise split (no `z.txt`). The block split is approximate.
* Single seed per configuration. Bit-exact reruns are not guaranteed (cuDNN non-determinism).
* Quality scores can exceed 1 (no output activation) and are not calibrated.
* `experiments/figures/` (per-image overlays) is git-ignored. Regenerate with `scripts/evaluate.py`.

## Git
Branch `main` in `C:\Users\harsh\work\VisionBasedGripper` (its own repository; the parent folder
`C:\Users\harsh` is an unrelated repository — never commit there). Remote `origin` =
https://github.com/harshultyagi/vision_based_gripper. **Pushed**: `main` tracks `origin/main`; after each push, verify with `git fetch && git status -sb` (should show no ahead/behind).

## Prioritised next actions
1. Obtain Cornell's `z.txt` (needs the owner's approval to download) → add a true object-wise
   split (`data/splits.py`) → retrain and re-evaluate.
2. 3–5 seeds or 5-fold cross-validation for confidence intervals on the model comparison.
3. Back-project the selected grasp to 3-D using depth and camera intrinsics.
4. Test-time augmentation; then Jacquard / cluttered datasets.

## Deployment status
* **Hugging Face Space prepared, not published** (the owner has no HF account yet). Target:
  https://huggingface.co/spaces/btechbillu/Vission-Based-grasper. Build with
  `scripts/build_space.py`; upload steps are in `docs/deployment.md`. Tested locally on CPU: both
  models load, and predictions match the GPU run.
* Slim checkpoints are committed in `models/` (owner-approved exception to the no-checkpoint rule).
* Presentation: `docs/presentation/Vision_Based_Grasp_Planner.pptx` (7 slides).

## Earlier deployment notes (owner first mentioned Vercel)
* `app/app.py` is a Gradio app: a long-running Python server that loads PyTorch and a 23 MB
  checkpoint. Vercel's Python support runs short-lived serverless functions with a bundle-size
  limit, and the PyTorch dependency alone is likely to exceed it. **Vercel is probably not a
  fit as-is.** This has not been tried.
* Realistic options: (a) Hugging Face Spaces (Gradio SDK) or another container host (Render,
  Railway, Fly.io) with CPU-only torch; (b) Vercel for a static front-end that calls a model API
  hosted elsewhere; (c) export the model to ONNX and run it with a lighter runtime.
* Either way the checkpoint is **not in Git**: it must be uploaded to the host (or to a model
  hub) separately, and the app must load it from there. `app/app_config.yaml` holds the paths.
* `.claude/launch.json` (untracked) is only the local launcher used for the in-app browser preview.
