# CLAUDE.md — standing rules for AI assistants working on this repository

This repository is a vision-based planar grasp planner trained on the Cornell Grasping Dataset.
Read `PROJECT_STATUS.md` first: it is the hand-off record and says exactly where work stopped.

## Non-negotiable rules

1. **Dataset path comes from local configuration.** It is resolved (in order) from the
   `GRASP_DATASET_ROOT` environment variable, then `configs/local.yaml` (git-ignored, see
   `configs/local.example.yaml`). Never hard-code a machine-specific path in source code.
2. **The dataset is read-only.** Never move, rename, modify or delete anything inside the
   dataset folder. All derived artefacts go under `outputs/` or `experiments/`.
3. **Never commit data, training checkpoints, virtual environments or secrets.** Training
   checkpoints live in `outputs/` (git-ignored). Only small metric JSON/CSV files and figures used
   by the report may be committed under `experiments/` and `docs/figures/`. **Exception (approved
   by the owner, 2026-10-23):** the slim inference checkpoints in `models/*.pt` (weights + config,
   from `scripts/export_slim_checkpoint.py`) are committed so the app and the Hugging Face Space
   work from a fresh clone. Raw dataset files (images, depth, annotations) never go to GitHub;
   example scenes are copied into the Space only (`scripts/build_space.py`). Small annotated
   figures in `docs/figures/` and `docs/presentation/` are allowed.
4. **Never fabricate results.** Every number in the docs must come from a file produced by a
   script in `scripts/`. If something was not run, say so.
5. **The test split is touched only by `scripts/evaluate.py --split test`, after model
   selection.** Never use test data for training, tuning or checkpoint selection.
6. **Always update `PROJECT_STATUS.md` before stopping.**

## Environment

- Windows 11, Python 3.12, project venv at `.venv` (`.venv\Scripts\python.exe`).
- GPU: NVIDIA RTX 4060 Laptop (8 GB). PyTorch CUDA build installed from the cu128 index.
- Run long training jobs in the background, writing to a log under `outputs/`.
- DataLoader workers: entry points are guarded with `if __name__ == "__main__":`.

## Common commands

```
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check src scripts tests app
.venv\Scripts\python.exe scripts\train.py --config configs\experiment_configs\<exp>.yaml
.venv\Scripts\python.exe scripts\evaluate.py --run outputs\runs\<exp> --split val
```

## Conventions

- Pixel coordinates: origin top-left, `x` to the right (column), `y` down (row).
- Grasp angle `theta`: radians, measured counter-clockwise from the +x image axis *as seen on
  screen*, wrapped to `[-pi/2, pi/2)`. Grasps are symmetric under `theta + pi`.
- Grasp `width`: gripper opening in pixels (distance between the two jaw contact points).
