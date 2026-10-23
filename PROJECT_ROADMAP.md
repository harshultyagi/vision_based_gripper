# Project Roadmap — Vision-Based Robotic Grasp Planner

Planar parallel-jaw grasp detection on the Cornell Grasping Dataset with dense fully
convolutional networks (GG-CNN baseline, GR-ConvNet main model).

## Phases, deliverables and acceptance criteria

| # | Phase | Deliverables | Acceptance criteria | Depends on |
|---|-------|--------------|---------------------|------------|
| 1 | Inspection & planning | `CLAUDE.md`, local config template, venv with CUDA PyTorch, this roadmap | GPU visible to PyTorch; dataset structure documented | – |
| 2 | Data pipeline | `data/cornell.py`, `transforms.py`, `dataset.py`, `splits.py`; `scripts/inspect_dataset.py`, `scripts/make_splits.py`; stats + exclusions JSON; diagnostic figures; split files | All 885 scenes parsed or excluded with a reason; split files disjoint; transform tests pass | 1 |
| 3 | Models | GG-CNN, GR-ConvNet, shared heads; shape/gradient tests | Forward pass shape = input grid; finite gradients | 2 |
| 4 | Training | `training/trainer.py`, `scripts/train.py`, YAML configs; smoke test; full background runs | Each full run finishes (log says `TRAINING FINISHED`), `best.pt` selected on validation only | 3 |
| 5 | Evaluation | `evaluation/`, `scripts/evaluate.py`; val comparison; one final test evaluation per selected model | Metrics JSON in `experiments/results/`; figures from real runs | 4 |
| 6 | Inference & app | `inference/`, `scripts/predict.py`, `app/app.py` (Gradio) | Real prediction from saved checkpoint, contact points shown, clear error on missing model | 4 |
| 7 | Validation & reproducibility | full pytest run, ruff, checkpoint reload, metrics re-derived | All tests pass; recorded in STATUS | 5, 6 |
| 8 | Documentation | technical report (textbook), README, CONTRIBUTING, docs/* | Every number traceable to a result file; every snippet copied from code | 5–7 |
| 9 | GitHub & audit | milestone commits, push to `harshultyagi/vision_based_gripper` | No data/checkpoints/secrets committed; push verified | 8 |

## Experiment plan

All runs: identical preprocessing (fixed crop chosen from the train split, 224×224 input),
identical augmentation, loss (sum of Smooth-L1), optimiser (Adam 1e-3, cosine), epochs,
seed and decoding. Selection criterion: validation top-1 rectangle accuracy.

| Run | Model | Input | Split | Purpose |
|-----|-------|-------|-------|---------|
| A | GG-CNN | RGB-D | image-wise | baseline |
| B | GR-ConvNet | RGB-D | image-wise | main model |
| C | GG-CNN | RGB-D | block-wise | baseline, harder split |
| D | GR-ConvNet | RGB-D | block-wise | main model, harder split |
| E | GR-ConvNet | RGB | image-wise | RGB-only model for uploads without depth |

## Key risks

* **No object-ID file** in the local dataset → object-wise split cannot be built reliably;
  a contiguous-block split with boundary gaps is used and documented as an approximation.
* **Small dataset (885 scenes)** → overfitting; mitigated by augmentation, early stopping,
  validation-based selection.
* **Limited free RAM (~1.5 GB free at start)** → no in-memory image cache; inpainted depth
  cached on disk.

## Deferred / future work

* True object-wise split (needs Cornell's `z.txt` object mapping; would require downloading
  it — not done without the owner's approval).
* Multi-seed runs / cross-validation for confidence intervals.
* Jacquard dataset, cluttered scenes, robot-frame transforms, physical validation.
