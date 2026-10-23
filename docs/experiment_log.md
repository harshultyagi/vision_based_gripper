# Experiment log

Chronological record of every run in this project, including smoke tests. Numbers come from
`outputs/runs/<run>/train_result.json`, `history.csv` and `experiments/results/*.json`.

## 2026-10-23 — Session 1

### Environment
Windows 11, Python 3.12.7 (project venv), PyTorch 2.11.0+cu128 (CUDA runtime 12.8), NVIDIA GeForce RTX 4060
Laptop GPU (8 GB, driver 592.82), AMD Ryzen 7 7840HS, 15.3 GB RAM. Recorded per run in `env.json`.

### Smoke tests (NOT real training)
`scripts/train.py --smoke`: 2 epochs × 3 batches, 16 validation images. Purpose: check that the
pipeline runs end to end on real data (loading, AMP, checkpointing, validation decoding).
All three passed. Their accuracies (0 %) are meaningless and are not reported anywhere.
Smoke run directories were deleted before the real runs.

| Config | Result |
|---|---|
| grconvnet_rgbd_image --smoke | ran; 1,900,900 parameters; checkpoints written |
| ggcnn_rgbd_image --smoke | ran; 70,196 parameters |
| grconvnet_rgb_image --smoke | ran |

### Full training runs
All runs use `configs/default.yaml` plus a 6-line override: identical crop, augmentation,
loss, optimiser (Adam 1e-3, cosine to 1e-5), batch 8, ≤ 100 epochs, early-stopping patience 30,
seed 42, decoding. Launched sequentially in the background with `scripts/run_experiments.py`.

| Run | Model | Input | Split | Epochs run | Best epoch | Best val top-1 | Wall time |
|---|---|---|---|---|---|---|---|
| grconvnet_rgbd_image | GR-ConvNet | RGB-D | image | 100 | 85 | 93.2 % | 631 s |
| ggcnn_rgbd_image | GG-CNN | RGB-D | image | 75 (early stop) | 44 | 88.0 % | 406 s |
| grconvnet_rgbd_block | GR-ConvNet | RGB-D | block | 49 (early stop) | 18 | 80.3 % | 319 s |
| ggcnn_rgbd_block | GG-CNN | RGB-D | block | 50 (early stop) | 19 | 73.8 % | 224 s |
| grconvnet_rgb_image | GR-ConvNet | RGB | image | 80 (early stop) | 49 | 90.2 % | 466 s |

(Wall time from `run_experiments.log`, including start-up and per-epoch validation.)

### Model selection — frozen before any test evaluation
Decided at 2026-10-23 ≈ 18:55 local, using validation results only
(`experiments/results/*__val.json`):

* **Final model: `grconvnet_rgbd_image`** (`outputs/runs/grconvnet_rgbd_image/checkpoints/best.pt`,
  epoch 85). It has the highest validation top-1 accuracy among image-wise runs (93.2 % vs 88.0 %
  for GG-CNN and 90.2 % for RGB-only GR-ConvNet). GR-ConvNet also beats GG-CNN on the block split
  (80.3 % vs 73.8 %).
* **RGB-only model for the app** (uploads without depth): `grconvnet_rgb_image`, epoch 49.
* No hyperparameter was tuned on validation results. The configuration was fixed before training
  and each configuration was trained exactly once.

After this point each run's `best.pt` is evaluated **once** on its test split
(`scripts/evaluate.py --split test`). The test numbers are reported as they come and are not
used to change any model, setting or selection.

### Test evaluation (run once per model, ≈ 19:00 local, after the selection above)

`scripts/evaluate.py --split test` for each run's `best.pt`. GPU idle during latency timing.

| Run | Test split | n | Test top-1 | 95 % CI | Test top-5 | Val top-1 |
|---|---|---|---|---|---|---|
| **grconvnet_rgbd_image (final)** | image | 132 | **88.6 %** | 82.1–93.0 % | 95.5 % | 93.2 % |
| ggcnn_rgbd_image | image | 132 | 84.8 % | 77.8–90.0 % | 93.2 % | 88.0 % |
| grconvnet_rgb_image | image | 132 | 90.2 % | 83.9–94.2 % | 97.7 % | 90.2 % |
| grconvnet_rgbd_block | block | 116 | 75.9 % | 67.3–82.7 % | 90.5 % | 80.3 % |
| ggcnn_rgbd_block | block | 116 | 78.4 % | 70.1–85.0 % | 91.4 % | 73.8 % |

Observations, recorded without acting on them:
* The selected model's test accuracy is 4.6 points below its validation accuracy. Picking the best
  of 100 noisy validation epochs is optimistic, and the 133-image validation set is small.
* The RGB-only GR-ConvNet scored higher on test than the RGB-D model (90.2 vs 88.6 %). Paired
  exact McNemar test: p = 0.80, so this is not a significant difference. The selection, made on
  validation, is **not** changed.
* On the block split GG-CNN scored higher on test (78.4 vs 75.9 %), the reverse of validation.
  McNemar p = 0.65.
* No pairwise difference between models is significant at the 5 % level on either split
  (`experiments/results/statistics.json`).
