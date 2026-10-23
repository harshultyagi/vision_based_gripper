# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ConfigError: Dataset path not configured` | no `configs/local.yaml` and no `GRASP_DATASET_ROOT` | copy `configs/local.example.yaml` → `configs/local.yaml` and set `dataset_root` |
| `Configured dataset root does not exist` | typo / drive not mounted | check the path; forward slashes work on Windows |
| `torch.cuda.is_available()` is `False` | CPU-only wheel (plain `pip install torch` on Windows) | `pip uninstall torch torchvision` then install from `https://download.pytorch.org/whl/cu128` (or the index matching your driver; see pytorch.org) |
| DataLoader hangs or crashes on Windows | multi-process workers | set `data.num_workers: 0` in the config (slower but always works). All entry points already use `if __name__ == "__main__":` |
| `NonFiniteLossError` | diverging training (too-high LR, bad data) | lower `train.lr`, keep `grad_clip`; the message names the batch and per-head losses |
| CUDA out of memory | GPU < 4 GB or other GPU apps | lower `train.batch_size` (e.g. 4) or `data.output_size`, keep `amp: true` |
| `split 'train' references scenes not found` | split file from a different dataset copy | regenerate: `python scripts/make_splits.py` |
| `--resume given but .../last.pt does not exist` | run never saved an epoch | start without `--resume` |
| App shows "No model loaded" | checkpoint path in `app/app_config.yaml` wrong or file missing | train a model, or pass `--checkpoint outputs/runs/<run>/checkpoints/best.pt` |
| App error "Only an RGB-D model is loaded" | no RGB-only checkpoint configured | upload the matching depth TIFF, or set `rgb_checkpoint` |
| Predictions on own photos look poor | domain shift: different camera, height, background, clutter | expected; the model only saw Cornell scenes. See the report's limitations chapter |
| `pip` very slow downloading torch | ~2.7 GB wheel | let it finish; do not interrupt (a partial download is discarded) |
