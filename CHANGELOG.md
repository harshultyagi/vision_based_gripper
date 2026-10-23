# Changelog

All notable changes. Dates are UTC+5:30 (local).

## [1.0.0] — 2026-10-23

### Added
- Cornell loader for the local layout (`01`–`10`, `pcdXXXX{r.png,d.tiff,cpos.txt,cneg.txt}`) with
  per-rectangle validation and transparent exclusion reports.
- Single-affine preprocessing and augmentation (crop, resize, rotation, zoom, translation, flip)
  applied identically to pixels and grasp vertices.
- Image-wise and contiguous-block (approximate object-wise) split files, seed 42.
- Training-split-only crop selection (`experiments/results/crop_analysis.json`).
- GG-CNN baseline and GR-ConvNet models with shared dense output heads.
- Trainer: AMP, cosine LR, gradient clipping, NaN guard, validation-based best checkpoint,
  early stopping, atomic checkpoint writes, resume.
- Candidate decoding: Gaussian smoothing, local maxima, validity checks, angle-aware NMS, ranking,
  inverse mapping to original pixels, contact points.
- Cornell rectangle-metric evaluation (top-1/top-5, centre/angle/width error, latency).
- `scripts/`: inspect_dataset, make_splits, visualize_targets, train, run_experiments, evaluate,
  predict, summarize_results, make_report_figures.
- Gradio app (`app/app.py`) with RGB-D and RGB-only models.
- pytest suite (unit + integration on synthetic scenes + real-dataset discovery test).
- Documentation: technical report, dataset/evaluation/methodology/architecture docs.

### Fixed during development
- Depth TIFFs contain near-zero artefact pixels (~0.01 %): values < 0.3 m are now treated as
  invalid and inpainted.
- Depth cache keyed by source path + size + mtime, so scenes with the same ID from different
  folders (e.g. test fixtures) cannot collide.
- Validation log printed `nan` for a top-5 accuracy of exactly 0.
- `.gitignore` pattern `data/` also matched `src/grasp_planner/data/`, so the data package was
  missing from the first two commits. Found by the clean-clone reproduction test; patterns are
  now anchored to the repository root (`/data/`, `/outputs/`).
- App crashed at start-up on Windows consoles (cp1252) when printing the status text; stdout is
  now reconfigured to UTF-8.
- Per-image overlay figures (full-size dataset images, ~23 MB) are no longer committed.

### Deployment (added later on 2026-10-23)
- `models/*.pt`: slim inference checkpoints (7.3 MB, weights + config) committed with the
  owner's approval; `app/app_config.yaml` now points to them.
- `scripts/export_slim_checkpoint.py` (verifies identical outputs) and `scripts/build_space.py`
  (assembles a Hugging Face Space in `outputs/hf_space/`, CPU-only torch, four test-split example
  scenes with depth). Guide: `docs/deployment.md`.
- `app/app.py`: `--host` option (Spaces need 0.0.0.0); examples can include depth TIFFs.
- Fix: device selection now also requires a visible CUDA device. With `CUDA_VISIBLE_DEVICES=""`
  PyTorch reported CUDA as available with 0 devices, and checkpoint loading failed.

### Results
- Five full training runs and a one-time test evaluation; see `docs/experiment_log.md` and
  `experiments/results/`.
