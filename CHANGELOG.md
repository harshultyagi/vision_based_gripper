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
