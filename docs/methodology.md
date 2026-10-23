# Methodology (summary)

The full treatment is `technical_report.md` (Chapters 3, 4, 7, 8). This page lists the decisions
in the order the data flows.

1. **Parse** every scene. Reject and report invalid rectangles: NaN, degenerate, corners off by
   more than 10°, or vertices outside the image. Exclude scenes that have no valid positive
   rectangle. (`data/cornell.py`)
2. **Split** before anything else: image-wise random (seed 42, 70/15/15) and contiguous-block
   (approximate object-wise). (`data/splits.py`)
3. **Choose the crop** (centre, side) from training annotations only. (`scripts/make_splits.py`)
4. **Preprocess**: one affine `M` combines crop, resize to 224, and, for training only, rotation
   ±180°, zoom 0.85–1, translation ±20 px and a horizontal flip with p = 0.5. Images are
   resampled with `M`; rectangle vertices are mapped with `M`; grasp parameters are recomputed
   from the mapped vertices. Depth: values < 0.3 m are invalid → Navier–Stokes inpainting →
   subtract the per-image mean → clip to [−1, 1]. RGB: scale to [0, 1] → subtract the per-image
   mean. (`data/transforms.py`, `data/dataset.py`)
5. **Targets**: the centre third of each positive rectangle gets Q = 1, cos 2θ, sin 2θ and
   W = width / 150. Everything else is 0.
6. **Train**: Adam (lr 1e-3), cosine decay to 1e-5, batch 8, 100 epochs, AMP, gradient clipping 5,
   early stopping after 30 epochs without improvement. Loss: sum of four Smooth-L1 terms.
   (`training/`)
7. **Select** the checkpoint with the best validation top-1 rectangle accuracy.
8. **Decode**: Gaussian-smooth Q, find local maxima, apply validity checks, angle-aware NMS, rank
   by Q. (`inference/decode.py`)
9. **Evaluate** once on the held-out test split with the Cornell rectangle metric.
   (`evaluation/`, `scripts/evaluate.py`)
