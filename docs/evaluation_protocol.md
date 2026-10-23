# Evaluation protocol

Implemented in `src/grasp_planner/evaluation/metrics.py` and `evaluation/runner.py`, run by
`scripts/evaluate.py`.

## Procedure per image

1. Preprocess with the **deterministic** pipeline (fixed crop → 224 × 224; no augmentation).
2. Forward pass → four maps; decode to ≤ 10 ranked candidates (see `inference/decode.py`).
3. Map candidates back to original 640 × 480 pixel coordinates with the inverse crop affine.
   Predicted rectangles use `jaw_size = width / 2` (GG-CNN/GR-ConvNet convention; the network
   does not predict jaw size).
4. Compare with **all valid positive** GT rectangles of that image, in original coordinates.

## Metrics

| Metric | Definition | Aggregation |
|---|---|---|
| top-1 accuracy | top candidate matches ≥ 1 GT rectangle under the Cornell criterion | mean over images; no prediction = failure |
| top-5 accuracy | any of the first 5 candidates matches ≥ 1 GT rectangle | mean over images |
| best IoU | max IoU of the top candidate with any GT rectangle | mean/median over images with a prediction |
| centre error | Euclidean px distance from top candidate centre to the **nearest** GT centre | mean/median over images with a prediction |
| angle error | angle difference (mod 180°) to that nearest-centre GT rectangle | same |
| width error | absolute opening difference (px) to that nearest-centre GT rectangle | same |
| latency | median of 100 timed batch-size-1 runs after 10 warm-ups: forward pass, and forward + decode | per checkpoint, hardware recorded |

**Cornell rectangle criterion.** A prediction *P* matches ground truth *G* when

* `IoU(P, G) = |P ∩ G| / |P ∪ G| > 0.25` (exact polygon areas via Shapely), **and**
* `Δθ = min(|θ_P − θ_G| mod π, π − (|θ_P − θ_G| mod π)) < 30°`.

The modulo-π handles the 180° symmetry of a parallel-jaw grasp.

The nearest-centre pairing for the error metrics is our choice: an image has several valid GT
grasps and there is no single "correct" one. These errors are descriptive, not a success criterion.

## Rules

* Checkpoint selection: **best validation top-1** (ties: lower validation loss), fixed before training.
* The test split is evaluated **once** per selected model, after all selection decisions; those
  results are never used to change anything.
* Models are compared only on the same split file, preprocessing, decoding and metric.
* Training-set accuracy is never reported as test accuracy.
