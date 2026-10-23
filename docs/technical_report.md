# Vision-Based Robotic Grasp Planner — Technical Report and Textbook

*Planar parallel-jaw grasp detection on the Cornell Grasping Dataset with GG-CNN and GR-ConvNet.*

> **How to read this document.** Chapters 2–4 explain the problem and the mathematics, with
> worked examples computed by the project's own code. Chapters 5–8 walk through the
> implementation and quote the real source (every quoted block carries a `src:` tag that
> `scripts/check_report.py` verifies against the repository). Chapter 9 reports the measured
> results. Every number tagged `num:` is checked against the JSON file it came from. Read it next
> to the code: Appendix H maps each chapter to its files.

## 1. Executive Summary

**Objective.** Build, train and evaluate a vision-based grasp planner. It takes an image of a
single object and predicts planar grasps for a parallel-jaw gripper: centre, orientation,
opening width, a quality score and the two contact points. It ranks several candidates and
selects the best one according to the model. Training and evaluation use the local copy of the
Cornell Grasping Dataset.

**What was built.** A complete, tested PyTorch project (`src/grasp_planner/`):
* a validated Cornell loader (885/885 scenes usable; 1 malformed rectangle rejected and reported);
* single-affine preprocessing and augmentation that keeps pixels and labels aligned by construction;
* reproducible image-wise and contiguous-block (approximately object-disjoint) splits;
* two dense models trained under identical conditions: GG-CNN (70 k parameters) and GR-ConvNet
  (1.9 M);
* a decoder: peak finding, validity checks, angle-aware NMS, ranking and back-projection;
* the Cornell rectangle metric;
* a CLI predictor and a Gradio web app;
* 63 automated tests;
* this report, whose snippets and numbers are machine-checked against the code and result files.

**Final model.** GR-ConvNet with RGB-D input, selected on validation accuracy, at
`outputs/runs/grconvnet_rgbd_image/checkpoints/best.pt` (epoch 85, not committed to Git).

**Measured results (held-out test split, evaluated once after selection was frozen).**

| Model | Split | Test top-1 | Test top-5 | Latency (fwd / fwd + decode) |
|---|---|---|---|---|
| **GR-ConvNet RGB-D (final)** | image-wise, 132 images | **<!-- num: experiments/results/grconvnet_rgbd_image__test.json:top1_accuracy:.1% -->88.6%** (95 % CI 82.1–93.0) | <!-- num: experiments/results/grconvnet_rgbd_image__test.json:top5_accuracy:.1% -->95.5% | 3.7 / 5.8 ms |
| GG-CNN RGB-D | image-wise | <!-- num: experiments/results/ggcnn_rgbd_image__test.json:top1_accuracy:.1% -->84.8% | <!-- num: experiments/results/ggcnn_rgbd_image__test.json:top5_accuracy:.1% -->93.2% | 1.2 / 3.1 ms |
| GR-ConvNet RGB only | image-wise | <!-- num: experiments/results/grconvnet_rgb_image__test.json:top1_accuracy:.1% -->90.2% | <!-- num: experiments/results/grconvnet_rgb_image__test.json:top5_accuracy:.1% -->97.7% | 3.9 / 5.8 ms |
| GR-ConvNet RGB-D | block-wise, 116 images | <!-- num: experiments/results/grconvnet_rgbd_block__test.json:top1_accuracy:.1% -->75.9% | <!-- num: experiments/results/grconvnet_rgbd_block__test.json:top5_accuracy:.1% -->90.5% | — |
| GG-CNN RGB-D | block-wise | <!-- num: experiments/results/ggcnn_rgbd_block__test.json:top1_accuracy:.1% -->78.4% | <!-- num: experiments/results/ggcnn_rgbd_block__test.json:top5_accuracy:.1% -->91.4% | — |

Metric: the top-ranked grasp has IoU > 25 % and angle difference < 30° with at least one
labelled grasp. Latency is measured on an RTX 4060 Laptop GPU with batch size 1.

**Key findings.**
1. A dense network trained in about 10 minutes on a laptop GPU finds a Cornell-valid grasp first
   in about 9 of 10 held-out scenes, and among its top five in about 19 of 20.
2. Holding out blocks of consecutive scenes (usually the same object) costs about 6–13 points,
   so image-wise accuracy partly reflects familiarity with the object.
3. The differences between GR-ConvNet and GG-CNN, and between RGB-D and RGB, are **not
   statistically significant** on test sets of this size (paired McNemar p ≥ 0.33). The large
   published gap between these models was not reproduced under our identical training recipe.
4. The quality score ranks candidates within an image but does not predict whether the chosen
   grasp matches a label (identical median scores for successes and failures).

**Key limitations.** Single-object Cornell scenes only. 2-D image-plane grasps only: no 3-D pose,
no calibration, no collision checking. **No physical validation.** No true object-wise split
(object IDs unavailable). One seed per configuration. Published numbers (e.g. 97.7 % for
GR-ConvNet) were obtained under a different protocol and are not directly comparable.

---

## 2. Problem Formulation

### 2.1 What robotic grasp detection is

A robot that has to pick up an object must decide **where** to put its fingers and **how** to
orient them. *Grasp detection* is the perception part of that decision: given sensor data
(here an RGB image and a depth image taken from above), output one or more **grasp
configurations** that would let the gripper hold the object. In this project the gripper is a
**parallel-jaw gripper**: two flat plates that move towards each other along one line until they
touch the object. That is the most common industrial and research gripper.

Grasp detection is *not* the whole grasping pipeline. A complete system also needs (i) camera
calibration to convert image pixels into 3-D robot coordinates, (ii) motion planning to reach
the grasp without collisions, (iii) control to close the gripper and lift, and (iv) a check that
the object was actually lifted. This project implements only the image-level detection step
and evaluates it against human annotations. Section 2.5 makes the distinction precise.

### 2.2 Parallel-jaw geometry and the grasp rectangle

Jiang, Moseson and Saxena [Jiang 2011] proposed representing a top-down parallel-jaw grasp as
an **oriented rectangle in the image plane**:

```
            jaw plate (p1 → p2)
     p0 ────────────────────── p1         ← edge p0→p1: the closing direction,
      │            ↔            │            its length = gripper opening "width" w
      │         centre c        │
      │                         │
     p3 ────────────────────── p2
            jaw plate (p3 → p0)            ← edge p1→p2: a jaw plate, length = "jaw size" h
```

Two sides of the rectangle are the gripper plates. The other two sides join the plates and
lie along the **closing direction**. A rectangle has five degrees of freedom:

| Symbol | Meaning | Unit |
|---|---|---|
| `x, y` | rectangle centre = midpoint between the two plates | pixels |
| `θ` | angle of the closing direction | radians / degrees |
| `w` | opening: distance between the plates | pixels |
| `h` | jaw size: length of each plate | pixels |

The networks in this project predict `(x, y, θ, w)` plus a **quality score** `q` at every pixel.
They do **not** predict `h`. When a predicted grasp must be drawn or compared as a rectangle,
`h = w/2` is used, the same convention as the GG-CNN and GR-ConvNet reference code.

### 2.3 Contact points

The two **contact points** are where the plates' centres meet the object's outline, at half
the opening on either side of the centre along the closing direction:

$$\mathbf{c}_{1,2} = (x, y) \mp \tfrac{w}{2}\,\mathbf{d}, \qquad \mathbf{d} = (\cos\theta,\; -\sin\theta).$$

The minus sign in `d` comes from the image coordinate convention (next section). For the first
positive rectangle of scene 0100 (worked example in §3.3) the contact points are
(252.0, 332.7) and (308.0, 337.0). These are exactly the midpoints of the two plate edges.

### 2.4 Coordinate and angle conventions (used everywhere in the code)

* **Pixels.** Origin at the **top-left** corner; `x` = column, growing to the right; `y` = row,
  growing **downwards**. Cornell files store vertices as `x y`.
* **Angle.** `θ` is the direction of the closing line (`p0 → p1`), measured
  **counter-clockwise from the +x axis as seen on the screen**. Because `y` grows downwards,
  a vector `(dx, dy)` has on-screen angle `atan2(−dy, dx)`.
* **Symmetry.** Rotating a parallel-jaw grasp by 180° gives the same physical grasp, so `θ` is
  wrapped to `[−π/2, π/2)`, i.e. [−90°, 90°).
* **Width.** `w` is the opening in pixels of whichever image is being discussed. The network's
  input is a resized crop, so widths are converted whenever coordinates change (§3.4).

<!-- src: src/grasp_planner/data/grasp.py -->
```python
def wrap_angle(theta: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (radians) to ``[-pi/2, pi/2)``, the canonical range of a grasp angle."""
    return (theta + HALF_PI) % np.pi - HALF_PI


def angle_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two grasp angles, respecting the pi symmetry.

    Result is in ``[0, pi/2]``. Example: 80 deg vs -80 deg -> 20 deg.
    """
    d = abs(a - b) % np.pi
    return float(min(d, np.pi - d))
```

`wrap_angle` shifts by π/2, takes the remainder modulo π (Python's `%` always returns a value in
`[0, π)` for a positive modulus) and shifts back. `angle_difference` first reduces the raw
difference modulo π, then takes the shorter way round. 80° and −80° differ by 160°, which is
the same orientation as a 20° difference.

### 2.5 What a 2-D grasp prediction is — and is not

| Level | What it gives | Done here? |
|---|---|---|
| 2-D planar grasp (this project) | `(x, y, θ, w)` in image pixels + a score | **yes** |
| 3-D grasp pose | gripper position and orientation in the camera or robot frame (6-DoF + opening) | **no**; needs depth back-projection with camera intrinsics, plus a camera-to-robot transform |
| Executable grasp | 3-D pose + collision-free approach trajectory | **no** |
| Physically verified grasp | the robot lifted the object | **no**; no robot was used |

A prediction that passes the Cornell rectangle metric is a **geometrically plausible** grasp:
it overlaps a human-labelled good grasp and has a similar orientation. It is **not** evidence
that a robot would succeed. Nothing in this report claims physical grasp success.

---

## 3. Dataset and Data Engineering

### 3.1 The Cornell Grasping Dataset

Cornell [Jiang 2011] contains 885 RGB-D scenes. Each shows a single household object on a
white board, photographed from above by a Kinect-style sensor. Every scene has hand-labelled
**positive** (good) and **negative** (bad) grasp rectangles. It is small by deep-learning
standards, which shapes many decisions below (augmentation, a light model, early stopping).

### 3.2 The local copy: structure verified by inspection

`scripts/inspect_dataset.py` walks the dataset folder **read-only** and writes
`experiments/results/dataset_stats.json`. Findings:

| Item | Finding |
|---|---|
| Scene folders | `01`–`08` with 100 scenes each, `09` with 50, `10` with 35 → <!-- num: experiments/results/dataset_stats.json:num_usable_scenes -->885 usable scenes |
| Files per scene | `pcdXXXXr.png` (RGB), `pcdXXXXd.tiff` (depth), `pcdXXXXcpos.txt`, `pcdXXXXcneg.txt`, `pcdXXXX.txt` (point cloud) |
| Other | `backgrounds/` with <!-- num: experiments/results/dataset_stats.json:num_background_images -->11 object-free RGB images (no annotations) |
| RGB | 640 × 480 × 3, uint8, all 885 |
| Depth | float32, 640 × 480, metres; median depth ≈ 1.1–1.2 m |
| Positive rectangles | <!-- num: experiments/results/dataset_stats.json:positive_rectangles.total -->5110 valid (2–25 per scene) |
| Negative rectangles | <!-- num: experiments/results/dataset_stats.json:negative_rectangles.total -->2909 |
| Rejected rectangles | <!-- num: experiments/results/dataset_stats.json:rejected_rectangles -->1 (`pcd0165cpos.txt` #2 has `NaN` coordinates) |
| Excluded scenes | <!-- num: experiments/results/dataset_stats.json:num_excluded_scenes -->0 |
| Object identities | not available: the original `z.txt` object mapping is absent |

The depth TIFFs are not part of the original Cornell download. They are the dense depth images
produced from the point clouds by the GG-CNN repository's conversion script, and were already
present in the local copy. They have no missing pixels, but about 0.01 % of pixels per image (in
the image corners, outside the crop) have depths of a few millimetres. A Kinect cannot measure
that, so they are artefacts. The loader treats any depth below 0.3 m as invalid and inpaints it.

![Annotated examples](figures/dataset_examples.png)
*Figure 3.1 — Eight random scenes with positive (green) and negative (red) rectangles, drawn
from the parsed files by `scripts/inspect_dataset.py`.*

![Histograms](figures/dataset_histograms.png)
*Figure 3.2 — Distribution of opening width, jaw size, angle and rectangles per scene over all
valid positive rectangles.*

### 3.3 Parsing a rectangle — a worked example

The first four lines of `pcd0100cpos.txt` are

```
253 319.7
309 324
307 350
251 345.7
```

so `p0 = (253, 319.7)`, `p1 = (309, 324)`, `p2 = (307, 350)`, `p3 = (251, 345.7)`.
`rect_to_grasp` converts this to the grasp representation:

<!-- src: src/grasp_planner/data/grasp.py -->
```python
    dx, dy = pts[1] - pts[0]
    theta = wrap_angle(float(np.arctan2(-dy, dx)))
    width = float(np.hypot(dx, dy))
    jaw_size = float(np.linalg.norm(pts[2] - pts[1]))
    cx, cy = pts.mean(axis=0)
```

* `dx, dy = 56, 4.3` → `θ = atan2(−4.3, 56) = −4.39°`. The closing line points slightly
  *down* on screen, so the angle is slightly negative.
* `w = √(56² + 4.3²) = 56.16 px`, `h = |p2 − p1| = √(2² + 26²) = 26.08 px`.
* Centre `= mean of the vertices = (280.00, 334.85)`.
* Visual check (Figure 3.1 style overlay of scene 0100): the rectangle spans the remote control
  across its narrow side. So `p0→p1` really is the closing direction, which confirms the
  convention.

**Validation.** `parse_rectangle_file` (`data/cornell.py`) checks that the file has a multiple
of four lines and parses every value. `rectangle_geometry_errors` (`data/grasp.py`) then
rejects a rectangle that has non-finite coordinates, a zero-length side, or a corner more than
10° away from a right angle (hand labels are only approximately rectangular). A vertex outside
the 640 × 480 image is also rejected. Rejected rectangles are **listed** in `sample.issues`,
logged as warnings and written to `dataset_exclusions.json`; they are never dropped silently.
A scene left with no valid positive rectangle would be excluded with a reason. That did not
happen.

### 3.4 Preprocessing: one affine map for everything

The network sees a **224 × 224** image. To get there from 640 × 480, a fixed square window is
cropped and resized. During training it is also randomly rotated, zoomed, shifted and
mirrored. The project composes **all** of these into a single 2 × 3 affine matrix `M`:

$$\begin{pmatrix}x'\\y'\end{pmatrix} = \underbrace{s\,F\,R(\alpha)}_{A}\begin{pmatrix}x-c_x\\y-c_y\end{pmatrix} + \begin{pmatrix}\tfrac{S-1}{2}\\\tfrac{S-1}{2}\end{pmatrix},\qquad
R(\alpha) = \begin{pmatrix}\cos\alpha & \sin\alpha\\ -\sin\alpha & \cos\alpha\end{pmatrix},$$

where `(c_x, c_y)` is the crop centre (plus a random shift during training), `s = S / (L·z)`
is the scale for crop side `L`, output size `S` and zoom factor `z`, `F = diag(−1, 1)` if
mirrored (else the identity), and `α` is the rotation angle. `R(α)` is a **counter-clockwise
rotation as seen on screen** in a y-down frame. That is why its signs differ from the textbook
y-up rotation matrix.

<!-- src: src/grasp_planner/data/transforms.py -->
```python
    cx, cy = crop.cx + shift[0], crop.cy + shift[1]
    s = output_size / (crop.size * zoom)
    c, sn = np.cos(rotation_rad), np.sin(rotation_rad)
    rot = np.array([[c, sn], [-sn, c]])
    lin = s * rot
    if hflip:
        lin = np.diag([-1.0, 1.0]) @ lin
    half = (output_size - 1) / 2.0
    t = np.array([half, half]) - lin @ np.array([cx, cy])
    return np.hstack([lin, t[:, None]])
```

**Why one matrix?** The image is resampled once with `cv2.warpAffine(img, M)`. The rectangle
**vertices** are transformed with the same `M`, and `(x, y, θ, w)` is then *recomputed* from
the transformed vertices with `rect_to_grasp`. Because labels and pixels go through exactly
the same mapping, they cannot drift apart. No separate formula is needed for "how does θ change
under a flip", which is a classic source of bugs. The consequences follow from the algebra:

* rotation by α adds α to θ: `R(α)(cos θ, −sin θ)ᵀ = (cos(θ+α), −sin(θ+α))ᵀ`;
* the mirror `F` maps `(cos θ, −sin θ)` to `(−cos θ, −sin θ)`, which is the direction of `−θ`
  modulo π. So mirroring negates the angle;
* every length, including the opening `w`, is multiplied by `s`.

**Worked example** (computed by the code). With the chosen crop `c = (304, 275)`, `L = 288`,
`S = 224`: `s = 224/288 = 0.7778`, and

`M = [[0.7778, 0, −124.94], [0, 0.7778, −102.39]]`.

The scene-0100 grasp from §3.3 maps to centre (92.83, 158.05), `w = 43.68 px` and unchanged
`θ = −4.39°`. Adding a 90° rotation gives centre (158.05, 130.17) and `θ = 85.61°` (= −4.39° +
90°), with the same width. `tests/test_transforms.py` checks these identities, and also paints
a rectangle into an image, warps it, and checks the blob's centroid and principal axis against
the transformed grasp. That catches any mismatch between the pixel path and the label path.

**The crop window** is fixed and was chosen **only from training annotations**
(`scripts/make_splits.py`, written to `experiments/results/crop_analysis.json`). Its centre is
the median training grasp centre, (304, 275). Its side is the smallest multiple of 16 px that
contains at least 99 % of training rectangle vertices: 288 px. On the training split it covers
<!-- num: experiments/results/crop_analysis.json:chosen_train_coverage.vertex_fraction:.1% -->99.4% of
vertices and <!-- num: experiments/results/crop_analysis.json:chosen_train_coverage.rectangle_fraction:.1% -->98.4% of
whole rectangles. The GR-ConvNet reference code instead centres each crop on the **ground-truth
grasps of that image**. That is a mild form of label leakage at test time: the network is
always shown the object in the middle. The fixed window avoids it, at the cost of seeing a
little more background.

**Depth.** The pipeline masks invalid pixels (< 0.3 m), fills them by Navier–Stokes inpainting
(`cv2.INPAINT_NS` [Bertalmio 2001]), subtracts the per-image mean depth and clips to [−1, 1]
metres. **RGB** is scaled to [0, 1] and the per-image mean is subtracted. Mean subtraction makes
the input independent of the absolute camera height and of global brightness. The channel order
is (depth, R, G, B).

<!-- src: src/grasp_planner/data/dataset.py -->
```python
def normalise_depth(depth: np.ndarray) -> np.ndarray:
    """float depth (H, W) in metres -> (1, H, W): per-image mean removed, clipped to [-1, 1]."""
    d = depth - depth.mean()
    return np.clip(d, -1.0, 1.0)[None].astype(np.float32)
```

### 3.5 Label construction: from rectangles to dense maps

A dense network predicts a value at every pixel, so the labels must also be images. For each
positive rectangle (in network coordinates) the **centre third** of the rectangle is filled:
`w/3` along the closing direction, the full jaw size across it. That region receives

| map | value inside | value outside |
|---|---|---|
| quality `Q` | 1 | 0 |
| `cos 2θ` | cos 2θ | 0 |
| `sin 2θ` | sin 2θ | 0 |
| width `W` | min(w, 150)/150 | 0 |

<!-- src: src/grasp_planner/data/dataset.py -->
```python
    for g in grasps:
        centre_region = Grasp(g.x, g.y, g.theta, g.width / 3.0, g.jaw_size)
        poly = np.round(centre_region.polygon()).astype(np.int32)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        m = mask.astype(bool)
        if not m.any():
            continue
        out[0][m] = 1.0
        out[1][m] = np.cos(2 * g.theta)
        out[2][m] = np.sin(2 * g.theta)
        out[3][m] = min(g.width, max_width) / max_width
```

Only the centre third is labelled because a grasp is only valid when the plates are roughly
centred on the object. Pixels near the ends of the rectangle would be bad grasp **centres**.
Overlapping rectangles overwrite each other in file order (as in GG-CNN). In the worked example
the target width is 43.68 / 150 = 0.291.

![Training targets](figures/training_targets.png)
*Figure 3.3 — Generated by `scripts/visualize_targets.py`. Top three rows: training scenes
without augmentation. Bottom three: the same scenes with a random augmentation. Columns: RGB
with transformed GT rectangles, normalised depth channel, quality target, angle target, width
target. The rotated remote (row 4) keeps targets aligned with the object, and its angle colour
changes by the rotation angle.*

### 3.6 Augmentation and its mathematics

Training samples draw, independently per scene and epoch: rotation `α ~ U(−180°, 180°)`, zoom
`z ~ U(0.85, 1)` (the crop shrinks, so objects look up to 1/0.85 ≈ 18 % larger), shift
`~ U(−20, 20)` px per axis, and a mirror with probability 0.5. The generator is seeded with
`(seed, epoch, index)`, so augmentation is reproducible regardless of how many DataLoader
workers are used. Pixels that come from outside the 640 × 480 image (possible after rotation)
are filled by border replication. Augmentation is applied **only** to the training split:
`make_dataset(..., train=False)` gives `augment=None`, and `evaluate_dataset` refuses an
augmented dataset.

### 3.7 Splits and leakage prevention

Splits are created **before** anything is learned. They are stored as JSON with seed, ratios,
parameters and creation time (`experiments/splits/`):

| Split file | Strategy | train | val | test | dropped |
|---|---|---|---|---|---|
| `image_wise_seed42.json` | random scenes, 70/15/15 | 620 | 133 | 132 | 0 |
| `block_wise_seed42.json` | contiguous blocks of 20 scene IDs, 2-scene gaps at split boundaries | 575 | 122 | 116 | 72 |

*Image-wise* splits are what most published Cornell numbers use. They test generalisation to new
views of objects that may also appear in training. *Object-wise* splits keep each object in one
split and test generalisation to unseen objects. Building one requires an object ID per scene
(Cornell's `z.txt`), and the local copy lacks it. Inspection showed that consecutive IDs
**usually but not always** show the same object (scene 0100 is a remote; 0103–0105 are a
stapler). The *block* split therefore keeps runs of consecutive IDs together and drops two
scenes on each side of a boundary between different splits. This **reduces** object overlap
between train and test but cannot **guarantee** zero overlap, because an object may be
re-photographed far away in the ID sequence. **Results on the block split are therefore reported
as a harder, approximately object-disjoint split, not as object-wise generalisation.**

`check_disjoint` runs whenever a split is created or loaded. `tests/test_splits.py` checks
completeness, reproducibility, disjointness and the gap rule.

### 3.8 Dataset limitations

* **Size.** 885 scenes. Validation and test have 116–133 images each, so one image ≈ 0.75–0.86
  accuracy points. Differences of 1–2 points between models are within noise (§9.6).
* **Incomplete labels.** Annotators marked a handful of grasps per object. Many plausible grasps
  are unlabelled, so the metric under-counts correct predictions (examples in §9.5).
* **Single object, fixed background.** One object on a white board, one camera height, one lab.
  Nothing here tests clutter, other backgrounds or other cameras.

---

## 4. Mathematical Foundations

This chapter covers only the mathematics the code uses. Notation is collected in Appendix B.

### 4.1 Convolution, stride, padding and receptive fields

A 2-D convolutional layer with `C_in` input channels, `C_out` output channels and a `k × k`
kernel computes, for output channel `o` and position `(i, j)`,

$$y_{o,i,j} = b_o + \sum_{c=1}^{C_{in}}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} W_{o,c,u,v}\; x_{c,\;s i + u - p,\; s j + v - p},$$

where `s` is the **stride** and `p` the zero **padding**. The output size along one axis is

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1.$$

*Worked example (GR-ConvNet `conv2`):* `H_in = 224, k = 4, s = 2, p = 1` → `⌊(224+2−4)/2⌋+1 = 112`.
The layer has `C_in·C_out·k² + C_out = 32·64·16 + 64 = 32,832` parameters.

A **transposed convolution** ("deconvolution") is the gradient of a convolution with respect to
its input. It is used to upsample:

$$H_{out} = (H_{in}-1)\,s - 2p + k + p_{out},$$

where `p_out` is the extra output padding. GR-ConvNet `conv4`: `(56−1)·2 − 2 + 4 + 1 = 113`.
`conv5`: `(113−1)·2 − 4 + 4 + 1 = 225`.

The **receptive field** of an output pixel is the region of the input that can influence it.
Each layer adds `(k−1)·∏(earlier strides)` pixels. A large receptive field matters here because
deciding whether a pixel is a good grasp centre requires seeing the **whole object** around it
For GR-ConvNet the theoretical receptive field is about 113 × 113 input pixels:
9 (conv1) → 12 → 18 (two stride-2 convs, jump 4) → 98 (ten 3 × 3 convs in the residual blocks at
jump 4) → about 113 after the three transposed convolutions and the 2 × 2 head. That is half the
224-px input, or about 145 px of the original image, which is larger than most Cornell objects.
GG-CNN's three convolutions alone already reach 9 + 4·3 + 2·6 = 33 px (jumps 1, 3, 6). Its
decoder adds a similar amount, but it has far fewer non-linear stages.

A **feature map** is one output channel viewed as an image: a spatial map of where a learned
pattern is present.

### 4.2 Activation functions

Without non-linearities a stack of convolutions collapses into one linear map. The models use
the **ReLU**, `ReLU(z) = max(0, z)`, whose derivative is 1 for `z > 0` and 0 otherwise. It is
cheap to compute, and gradients do not vanish on the positive side. The four output heads have
**no** activation: they regress real values directly, as in GR-ConvNet. A sigmoid on `Q` would
be possible, but it would change the loss geometry compared with the reference implementation.

### 4.3 Batch normalisation

For each channel, BatchNorm [Ioffe 2015] standardises activations over the batch and the spatial
positions, then applies a learned scale and shift:

$$\hat z = \frac{z-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y = \gamma \hat z + \beta.$$

At evaluation time `μ_B, σ_B` are replaced by running averages collected during training. This
is why the code calls `model.eval()` before validation.

### 4.4 Encoder–decoder networks and residual connections

An **encoder** reduces spatial resolution while increasing channels, so later layers see
larger context. A **decoder** restores resolution so that every input pixel gets an output. A
**residual block** [He 2016] learns a correction to its input rather than a full mapping:

$$\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x}),\qquad \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = I + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}.$$

The identity term passes gradients straight through, so deep stacks train reliably. If a block
is not useful, it can learn `F ≈ 0`.

<!-- src: src/grasp_planner/models/grconvnet.py -->
```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return x + y
```

### 4.5 Dense regression and the loss

Let `ŷ_h` be the prediction for head `h ∈ {Q, cos, sin, W}` and `y_h` its target, over all
`N = B·H·W` pixels of a batch. The loss is

$$\mathcal{L} = \sum_{h}\frac{1}{N}\sum_{n=1}^{N}\ell_\beta\!\left(\hat y_{h,n}-y_{h,n}\right),\qquad
\ell_\beta(r) = \begin{cases}\tfrac{1}{2}r^2/\beta & |r|<\beta\\ |r|-\tfrac{1}{2}\beta & \text{otherwise}\end{cases},\quad \beta=1.$$

This is the **Smooth-L1** (Huber) loss [Girshick 2015]. Its gradient is
`∂ℓ/∂r = r` for `|r| < 1` and `sign(r)` otherwise. It behaves like MSE for small errors and is
bounded for large ones, so a few badly mislabelled pixels cannot dominate an update. *Worked
example* (`tests/test_targets_and_models.py::test_smooth_l1_hand_value`): on a 2 × 2 map, one
pixel off by 0.5 contributes `0.5·0.5² = 0.125`, averaged over 4 pixels = 0.03125. One pixel off
by 3 contributes `3 − 0.5 = 2.5` → 0.625.

Because about 98 % of target pixels are zero (background), the loss is dominated by teaching
the network to output 0 away from grasps. This is intended for `Q`. For the angle and width
maps it means those heads are only constrained where `Q` is high, which is where they are
read.

<!-- src: src/grasp_planner/training/losses.py -->
```python
    for i, name in enumerate(HEAD_NAMES):
        loss = F.smooth_l1_loss(pred[name].float(), target[:, i : i + 1].float())
        parts[name] = float(loss.detach())
        total = total + loss
```

### 4.6 Angle representation and periodicity

Regressing `θ` directly fails at the wrap-around: a grasp at +89° and one at −89° are 2° apart,
yet their numeric values differ by 178°. Encoding the **doubled** angle on the unit circle fixes
this:

$$\theta \mapsto (\cos 2\theta,\ \sin 2\theta),\qquad \hat\theta = \tfrac12\,\mathrm{atan2}(\sin 2\theta, \cos 2\theta).$$

Doubling maps the grasp period π onto the circle's period 2π: `θ` and `θ + π` give the same
code, and nearby orientations give nearby codes. `atan2` returns a value in `(−π, π]`, so the
decoded `θ` lies in `(−π/2, π/2]`. For the scene-0100 grasp, `θ = −4.39°` → `(0.988, −0.153)`.
The network outputs both components independently, so their norm is not exactly 1. `atan2`
only uses their ratio, so the norm does not matter.

### 4.7 Optimisation, learning-rate schedule and regularisation

**Adam** [Kingma 2015] keeps exponential moving averages of the gradient `g` and its square:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad
\theta_t = \theta_{t-1} - \eta_t\,\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon},$$

with `β1 = 0.9, β2 = 0.999` (PyTorch defaults). Each parameter gets its own step size, so the
default learning rate 10⁻³ works without tuning.

**Cosine annealing** [Loshchilov 2017], without restarts, decays the learning rate over `T` epochs:

$$\eta_t = \eta_{min} + \tfrac12(\eta_{max}-\eta_{min})\left(1+\cos\frac{\pi t}{T}\right),$$

from 10⁻³ to 10⁻⁵ over 100 epochs (the `lr` column of each `history.csv`). Large early steps
explore; small late steps settle into a minimum.

**Regularisation** in this project: data augmentation (the strongest one for 620 images),
dropout with p = 0.1 before GR-ConvNet's heads [Srivastava 2014], **early stopping**
(patience 30 epochs), and best-checkpoint selection on validation data. **Gradient clipping**
rescales the gradient to norm ≤ 5 if it is larger, which guards against rare spikes. **Mixed
precision** [Micikevicius 2018] runs the forward pass in float16 on the GPU, with a dynamic
loss scale (`GradScaler`) so that small gradients do not underflow. The loss itself is computed
in float32 (`.float()` above).

### 4.8 Rectangle geometry, IoU and the Cornell criterion

The Jaccard index (**IoU**) of two rectangles `P, G` is the area of their intersection over the
area of their union:

$$J(P,G) = \frac{|P\cap G|}{|P\cup G|} = \frac{|P\cap G|}{|P| + |G| - |P\cap G|}.$$

Both are convex polygons, so the intersection is also a convex polygon. The project computes
it exactly with Shapely (`polygon_iou`). *Hand-checked cases* (`tests/test_grasp_geometry.py`):
two 10 × 10 squares overlapping by half → 50 / 150 = 1/3. A square and the same square rotated
45° about its centre → the intersection is a regular octagon of area `2(1+√2)s²` with
`s = 10/(1+√2)`, i.e. 82.84, and IoU = 82.84 / (200 − 82.84) = 0.707.

A prediction **matches** a ground-truth rectangle under the Cornell criterion [Jiang 2011] when

$$J(P,G) > 0.25 \quad\text{and}\quad \Delta\theta(P,G) < 30^\circ,\qquad
\Delta\theta = \min\big(|\theta_P-\theta_G| \bmod \pi,\ \pi - (|\theta_P-\theta_G| \bmod \pi)\big).$$

*Worked example* (computed by the code): take the scene-0100 ground truth (`w = 56.2`,
`h = 26.1`). A prediction shifted 10 px along the closing direction, rotated by 12°, with
`w = 50, h = 25`, has IoU 0.592 and `Δθ = 12°`. It is therefore a match.

### 4.9 Candidate decoding as mathematics

Given the four maps, the grasp candidates are the local maxima of a smoothed quality map:

$$\tilde Q = G_\sigma * Q,\qquad \mathcal{P} = \{(i,j) : \tilde Q_{ij} = \max_{|u|,|v|\le r}\tilde Q_{i+u,j+v},\ \tilde Q_{ij}\ge \tau\},$$

with Gaussian `G_σ` (σ = 2 px), window radius `r = 5` and threshold `τ = 0.1`. At each peak,
`θ = ½ atan2(sin, cos)` and `w = 150·(G₁ * W)`, with the width map smoothed by σ = 1. The angle
is **not** smoothed, because averaging angles across the wrap-around would be wrong. Chapter 8
describes the full algorithm, including NMS.

---

## 5. Model Architecture

### 5.1 Why dense fully convolutional models

Two families exist for Cornell. **Two-stage / sliding-window** methods [Jiang 2011, Lenz 2015]
score many sampled rectangles one at a time. They are slow (seconds per image).
**Direct regression** methods [Redmon 2015] output one grasp per image. They are fast, but
averaging over many valid grasps tends to produce a grasp in the middle of two good ones.
**Dense generative** models [Morrison 2018, Kumra 2020] predict a grasp at every pixel in one
forward pass. They are fast (milliseconds), produce many candidates naturally, and fit this
hardware (RTX 4060 Laptop, 8 GB) and dataset size. The project implements two of them under
identical conditions:

* **GG-CNN** — the baseline: tiny (70 k parameters), shallow, no normalisation.
* **GR-ConvNet** — the main model: 1.9 M parameters, batch norm, residual bottleneck. It reports
  the higher Cornell accuracy of the two [Kumra 2020].

Considered and rejected: large backbones (ResNet-50/101 two-stage detectors, transformers).
Their reported Cornell accuracies are similar to GR-ConvNet's, but they cost 10–100× more
compute and overfit 620 training images more easily. A pretrained ImageNet backbone cannot
take the 4-channel RGB-D input without modification.

### 5.2 GR-ConvNet: layer by layer

```mermaid
flowchart LR
  X["x: 4×224×224"] --> C1["conv1 9×9 s1 p4<br/>32×224×224 · BN · ReLU"]
  C1 --> C2["conv2 4×4 s2 p1<br/>64×112×112 · BN · ReLU"]
  C2 --> C3["conv3 4×4 s2 p1<br/>128×56×56 · BN · ReLU"]
  C3 --> R1["res ×5<br/>128×56×56"]
  R1 --> D4["conv4ᵀ 4×4 s2 p1 op1<br/>64×113×113 · BN · ReLU"]
  D4 --> D5["conv5ᵀ 4×4 s2 p2 op1<br/>32×225×225 · BN · ReLU"]
  D5 --> D6["conv6ᵀ 9×9 s1 p4<br/>32×225×225"]
  D6 --> DO["dropout 0.1"]
  DO --> HQ["Q: conv 2×2 → 1×224×224"]
  DO --> HC["cos2θ: conv 2×2"]
  DO --> HS["sin2θ: conv 2×2"]
  DO --> HW["W: conv 2×2"]
```
*Figure 5.1 — GR-ConvNet as implemented in `models/grconvnet.py`. Shapes are for one 224 × 224
RGB-D image; each residual block is conv 3×3 → BN → ReLU → conv 3×3 → BN, plus the skip.*

<!-- src: src/grasp_planner/models/grconvnet.py -->
```python
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = x.shape[-2:]
        z = F.relu(self.bn1(self.conv1(x)))
        z = F.relu(self.bn2(self.conv2(z)))
        z = F.relu(self.bn3(self.conv3(z)))
        z = self.res(z)
        z = F.relu(self.bn4(self.conv4(z)))
        z = F.relu(self.bn5(self.conv5(z)))
        z = self.conv6(z)
        if z.shape[-2:] != (h + 1, w + 1):  # only for input sizes not divisible by 4
            z = F.interpolate(z, size=(h + 1, w + 1), mode="bilinear", align_corners=False)
        return self.heads(z)
```

The decoder deliberately produces 225 × 225. Each 2 × 2 head convolution (no padding) then
shrinks this by one pixel to exactly 224 × 224, aligned with the input. Inputs whose side is not
a multiple of 4 would not land on 225, so the `interpolate` fallback exists for those. It is
never triggered at 224.

**Parameter count.** The training log prints `model parameters: 1900900` for the RGB-D model.
This equals the 1,900,900 reported by Kumra et al., which confirms the layer configuration. The
five residual blocks hold 5 × 2 × (128·128·9 + 128) + BN ≈ 1.48 M of these, which is where
most of the capacity sits.

### 5.3 GG-CNN baseline

GG-CNN [Morrison 2018] has three strided convolutions (9×9/3 → 32 ch, 5×5/2 → 16, 3×3/2 → 8),
three mirrored transposed convolutions, and 2 × 2 heads. There is no batch norm and no residual
path. The original paper uses 300 × 300 depth input. Here the input channel count is
configurable. Because its strides (3, 2, 2) do not divide 224 evenly, the decoder output is
bilinearly resized to 225 × 225 before the heads. It has **70,196** parameters with 4 input
channels (`model parameters` line in `outputs/logs/ggcnn_rgbd_image.log`), 27× fewer than
GR-ConvNet. It is useful as a baseline because it isolates the effect of model capacity: data,
loss, optimiser, augmentation and decoding are all identical.

### 5.4 Output heads

<!-- src: src/grasp_planner/models/heads.py -->
```python
    def __init__(self, in_channels: int, kernel_size: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.convs = nn.ModuleDict({n: nn.Conv2d(in_channels, 1, kernel_size) for n in HEAD_NAMES})
```

Four independent 1-channel convolutions share the last feature map. Keeping them separate (rather
than one 4-channel conv) mirrors the reference implementation and makes per-head losses easy to
log (`train_quality`, `train_cos2`, … in `history.csv`).

---

## 6. Source Code Walkthrough

The package lives in `src/grasp_planner/`. Scripts in `scripts/` are thin command-line wrappers
around it. Each subsection gives purpose, inputs and outputs, key functions, interactions,
failure modes and how to extend the module.

### 6.1 `config.py` — configuration
* **Purpose:** load YAML experiment configs (with a `base:` inheritance chain) and find the
  dataset without hard-coded paths.
* **Key functions:** `load_config(path)` → dict, merging `configs/default.yaml` via
  `deep_update`; `resolve_dataset_root()` → `Path`, in priority order: explicit argument →
  `GRASP_DATASET_ROOT` → `configs/local.yaml`.
* **Failure modes:** missing or wrong path → `ConfigError`, with the exact setup instruction.
* **Extend:** add keys to `default.yaml`; experiment files only override what changes.

<!-- src: src/grasp_planner/config.py -->
```python
    candidate: str | Path | None = explicit or os.environ.get(ENV_VAR)
    if not candidate and local_config.exists():
        with local_config.open("r", encoding="utf-8") as f:
            candidate = (yaml.safe_load(f) or {}).get("dataset_root")
```

### 6.2 `data/cornell.py` — discovery and parsing
* **Purpose:** find scenes, parse `cpos`/`cneg` files, validate, report.
* **Inputs/outputs:** dataset root → `list[CornellSample]` (paths + `(N,4,2)` rectangle arrays +
  `issues`) and a list of excluded scenes with reasons.
* **Key functions:** `parse_rectangle_file`, `discover_samples`, `load_rgb`, `load_depth`,
  `mask_invalid_depth`.
* **Decisions:** never write to the dataset; reject individual rectangles instead of whole
  scenes; log every rejection.
* **Extend:** another dataset (e.g. Jacquard) needs only a new discovery function that returns
  `CornellSample`-like records. Everything downstream works on rectangles.

### 6.3 `data/grasp.py` — geometry
`Grasp` (frozen dataclass: `x, y, theta, width, jaw_size, score`) with `direction`,
`contact_points`, `polygon`, `iou`. Also `rect_to_grasp`, `polygon_iou`, `angle_difference`,
`wrap_angle`, `grasp_matches` (the Cornell criterion) and `rectangle_geometry_errors`. Every
other module uses these functions, so the conventions are defined in exactly one place.

<!-- src: src/grasp_planner/data/grasp.py -->
```python
def grasp_matches(pred: Grasp, gt: Grasp, iou_thresh: float = 0.25, angle_thresh_deg: float = 30.0) -> bool:
    """Cornell rectangle metric: IoU > ``iou_thresh`` and angle difference < ``angle_thresh_deg``."""
    if angle_difference(pred.theta, gt.theta) >= np.deg2rad(angle_thresh_deg):
        return False
    return pred.iou(gt) > iou_thresh
```

The cheap angle test runs first, so the polygon intersection is only computed when it can matter.

### 6.4 `data/transforms.py` — affine preprocessing
`CropSpec`, `AugmentSpec`, `build_affine` (§3.4), `sample_affine` (random draw),
`apply_affine_points`, `invert_affine`, `warp_image` (`cv2.warpAffine` with border
replication) and `inpaint_depth`. *Failure mode:* a depth image with no valid pixel raises
`ValueError` rather than producing garbage. *Extend:* new augmentations go into `build_affine`
as one more matrix factor, and labels follow automatically.

### 6.5 `data/dataset.py` — the PyTorch dataset
`CornellGraspDataset.__getitem__` is the whole data path in ten lines:

<!-- src: src/grasp_planner/data/dataset.py -->
```python
        rng = np.random.default_rng([self.seed, self.epoch, i]) if self.augment else None
        M = sample_affine(self.spec.crop, self.spec.output_size, self.augment, rng)
        rgb, depth = self.load_raw(i)
        S = self.spec.output_size
        rgb_w = warp_image(rgb, M, S) if rgb is not None else None
        depth_w = warp_image(depth, M, S) if depth is not None else None
        x = build_input(rgb_w, depth_w, self.spec.modality)
        grasps = [rect_to_grasp(apply_affine_points(M, r)) for r in s.pos]
        y = draw_grasp_targets(grasps, S, self.spec.max_width)
```

It returns `x (C,224,224)`, `y (4,224,224)`, the matrix `M` (needed to map predictions back) and
the sample index. `DepthCache` stores inpainted depth under `outputs/cache/depth_inpainted/`. The
cache key includes the file path, size and modification time, so two dataset copies or test
fixtures never collide. `PreprocessSpec.from_config` is the single place where preprocessing
settings are read. Training, evaluation and the app all use it.

### 6.6 `data/splits.py`
`image_wise_split`, `block_split`, `make_split` (adds metadata), `check_disjoint`,
`save_split`/`load_split`. A split file always records how it was made.

### 6.7 `models/` — networks
`ggcnn.py`, `grconvnet.py`, `heads.py` and the factory `build_model(model_cfg, in_channels)`.
Every model returns a dict `{quality, cos2, sin2, width}` of `(B,1,H,W)` tensors.
*Extend:* add a class with the same output contract and one `if` in `build_model`.

### 6.8 `training/` — losses and the trainer
`losses.grasp_loss` (§4.5). `Trainer` owns the model, optimiser, scheduler, `GradScaler`,
DataLoader, checkpointing and history. One training step:

<!-- src: src/grasp_planner/training/trainer.py -->
```python
            with torch.autocast(self.device.type, enabled=self.amp):
                out = self.model(x)
            loss, parts = grasp_loss(out, y)
            if not torch.isfinite(loss):
                raise NonFiniteLossError(f"non-finite loss at epoch {epoch} batch {b}: {parts}")
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if clip > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.scaler.step(self.opt)
            self.scaler.update()
```

`unscale_` must come before clipping: otherwise the gradient norm would be measured on the
loss-scaled gradients. After every epoch the trainer validates, applies the selection rule, and
writes `last.pt` (always) and `best.pt` (on improvement) through a temporary file and an atomic
rename. A crash therefore never leaves a half-written checkpoint. `resume(path)` restores model,
optimiser, scheduler, scaler, the best-so-far record and the RNG state.

<!-- src: src/grasp_planner/training/trainer.py -->
```python
def _better(score: tuple[float, float], best: tuple[float, float] | None) -> bool:
    """score = (val_top1, val_loss): higher accuracy wins, then lower loss."""
    if best is None:
        return True
    return score[0] > best[0] or (score[0] == best[0] and score[1] < best[1])
```

### 6.9 `evaluation/` — metrics and runner
`metrics.evaluate_image` scores one image (top-1/top-k, errors). `metrics.summarise`
aggregates. `runner.evaluate_dataset` runs the model over a split (refusing augmented data),
decodes, maps candidates back to original pixels and scores them. `runner.measure_latency`
times the forward pass and the decode. Chapter 9 and `docs/evaluation_protocol.md` give the
definitions.

### 6.10 `inference/` — decoding and the predictor
`decode.py` (Chapter 8) and `predictor.GraspPredictor`. The predictor loads a checkpoint,
rebuilds the model **from the config stored inside the checkpoint** (so preprocessing can never
silently differ from training), preprocesses one image, runs the model and decodes. For images
that are not 640 × 480, the crop window is scaled proportionally (`crop_for_image`). Missing
inputs raise clear errors: an RGB-D model without depth, a missing checkpoint, or mismatched
RGB/depth sizes.

### 6.11 `visualization/plots.py`
`render_prediction` draws GT (green), candidates 2…k (amber, with `rank:score`), the selected
grasp (blue; plate edges thicker than the opening edges) and its contact points (magenta).
`plot_history` draws loss and accuracy curves. `save_maps_figure` shows the raw output maps.

### 6.12 `pipeline.py` and the scripts
`pipeline.py` glues config → samples per split → datasets → model → checkpoint. The scripts:
`inspect_dataset.py`, `make_splits.py`, `visualize_targets.py`, `train.py`,
`run_experiments.py` (sequential background queue), `evaluate.py`, `summarize_results.py`,
`make_report_figures.py`, `predict.py`, `check_report.py`.

## 7. Training Methodology

### 7.1 Pipeline

```mermaid
flowchart LR
  S["split file<br/>(train ids)"] --> DS["CornellGraspDataset<br/>augment=on"]
  DS --> DL["DataLoader<br/>batch 8, shuffle, 2 workers"]
  DL --> M["model (AMP fp16)"]
  M --> L["Smooth-L1 × 4"]
  L --> O["Adam + GradScaler<br/>clip 5"]
  O --> M
  M -. every epoch .-> V["validate: decode + rectangle metric<br/>on val split (no augment)"]
  V --> C{"val top-1 better?"}
  C -->|yes| B["save best.pt"]
  C --> LA["save last.pt, append history.csv"]
  LA --> E{"30 epochs w/o improvement?"}
  E -->|yes| STOP["stop"]
```

### 7.2 Environment

| Item | Value (from `outputs/runs/*/env.json`) |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Python | 3.12.7 (project `.venv`) |
| PyTorch | 2.11.0+cu128, CUDA runtime 12.8 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB (driver 592.82) |
| CPU / RAM | AMD Ryzen 7 7840HS (8 cores / 16 threads), 15.3 GB |

PyTorch was installed from the official CUDA index (`--index-url https://download.pytorch.org/whl/cu128`),
because a plain `pip install torch` on Windows gives the CPU-only build. `torch.cuda.is_available()`
and a GPU matrix multiplication were checked before training.

### 7.3 Hyperparameters and rationale

<!-- src: configs/default.yaml -->
```yaml
train:
  epochs: 100
  batch_size: 8
  val_batch_size: 8
  optimizer: adam
  lr: 0.001
  weight_decay: 0.0
  scheduler: cosine         # cosine | step | none
  min_lr: 1.0e-5
  amp: true                 # mixed precision on CUDA
  grad_clip: 5.0
  early_stopping_patience: 30   # epochs without val top-1 improvement (0 = off)
```

| Setting | Value | Why |
|---|---|---|
| input | 224 × 224, RGB-D | GR-ConvNet's published setting; a batch-8 AMP training step peaks at ≈ 448 MB of allocated GPU memory (GG-CNN ≈ 235 MB; measured with `torch.cuda.max_memory_allocated`), far below 8 GB |
| batch size | 8 | GR-ConvNet's value; 77 updates per epoch on 620 images |
| optimiser, lr | Adam, 10⁻³ | GR-ConvNet's value; Adam needs little tuning (§4.7) |
| schedule | cosine to 10⁻⁵ over 100 epochs | smooth decay; no step-size hyperparameters |
| epochs / patience | ≤ 100 / 30 | full run ≈ 10 min here, so a long budget is cheap; patience stops runs that have plateaued |
| augmentation | rot ±180°, zoom 0.85–1, shift ±20 px, flip 0.5 | the dataset is small; Cornell objects can appear in any orientation |
| AMP, clip | fp16 autocast, ‖g‖ ≤ 5 | ≈ 6 s/epoch; clipping as a safety net (no NaN occurred) |
| workers | 2 | Windows spawns worker processes; 2 workers ran without problems, and `num_workers: 0` is the documented fallback |

These values were **fixed before training** and not tuned. Every configuration was trained once.
Each experiment config changes only `experiment.name`, `data.modality`, `data.split_file` and
`model.name`:

<!-- src: configs/experiment_configs/grconvnet_rgbd_image.yaml -->
```yaml
base: configs/default.yaml
experiment:
  name: grconvnet_rgbd_image
data:
  modality: rgbd
  split_file: experiments/splits/image_wise_seed42.json
model:
  name: grconvnet
```

### 7.4 Validation, checkpoint selection, resume and seeds

* After every epoch the **whole validation split** is decoded and scored with the same rectangle
  metric used for testing. `best.pt` is the epoch with the highest validation top-1 accuracy
  (ties: lower validation loss). This criterion was fixed in the code before any run.
* `last.pt` is written every epoch. `--resume` continues from it, restoring
  optimiser/scheduler/scaler/RNG state and the best-so-far record. `tests/test_integration.py`
  checks that a resumed run starts at the next epoch.
* Seed 42 for Python, NumPy and PyTorch. Augmentation RNG is seeded per (seed, epoch, index).
  cuDNN autotuning and non-deterministic GPU kernels mean that bit-exact reruns are **not**
  guaranteed. Re-running a configuration gives a similar, not identical, curve.

### 7.5 Smoke tests versus real training

| Kind | What | Status |
|---|---|---|
| Smoke test | `--smoke`: 2 epochs × 3 batches, 16 val images, separate `*_smoke` run dirs (deleted afterwards) | passed for GR-ConvNet RGB-D, GG-CNN RGB-D, GR-ConvNet RGB. Accuracies meaningless, never reported |
| Real training | full configs below, background queue `scripts/run_experiments.py` | all five finished. Each `train.log` ends with `TRAINING FINISHED` |

| Run | Epochs run | Best epoch | Stopped early | Wall time |
|---|---|---|---|---|
| grconvnet_rgbd_image | 100 | <!-- num: experiments/results/grconvnet_rgbd_image__test.json:checkpoint_epoch -->85 | no | 631 s |
| ggcnn_rgbd_image | 75 | <!-- num: experiments/results/ggcnn_rgbd_image__test.json:checkpoint_epoch -->44 | yes | 406 s |
| grconvnet_rgbd_block | 49 | <!-- num: experiments/results/grconvnet_rgbd_block__test.json:checkpoint_epoch -->18 | yes | 319 s |
| ggcnn_rgbd_block | 50 | <!-- num: experiments/results/ggcnn_rgbd_block__test.json:checkpoint_epoch -->19 | yes | 224 s |
| grconvnet_rgb_image | 80 | <!-- num: experiments/results/grconvnet_rgb_image__test.json:checkpoint_epoch -->49 | yes | 466 s |

Epoch indices start at 0. Wall times come from `outputs/logs/run_experiments.log` and include
start-up and per-epoch validation. The full queue of five runs took about 34 minutes.

## 8. Grasp Candidate Generation and Selection

The network outputs four 224 × 224 maps. This chapter turns them into a ranked list of discrete
grasps in original-image pixels. All of it lives in `src/grasp_planner/inference/decode.py`, and
the parameters are in the `decode:` section of `configs/default.yaml`.

### 8.1 Post-processing the maps

<!-- src: src/grasp_planner/inference/decode.py -->
```python
    q = gaussian_filter(quality.astype(np.float64), spec.q_sigma)
    ang = 0.5 * np.arctan2(sin2, cos2)
    w = gaussian_filter(width.astype(np.float64) * spec.max_width, 1.0)
    return q, ang, w
```

* Smoothing `Q` with σ = 2 px suppresses isolated single-pixel spikes. A good grasp centre should
  be a *region* of high quality, which also makes the location robust.
* The angle is decoded from the doubled-angle pair (§4.6) and deliberately **not** smoothed.
* Width is de-normalised (× 150) and lightly smoothed (σ = 1).

### 8.2 Peak extraction

<!-- src: src/grasp_planner/inference/decode.py -->
```python
    size = 2 * spec.min_distance + 1
    is_peak = (q == maximum_filter(q, size=size, mode="nearest")) & (q >= spec.q_threshold)
    rows, cols = np.nonzero(is_peak)
    order = np.argsort(-q[rows, cols], kind="stable")
    return [(int(rows[i]), int(cols[i])) for i in order[: spec.max_peaks]]
```

A pixel is a peak if it equals the maximum of its 11 × 11 neighbourhood and its quality is
≥ 0.1. At most 50 peaks are kept, highest first. Each peak becomes a candidate:
`Grasp(x=col, y=row, θ=ang[row,col], w=w[row,col], h=w/2, score=Q̃[row,col])`. The candidate's
centre lies at the pixel centre, so the localisation resolution is one network pixel
(1 / 0.778 ≈ 1.3 original pixels).

### 8.3 Validity checks

<!-- src: src/grasp_planner/inference/decode.py -->
```python
    return (
        np.isfinite([g.x, g.y, g.theta, g.width, g.score]).all()
        and spec.min_width <= g.width <= spec.max_width
        and 0 <= g.x <= w - 1
        and 0 <= g.y <= h - 1
    )
```

A candidate is discarded if any value is non-finite, if its opening is outside [2, 150] network
pixels (a near-zero or larger-than-trained opening is not a usable grasp), or if its centre is
outside the image. These checks are purely geometric. They do not check collisions or physical
feasibility.

### 8.4 Angle-aware non-maximum suppression

Neighbouring peaks often describe almost the same grasp. NMS walks the candidates in order of
decreasing score and drops any candidate that **both** overlaps an already accepted one with
IoU > 0.3 **and** differs from it in angle by less than 30°:

<!-- src: src/grasp_planner/inference/decode.py -->
```python
    for g in cands:  # already sorted by score
        dup = any(
            angle_difference(g.theta, k.theta) < np.deg2rad(spec.nms_angle_deg) and g.iou(k) > spec.nms_iou
            for k in keep
        )
        if not dup:
            keep.append(g)
        if len(keep) >= spec.max_candidates:
            break
```

Requiring both conditions keeps genuinely different alternatives, for example a perpendicular
grasp at the same location (`tests/test_decode_and_metrics.py::test_nms_removes_duplicates_but_keeps_other_orientations`).
At most 10 candidates survive.

### 8.5 Ranking, selection and back-projection

The survivors are already ordered by smoothed quality. The **selected grasp is the first one**.
Each candidate is mapped back to the original 640 × 480 image by transforming its rectangle's
vertices with `M⁻¹` and recomputing `(x, y, θ, w)`, exactly mirroring §3.4. The contact points
then follow from §2.3:

<!-- src: src/grasp_planner/inference/decode.py -->
```python
    pts = apply_affine_points(invert_affine(M), g.polygon())
    o = rect_to_grasp(pts)
    return Grasp(o.x, o.y, o.theta, o.width, o.jaw_size, g.score)
```

**What "best" means.** The selected grasp is the candidate with the highest *network quality
score* after the filters above. That score is a regression output trained to be 1 near labelled
grasp centres. It is **not a calibrated probability of success**, and the top candidate is not
"globally optimal" in any physical sense. It is the best candidate *according to the model and
this scoring procedure*.

### 8.6 A worked decoding example

![Decoding pipeline](figures/decode_pipeline_grconvnet_rgbd_image.png)
*Figure 8.1 — Final model on test scene pcd0101 (a stick-shaped object), generated by
`scripts/make_report_figures.py`. (a) Network input crop with the five GT rectangles. (b) Raw
quality map: high values along the object's long axis, where grasps across it are possible.
(c) Smoothed map with its two local maxima ≥ 0.1. (d) Both candidates pass the validity checks
and survive NMS, because they are about 55 px apart and do not overlap. (e) Back-projected to
the original image.*

Why does rank 1 beat rank 2? Both are grasps across the stick (θ ≈ 85° and 89°). Rank 1's
smoothed quality is 1.122 versus 1.014. The network is somewhat more confident near the
object's right half, and nothing else enters the ranking. Both happen to match labelled grasps (best IoU 0.77 and 0.43; the stick
is labelled along its length). Had rank 1 been wrong, top-1 would fail while
top-5 would succeed. That is exactly the pattern seen in 9 of the 15 test failures (§9.5).


## 9. Evaluation and Results

### 9.1 Protocol

Full definitions are in `docs/evaluation_protocol.md`. In short: deterministic preprocessing →
decode up to 10 ranked candidates → map them to original pixels → compare with **all** valid
positive GT rectangles of the image. **Top-1 accuracy** is the fraction of images whose
top-ranked candidate satisfies `IoU > 0.25 and Δθ < 30°` against at least one GT rectangle. It
is aggregated **per image** (mean of a 0/1 outcome), and an image with no candidate counts as a
failure. **Top-5** is the same for any of the first five candidates. Centre, angle and width
errors compare the top candidate with the GT rectangle whose centre is nearest, and are
averaged over images. Latency is the median of 100 single-image runs on the RTX 4060 Laptop
GPU, excluding file loading and preprocessing.

Order of operations: train → validate every epoch → select checkpoints and freeze the
selection (`docs/experiment_log.md`, committed before testing) → evaluate each selected
checkpoint **once** on its test split.

### 9.2 Training curves

![Image-wise curves](figures/loss_curves_image_wise.png)
*Figure 9.1 — Image-wise runs, from `history.csv`. Left: training (solid) and validation (dashed)
loss, log scale. Right: validation top-1 accuracy per epoch.*

![Block-wise curves](figures/loss_curves_block_wise.png)
*Figure 9.2 — Block-wise runs.*

How to read them:
* **Training loss above validation loss early on** is expected, not a bug. Training images are
  augmented (harder) and GR-ConvNet applies dropout in training mode only.
* **Validation accuracy plateaus by epoch ≈ 20–40** at roughly 85–90 % (image-wise). Further
  training keeps lowering the training loss. For GR-ConvNet RGB-D, the training loss falls
  below the validation loss after epoch ≈ 60 while the validation loss creeps up. That is
  **overfitting**: the network memorises the 620 training scenes. The cosine schedule and
  best-epoch selection limit the damage, and early stopping ended four of the five runs.
* **Validation accuracy is noisy** (± 3–5 points from epoch to epoch). With 133 images, one
  image is 0.75 points. Choosing the best of 100 noisy epochs therefore picks a lucky epoch,
  and the selected validation score is **optimistically biased** (see §9.4).
* On the **block split** the best epoch comes early (18–19) and validation loss rises soon
  after. Overfitting to seen objects starts earlier when validation objects are further from
  the training objects.

### 9.3 Main results (test split, evaluated once)

| Model | Input | Split | Test images | Top-1 | Top-5 | Centre err. (px, mean) | Angle err. (°, mean) | Width err. (px, mean) |
|---|---|---|---|---|---|---|---|---|
| **GR-ConvNet (final)** | RGB-D | image | 132 | **<!-- num: experiments/results/grconvnet_rgbd_image__test.json:top1_accuracy:.1% -->88.6%** | <!-- num: experiments/results/grconvnet_rgbd_image__test.json:top5_accuracy:.1% -->95.5% | <!-- num: experiments/results/grconvnet_rgbd_image__test.json:center_error_px.mean:.1f -->8.4 | <!-- num: experiments/results/grconvnet_rgbd_image__test.json:angle_error_deg.mean:.1f -->15.0 | <!-- num: experiments/results/grconvnet_rgbd_image__test.json:width_error_px.mean:.1f -->19.2 |
| GG-CNN | RGB-D | image | 132 | <!-- num: experiments/results/ggcnn_rgbd_image__test.json:top1_accuracy:.1% -->84.8% | <!-- num: experiments/results/ggcnn_rgbd_image__test.json:top5_accuracy:.1% -->93.2% | <!-- num: experiments/results/ggcnn_rgbd_image__test.json:center_error_px.mean:.1f -->8.7 | <!-- num: experiments/results/ggcnn_rgbd_image__test.json:angle_error_deg.mean:.1f -->16.8 | <!-- num: experiments/results/ggcnn_rgbd_image__test.json:width_error_px.mean:.1f -->14.0 |
| GR-ConvNet | RGB | image | 132 | <!-- num: experiments/results/grconvnet_rgb_image__test.json:top1_accuracy:.1% -->90.2% | <!-- num: experiments/results/grconvnet_rgb_image__test.json:top5_accuracy:.1% -->97.7% | <!-- num: experiments/results/grconvnet_rgb_image__test.json:center_error_px.mean:.1f -->8.1 | <!-- num: experiments/results/grconvnet_rgb_image__test.json:angle_error_deg.mean:.1f -->12.9 | <!-- num: experiments/results/grconvnet_rgb_image__test.json:width_error_px.mean:.1f -->20.0 |
| GR-ConvNet | RGB-D | block | 116 | <!-- num: experiments/results/grconvnet_rgbd_block__test.json:top1_accuracy:.1% -->75.9% | <!-- num: experiments/results/grconvnet_rgbd_block__test.json:top5_accuracy:.1% -->90.5% | <!-- num: experiments/results/grconvnet_rgbd_block__test.json:center_error_px.mean:.1f -->11.7 | <!-- num: experiments/results/grconvnet_rgbd_block__test.json:angle_error_deg.mean:.1f -->18.4 | <!-- num: experiments/results/grconvnet_rgbd_block__test.json:width_error_px.mean:.1f -->16.6 |
| GG-CNN | RGB-D | block | 116 | <!-- num: experiments/results/ggcnn_rgbd_block__test.json:top1_accuracy:.1% -->78.4% | <!-- num: experiments/results/ggcnn_rgbd_block__test.json:top5_accuracy:.1% -->91.4% | <!-- num: experiments/results/ggcnn_rgbd_block__test.json:center_error_px.mean:.1f -->9.9 | <!-- num: experiments/results/ggcnn_rgbd_block__test.json:angle_error_deg.mean:.1f -->17.2 | <!-- num: experiments/results/ggcnn_rgbd_block__test.json:width_error_px.mean:.1f -->14.8 |

Every image of every split received at least one candidate. Medians, best-IoU and the
validation numbers are in `experiments/results/*.json`. The generated table
`experiments/results/model_comparison.md` lists validation and test side by side. Each row's
config is `outputs/runs/<run>/config.yaml`. The checkpoint path, SHA-256 prefix (the final
model's is `23fda02a829da78c`) and epoch are recorded in each result JSON.

### 9.4 Uncertainty and model comparison

Each accuracy is a proportion over 116–133 images. 95 % Wilson intervals
(`experiments/results/statistics.json`):

| Run | Val top-1 [95 % CI] | Test top-1 [95 % CI] |
|---|---|---|
| GR-ConvNet RGB-D image | 93.2 % [87.6, 96.4] | 88.6 % [82.1, 93.0] |
| GG-CNN RGB-D image | 88.0 % [81.4, 92.5] | 84.8 % [77.8, 90.0] |
| GR-ConvNet RGB image | 90.2 % [84.0, 94.2] | 90.2 % [83.9, 94.2] |
| GR-ConvNet RGB-D block | 80.3 % [72.4, 86.4] | 75.9 % [67.3, 82.7] |
| GG-CNN RGB-D block | 73.8 % [65.3, 80.8] | 78.4 % [70.1, 85.0] |

Two models evaluated on the same images are compared with an **exact McNemar test**, which looks
only at images where exactly one model succeeds:

| Split | A vs B | only A correct | only B correct | p |
|---|---|---|---|---|
| image, test | GR-ConvNet RGB-D vs GG-CNN RGB-D | 11 | 6 | 0.332 |
| block, test | GR-ConvNet RGB-D vs GG-CNN RGB-D | 8 | 11 | 0.648 |
| image, test | GR-ConvNet RGB-D vs GR-ConvNet RGB | 7 | 9 | 0.804 |
| image, val | GR-ConvNet RGB-D vs GG-CNN RGB-D | 12 | 5 | 0.143 |

**What this supports.** GR-ConvNet RGB-D had the best validation accuracy on both splits and was
selected. On the image-wise test split it scored 3.8 points above GG-CNN. That difference is
**not statistically significant** (p = 0.33), and on the block split the order reversed (also
not significant). The RGB-only model scored slightly higher than RGB-D on test, which is also
not significant. **The data cannot distinguish these models.** A much larger test set, or
repeated cross-validation with several seeds, would be needed. The GR-ConvNet paper's large
advantage over GG-CNN (97.7 % vs 73.0 %) was not reproduced here. A likely reason is that our
GG-CNN baseline uses the same strong training recipe as GR-ConvNet: RGB-D input, ±180° rotation
augmentation and the same decoding. The originally reported GG-CNN number used depth only and
a different protocol.

**Image-wise vs block-wise.** All models lose 6–13 points on the block split (test). The block
split keeps runs of consecutive scenes, which usually means the same object, out of training.
So this drop is a measure of how much image-wise accuracy depends on having seen the same or
similar objects in training. It is the most practically relevant number in this study. It is
**not** a validated object-wise result (§3.7).

**Generalisation gap.** Validation → test changes from −4.6 to +4.6 points. Selection bias
(best epoch of 100 on a small validation set) explains the downward shifts, and sampling noise
explains the rest.

### 9.5 Qualitative results and failure analysis

![Qualitative test results](figures/qualitative_grconvnet_rgbd_image_test.png)
*Figure 9.3 — Final model on random test-split successes (top) and failures (bottom), from
`scripts/make_report_figures.py`. Green: GT positives. Blue: selected grasp, with plate edges
drawn thicker. Amber: ranks 2–3. Magenta: contact points. Titles give the best IoU with any GT
rectangle and the angle error to the nearest-centre GT rectangle.*

The final model's 15 test failures (`grconvnet_rgbd_image__test_per_image.csv`) break down as:

| Failure type | Count | Example | Comment |
|---|---|---|---|
| wrong location: centre > 20 px from every GT centre | 7 | pcd0741 (centre error 52 px, IoU 0.04) | often a plausible grasp on an unlabelled part of the object |
| right place, wrong orientation (Δθ ≥ 30°) | 5 | pcd0802 (Δθ = 73°, IoU 0.32) | e.g. perpendicular grasps on roundish objects |
| right place and angle, IoU just ≤ 0.25 | 3 | pcd0105 (IoU 0.25, failing the strict `>`), pcd0401 (0.24), pcd0604 (0.25) | near misses; in pcd0105 the opening is 59 px wider than the nearest label |

In 9 of the 15 failures **one of the ranks 2–5 does match** a GT rectangle (top-5 accuracy
95.5 %). The model often proposes a labelled grasp, but not first. The median quality score is
the same for successes and failures (1.08 vs 1.08). The score orders candidates *within* an image
but does **not** indicate whether the selected grasp will match a label, let alone succeed
physically. Scores above 1 occur because the quality head has no output activation (§4.2).

Validation failures (Chapter 6 tooling, `experiments/figures/grconvnet_rgbd_image/val/`, not
committed because they are full-size dataset images) show the same patterns. In pcd0890 the
model chose the brim of a cap, a reasonable grasp that the annotators did not label. In pcd0152
the opening was wider than any label.

### 9.6 Inference examples on images not used for training

**Test-split scene (`scripts/predict.py`).** Running the final model on test scene pcd0101 (RGB +
depth) printed two candidates (`outputs/logs/predict_pcd0101.txt`):

| rank | centre (px) | θ | width (px) | score | contact points (px) |
|---|---|---|---|---|---|
| 1 | (305.93, 297.50) | 84.92° | 66.94 | 1.1224 | (302.97, 330.84), (308.89, 264.16) |
| 2 | (250.64, 292.36) | 88.65° | 56.48 | 1.0143 | (249.98, 320.59), (251.31, 264.12) |

The top candidate matches a GT rectangle (best IoU 0.77, angle error 11°). Figure 8.1 shows the
same scene step by step.

**External-style images (background only).** Cornell's `backgrounds/` folder contains
object-free images that were never used for training or evaluation. Run through the RGB-only
model, none of the four shown produced a candidate above the 0.1 quality threshold
(Figure 9.4). This is the desired behaviour: the model does not hallucinate grasps on an empty
table. Real external photos (other cameras, cluttered backgrounds) were **not** available, so
behaviour under domain shift is unmeasured.

![Background images](figures/background_grconvnet_rgb_image.png)
*Figure 9.4 — RGB-only model on background-only images (crop window shown): no candidates.*

### 9.7 Runtime

| Model | Forward pass (median) | Forward + decode (median) |
|---|---|---|
| GR-ConvNet RGB-D | <!-- num: experiments/results/grconvnet_rgbd_image__test.json:latency.forward_ms_median:.1f -->3.7 ms | <!-- num: experiments/results/grconvnet_rgbd_image__test.json:latency.forward_plus_decode_ms_median:.1f -->5.8 ms |
| GG-CNN RGB-D | <!-- num: experiments/results/ggcnn_rgbd_image__test.json:latency.forward_ms_median:.1f -->1.2 ms | <!-- num: experiments/results/ggcnn_rgbd_image__test.json:latency.forward_plus_decode_ms_median:.1f -->3.1 ms |

Measured on the RTX 4060 Laptop GPU with batch size 1, fp32 weights, no AMP at inference, and
the GPU otherwise idle (test evaluations were run after all training had finished). Decoding
(Gaussian filter, peak finding, Shapely NMS) runs on the CPU and costs about 2 ms. Image loading,
depth inpainting and warping are **excluded** and were not timed. Training took about 6 s per epoch for GR-ConvNet.

### 9.8 Comparison with published work (different conditions)

| Source | Model | Input | Image-wise | Object-wise | Protocol |
|---|---|---|---|---|---|
| [Kumra 2020] Table III | GR-ConvNet | RGB-D | 97.7 % | 96.6 % | cross-validation; crop centred on GT; their augmentation |
| [Kumra 2020] Table III | GG-CNN (as reported) | D | 73.0 % | 69.0 % | as reported there |
| **this project** | GR-ConvNet | RGB-D | **88.6 %** (single 132-image test split) | 75.9 % on block split (≈ object-wise) | fixed train-chosen crop, single seed |
| **this project** | GG-CNN | RGB-D | 84.8 % | 78.4 % on block split | same |

The published numbers were **obtained under different conditions** and are not reproduced
here. Known differences: (i) the GR-ConvNet reference code centres each test crop on the
image's ground-truth grasps, while this project uses a fixed crop chosen from training data;
(ii) cross-validation folds versus one 70/15/15 split; (iii) a true object-wise split versus our
approximate block split; (iv) unspecified epoch counts and selection rules in the paper. The
roughly 9-point image-wise gap to the paper is therefore **not** a like-for-like measurement of
implementation quality.

### 9.9 What the evidence does and does not support

**Supported:**
* A dense network trained here, on this machine, detects Cornell-criterion grasps on about 88–90 %
  of held-out image-wise test scenes (95 % CI roughly 82–94 %). It runs in a few milliseconds
  per image on a laptop GPU.
* Accuracy drops by about 10 points when consecutive-scene blocks are held out.
* The model proposes several distinct candidates; top-5 accuracy is 95.5 %.

**Not supported:**
* That GR-ConvNet is better than GG-CNN, or RGB-D better than RGB, on this data. The differences
  are not significant.
* Any object-level generalisation claim (no object IDs).
* Any physical grasp success, 3-D pose accuracy, robustness to clutter, or robustness to other
  cameras.

## 10. Inference Application

### 10.1 Architecture and data flow

```mermaid
flowchart LR
  U["user uploads RGB PNG<br/>(+ optional depth TIFF)"] --> A["app/app.py · App.run"]
  A -->|depth given and RGB-D model loaded| P1["GraspPredictor (RGB-D)"]
  A -->|otherwise| P2["GraspPredictor (RGB)"]
  P1 --> D["preprocess (fixed crop, 224²) → model → decode → map back"]
  P2 --> D
  D --> V["render_prediction → annotated image"]
  D --> T["table: rank, score, x, y, θ, width, contacts"]
  V --> F["PNG + JSON download"]
  T --> F
```

`app/app.py` builds a Gradio `Blocks` UI. At start-up it loads the checkpoints named in
`app/app_config.yaml` (or given with `--checkpoint` / `--rgb-checkpoint`). Each load is wrapped
so that a failure is **shown in the status panel**. If no model loads, the predict button raises
a visible `gr.Error`. There is no placeholder output. The model is rebuilt from the config stored
in the checkpoint (§6.10), so preprocessing is identical to training.

### 10.2 Using it

```bash
.venv\Scripts\python.exe app\app.py            # http://127.0.0.1:7860
```

Upload a Cornell-style image (for RGB-D, also the matching `pcdXXXXd.tiff`) and choose the number
of candidates. The result shows amber candidates with `rank:score`, the selected grasp in blue
(thicker edges = jaw plates) and its contact points in magenta. A markdown summary gives the
selected centre, angle, opening, score and contact points in original pixels, and a table lists
all candidates. The annotated PNG and a JSON file can be downloaded. The page states the pixel
and angle conventions and the single-object, planar-only limitation.

### 10.3 Common errors

| Message | Meaning |
|---|---|
| "No model loaded — predictions are disabled" | checkpoint missing or path wrong in `app/app_config.yaml` |
| "Only an RGB-D model is loaded: please also upload the matching depth TIFF" | no RGB-only checkpoint configured |
| "Depth size … does not match image size …" | RGB and depth come from different scenes or resolutions |
| poor grasps on your own photos | domain shift (§11). The model only saw Cornell scenes |

### 10.4 Verification performed

The app was launched with `python app/app.py --port 7861`. It loaded the RGB-D checkpoint (the
RGB-only run had not finished yet at that point and was reported as missing in the status
panel, as designed). It returned HTTP 200, and its `/run` endpoint was then called through
`gradio_client` with validation scene pcd0258 (RGB PNG + depth TIFF). The returned selected
grasp was centre (317.5, 283.4) px, θ = 86.7°, opening 101.5 px, score 1.316, contact points
(314.6, 334.0) and (320.4, 232.7). This is a grasp across the short side of the box. The
annotated PNG and JSON were produced for download. `tests/test_app.py` checks that a missing
checkpoint gives an explicit error and no prediction, and that a loaded model returns real
candidates.

**Final check, with both models present.** The app was relaunched on port 7862 and its status
panel listed `grconvnet_rgbd_image` (epoch 85) and `grconvnet_rgb_image` (epoch 49), both on
CUDA. For test scene pcd0101, uploading RGB + depth used the RGB-D model and returned the same
two candidates as `scripts/predict.py` (§9.6): rank 1 at (305.9, 297.5), 84.9°, 66.9 px, score
1.122. Uploading the RGB image alone used the RGB-only model, whose rank 1 was at
(291.8, 293.6), −87.6°, 45.7 px, score 0.839. That is also a grasp across the stick; −87.6° and
84.9° differ by only 7.5° modulo 180°.


## 11. Limitations and Future Work

### 11.1 Limitations

* **Small dataset, narrow domain.** 885 single-object scenes from one setup. Validation and test
  sets of ~120–133 images give accuracy estimates with confidence intervals of several points.
  The model will degrade under a different camera, height, lighting or background.
* **Annotation limits.** Labels are sparse, so correct-looking grasps can count as failures.
  Labels are also noisy (approximate rectangles; one rectangle with NaN coordinates).
* **Unseen objects.** Without object IDs, generalisation to genuinely new objects is not
  measured. The block split is only an approximation.
* **RGB vs RGB-D.** The RGB-D model needs registered depth. The RGB-only model is used for plain
  uploads and has its own measured accuracy (§9).
* **No clutter, no instance selection.** Each scene contains one object. The model does not
  choose between objects.
* **2-D only.** No camera intrinsics, no camera-to-robot calibration, no 3-D pose, no approach
  height, no collision checking, no physical validation. The score is not a success probability.
* **Single seed.** Each configuration was trained once (seed 42). Seed-to-seed variance is not
  measured.

### 11.2 Future work, prioritised by value and cost

| Priority | Work | Value | Cost |
|---|---|---|---|
| 1 | Obtain Cornell's `z.txt` and run a true object-wise split | measures unseen-object generalisation | low (one metadata file, owner approval needed to download) |
| 2 | 3–5 seeds per configuration / 5-fold cross-validation | confidence intervals; fair model comparison | low (≈ 10 min per run on this GPU) |
| 3 | Back-project the selected grasp to 3-D with depth + intrinsics | first step towards execution | low–medium |
| 4 | Train/test on Jacquard (≈ 54 k scenes, simulated) | far more data; standard second benchmark | medium (download ~50 GB) |
| 5 | Test-time augmentation (average predictions over rotations) | often +1–2 points | low |
| 6 | Cluttered-scene data (e.g. GraspNet-1Billion), instance segmentation | real-world relevance | high |
| 7 | Camera-to-robot calibration, collision checking, physical trials | the only way to claim grasp success | high (hardware) |

## 12. Reproduction and Deployment

All commands are run from the repository root on Windows (PowerShell or Git Bash). Replace
`.venv\Scripts\python.exe` with `.venv/bin/python` on Linux/macOS.

### 12.1 Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps
copy configs\local.example.yaml configs\local.yaml     # edit dataset_root
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

### 12.2 Data preparation (deterministic)

```bash
.venv\Scripts\python.exe scripts\inspect_dataset.py      # stats, exclusions, figures
.venv\Scripts\python.exe scripts\make_splits.py --seed 42 # split files + crop analysis
.venv\Scripts\python.exe scripts\visualize_targets.py    # optional diagnostic
```

### 12.3 Tests and training

```bash
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\train.py --config configs\experiment_configs\grconvnet_rgbd_image.yaml --smoke
.venv\Scripts\python.exe scripts\train.py --config configs\experiment_configs\grconvnet_rgbd_image.yaml
#   or all five runs in sequence:
.venv\Scripts\python.exe scripts\run_experiments.py configs\experiment_configs\grconvnet_rgbd_image.yaml configs\experiment_configs\ggcnn_rgbd_image.yaml configs\experiment_configs\grconvnet_rgbd_block.yaml configs\experiment_configs\ggcnn_rgbd_block.yaml configs\experiment_configs\grconvnet_rgb_image.yaml
#   resume an interrupted run:
.venv\Scripts\python.exe scripts\train.py --config configs\experiment_configs\grconvnet_rgbd_image.yaml --resume
```

Each run writes `outputs/runs/<name>/{config.yaml, env.json, train.log, history.csv,
train_result.json, checkpoints/best.pt, checkpoints/last.pt}`. A run is finished when
`train.log` ends with `TRAINING FINISHED`.

### 12.4 Evaluation, figures and inference

```bash
.venv\Scripts\python.exe scripts\evaluate.py --run outputs\runs\grconvnet_rgbd_image --split val
.venv\Scripts\python.exe scripts\evaluate.py --run outputs\runs\grconvnet_rgbd_image --split test
.venv\Scripts\python.exe scripts\summarize_results.py
.venv\Scripts\python.exe scripts\make_report_figures.py --run outputs\runs\grconvnet_rgbd_image --split test
.venv\Scripts\python.exe scripts\predict.py --checkpoint outputs\runs\grconvnet_rgbd_image\checkpoints\best.pt --rgb "<dataset>\01\pcd0100r.png" --depth "<dataset>\01\pcd0100d.tiff" --gt "<dataset>\01\pcd0100cpos.txt"
.venv\Scripts\python.exe app\app.py
.venv\Scripts\python.exe scripts\check_report.py
```

Checkpoints are **not** in Git (≈ 23 MB each for GR-ConvNet including optimiser state). To use
the app on another machine, copy `outputs/runs/<name>/checkpoints/best.pt` or retrain (about
10 minutes per run on an RTX 4060 Laptop GPU).

### 12.5 Expected outputs and troubleshooting

`experiments/results/<run>__<split>.json` holds the summary metrics with provenance (checkpoint
path, SHA-256 prefix, epoch, split file, environment). `<run>__<split>_per_image.csv` holds
one row per image. Figures go to `experiments/figures/<run>/<split>/` and `docs/figures/`. See
`docs/troubleshooting.md` for errors (CPU-only torch, DataLoader workers on Windows, OOM, missing
config).

## 13. Conclusion

**Work completed.** The project parses the real local Cornell data with full validation and
reporting, builds leakage-free reproducible splits, and trains two dense grasp networks to
completion under identical conditions (five full runs). Each selected checkpoint was evaluated
exactly once on a held-out test split, with confidence intervals and paired significance tests.
The trained model is exposed through a CLI and a Gradio app that show ranked candidates, the
selected grasp, its centre, angle, width, score and contact points.

**Findings.** (1) About 89 % top-1 and 96 % top-5 image-wise test accuracy for the final
GR-ConvNet RGB-D model, at about 6 ms per image for network plus decoding. (2) About 10 points
lower when blocks of consecutive scenes are held out. (3) No significant difference between
architectures or modalities at this test-set size. (4) A quality score that ranks but is not
calibrated.

**Lessons.**
* Transform the label *vertices* with the same matrix as the pixels. It removes a whole class
  of angle and flip bugs.
* With 120–130 test images, ±6-point confidence intervals make one-run comparisons between
  models unreliable. Significance tests should be reported, not just point estimates.
* Picking the best of many noisy validation epochs inflates validation accuracy. Expect the test
  score to come out lower.
* Protocol details (how the crop is centred, the split type, cross-validation) move Cornell
  numbers by several points. Published comparisons must be read with the protocol in hand.

**Next research steps** (§11.2): a true object-wise split once `z.txt` is available, repeated
seeds or cross-validation, back-projection to 3-D grasps with camera intrinsics, then a
cluttered dataset and physical trials.

---

## Appendix A — Configuration reference (`configs/default.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `experiment.name` | default | run directory name under `outputs/runs/` |
| `experiment.seed` | 42 | Python/NumPy/PyTorch seed; augmentation RNG seed |
| `data.split_file` | experiments/splits/image_wise_seed42.json | which split to use |
| `data.modality` | rgbd | `rgb`, `depth` or `rgbd` (channel order depth, R, G, B) |
| `data.output_size` | 224 | network input/output side (px) |
| `data.crop.{cx,cy,size}` | 304, 275, 288 | fixed crop window in original pixels |
| `data.max_width` | 150 | width-map normaliser (network px) |
| `data.num_workers` | 2 | DataLoader workers (0 = main process) |
| `data.augment.*` | enabled, 180°, 0.85, 20 px, 0.5 | rotation range, min zoom, shift, flip probability |
| `model.name` | grconvnet | `grconvnet` or `ggcnn` |
| `model.channels / num_residual / dropout` | 32 / 5 / 0.1 | GR-ConvNet width, depth, head dropout |
| `model.filters` | (32, 16, 8) | GG-CNN filters (when `name: ggcnn`) |
| `train.epochs / batch_size / val_batch_size` | 100 / 8 / 8 | |
| `train.optimizer / lr / weight_decay` | adam / 1e-3 / 0 | `adam` or `adamw` |
| `train.scheduler / min_lr` | cosine / 1e-5 | `cosine`, `step` (needs `step_size`, `gamma`) or `none` |
| `train.amp / grad_clip` | true / 5.0 | mixed precision; max gradient norm (0 = off) |
| `train.early_stopping_patience` | 30 | epochs without improvement (0 = off) |
| `decode.q_sigma / q_threshold / min_distance / max_peaks` | 2.0 / 0.1 / 5 / 50 | peak extraction |
| `decode.min_width` | 2.0 | minimum opening (network px) |
| `decode.nms_iou / nms_angle_deg / max_candidates` | 0.3 / 30 / 10 | NMS and output count |
| `eval.k` | 5 | top-k accuracy |

Machine-specific: `configs/local.yaml` → `dataset_root` (or env `GRASP_DATASET_ROOT`).
App: `app/app_config.yaml` → `rgbd_checkpoint`, `rgb_checkpoint`.

## Appendix B — Notation

| Symbol | Meaning |
|---|---|
| `(x, y)` | pixel coordinates, origin top-left, y down |
| `θ` | grasp (closing-direction) angle, counter-clockwise on screen, in [−90°, 90°) |
| `w` | gripper opening (px); `h` jaw size (px) |
| `d = (cos θ, −sin θ)` | unit closing direction in image coordinates |
| `c₁, c₂` | contact points `(x, y) ∓ (w/2)·d` |
| `Q, W` | quality and normalised width maps; `Q̃` smoothed quality |
| `M` | 2 × 3 affine, original → network pixels; `s` scale, `R(α)` rotation, `F` mirror |
| `S, L, z` | network size (224), crop side (288), zoom factor |
| `J(P, G)` | IoU of rectangles P and G; `Δθ` angle difference modulo π |
| `ℓ_β` | Smooth-L1 loss with β = 1 |
| `η` | learning rate; `m, v` Adam moments |

## Appendix C — Directory and module reference

```
configs/            default.yaml, experiment_configs/*.yaml, local.example.yaml (local.yaml ignored)
src/grasp_planner/
  config.py         YAML loading, dataset-root resolution
  pipeline.py       config -> samples/datasets/model/checkpoint glue
  data/             cornell.py (parse), grasp.py (geometry), transforms.py (affine),
                    dataset.py (targets, torch Dataset), splits.py
  models/           ggcnn.py, grconvnet.py, heads.py, __init__ (factory)
  training/         losses.py, trainer.py
  evaluation/       metrics.py, runner.py
  inference/        decode.py, predictor.py
  visualization/    plots.py
  utils/            seeding, device, environment info, logging
scripts/            inspect_dataset, make_splits, visualize_targets, train, run_experiments,
                    evaluate, summarize_results, make_report_figures, predict, check_report
app/                app.py (Gradio), app_config.yaml
tests/              unit + integration tests (pytest)
experiments/        splits/*.json, results/*.json|csv|md (committed); figures/ (ignored)
docs/               this report, supporting docs, figures/
outputs/            runs/, logs/, cache/, predictions/ (ignored)
```

## Appendix D — Experiment table

| Run | Config | Split file | Best epoch | Val top-1 | Test top-1 | Checkpoint SHA-256 (16) |
|---|---|---|---|---|---|---|
| grconvnet_rgbd_image | `configs/experiment_configs/grconvnet_rgbd_image.yaml` | image_wise_seed42 | 85 | 93.2 % | 88.6 % | 23fda02a829da78c |
| ggcnn_rgbd_image | `ggcnn_rgbd_image.yaml` | image_wise_seed42 | 44 | 88.0 % | 84.8 % | 46c1d0d0cac203fe |
| grconvnet_rgb_image | `grconvnet_rgb_image.yaml` | image_wise_seed42 | 49 | 90.2 % | 90.2 % | 0d6788f22496b7fc |
| grconvnet_rgbd_block | `grconvnet_rgbd_block.yaml` | block_wise_seed42 | 18 | 80.3 % | 75.9 % | 0e751622407536d9 |
| ggcnn_rgbd_block | `ggcnn_rgbd_block.yaml` | block_wise_seed42 | 19 | 73.8 % | 78.4 % | cc729a615867f5f9 |

Smoke tests are listed in `docs/experiment_log.md` and are excluded from this table.

## Appendix E — Further prediction examples

![RGB-only model, test split](figures/qualitative_grconvnet_rgb_image_test.png)
*Figure E.1 — RGB-only GR-ConvNet on random test successes and failures (same rendering as
Figure 9.3). This is the model the app uses when no depth file is uploaded.*

Per-image overlays for every run and split (six successes and six failures each) are
regenerated by `scripts/evaluate.py` into `experiments/figures/<run>/<split>/`.

## Appendix F — Glossary

* **Antipodal grasp:** two contacts on opposite sides of an object, with the closing line
  through both.
* **AMP:** automatic mixed precision (fp16 compute with fp32 master weights).
* **Augmentation:** random, label-consistent transformations of training samples.
* **Checkpoint:** saved model (and optimiser) state.
* **Dense prediction:** an output value for every input pixel.
* **Early stopping:** stopping training when validation performance stops improving.
* **IoU / Jaccard index:** intersection area over union area.
* **McNemar test:** a paired test of two classifiers on the same items, using only the items
  where they disagree.
* **NMS:** non-maximum suppression, removing near-duplicate detections.
* **Object-wise / image-wise split:** held-out objects vs held-out images.
* **Rectangle metric:** the Cornell success criterion (IoU > 0.25, Δθ < 30°).
* **Wilson interval:** a confidence interval for a proportion that behaves well near 0 and 1.

## Appendix G — References

See `docs/references.md` (verified bibliographic details and the published Cornell results
table).

## Appendix H — Chapter-to-source map

| Chapter | Files |
|---|---|
| 2 Problem formulation | `data/grasp.py` |
| 3 Data | `data/cornell.py`, `data/transforms.py`, `data/dataset.py`, `data/splits.py`, `scripts/inspect_dataset.py`, `scripts/make_splits.py`, `scripts/visualize_targets.py` |
| 4 Mathematics | `models/*`, `training/losses.py`, `training/trainer.py`, `data/grasp.py`, `inference/decode.py` |
| 5 Architecture | `models/grconvnet.py`, `models/ggcnn.py`, `models/heads.py` |
| 6 Code walkthrough | all of `src/grasp_planner/` |
| 7 Training | `training/trainer.py`, `scripts/train.py`, `scripts/run_experiments.py`, `configs/` |
| 8 Decoding | `inference/decode.py`, `inference/predictor.py` |
| 9 Evaluation | `evaluation/metrics.py`, `evaluation/runner.py`, `scripts/evaluate.py`, `scripts/summarize_results.py`, `scripts/make_report_figures.py` |
| 10 App | `app/app.py`, `visualization/plots.py` |
| 12 Reproduction | `README.md`, `requirements.txt`, `pyproject.toml` |
