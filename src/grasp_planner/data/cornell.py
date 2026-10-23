"""Discovery, parsing and validation of the local Cornell Grasping Dataset.

Local layout (verified by ``scripts/inspect_dataset.py``)::

    <root>/01 ... <root>/10/
        pcdXXXX.txt       point cloud (PCD v0.7 ASCII; not used for training)
        pcdXXXXr.png      RGB image, 640 x 480
        pcdXXXXd.tiff     depth image, 640 x 480, float
        pcdXXXXcpos.txt   positive grasp rectangles, 4 lines "x y" per rectangle
        pcdXXXXcneg.txt   negative grasp rectangles, same format
    <root>/backgrounds/   pcdbXXXXr.png background-only images (no annotations; unused)

Nothing in this module writes to the dataset folder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grasp import rectangle_geometry_errors

log = logging.getLogger(__name__)

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


@dataclass
class ParsedRectangles:
    """Rectangles parsed from one annotation file plus a record of rejected ones."""

    rects: np.ndarray  # (N, 4, 2) float64, (x, y)
    rejected: list[tuple[int, str]] = field(default_factory=list)  # (rect index, reason)
    file_error: str | None = None


@dataclass
class CornellSample:
    """Paths and annotations for one Cornell scene."""

    sample_id: str  # e.g. "0100"
    folder: str  # e.g. "01"
    rgb_path: Path
    depth_path: Path | None
    pos_path: Path
    neg_path: Path | None
    pos: np.ndarray  # (N, 4, 2)
    neg: np.ndarray  # (M, 4, 2)
    issues: list[str] = field(default_factory=list)

    @property
    def index(self) -> int:
        return int(self.sample_id)


def parse_rectangle_file(
    path: Path, image_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT), bounds_margin: float = 0.0
) -> ParsedRectangles:
    """Parse a Cornell ``cpos``/``cneg`` file into rectangles.

    Each rectangle is four consecutive lines ``x y``. Rectangles with NaN coordinates,
    degenerate geometry, or vertices outside the image (beyond ``bounds_margin`` pixels) are
    rejected and reported -- never dropped silently.
    """
    try:
        lines = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError as exc:
        return ParsedRectangles(np.zeros((0, 4, 2)), file_error=f"unreadable: {exc}")
    if len(lines) % 4 != 0:
        return ParsedRectangles(
            np.zeros((0, 4, 2)), file_error=f"{len(lines)} coordinate lines is not a multiple of 4"
        )
    rects: list[np.ndarray] = []
    rejected: list[tuple[int, str]] = []
    w, h = image_size
    for i in range(len(lines) // 4):
        chunk = lines[4 * i : 4 * i + 4]
        try:
            pts = np.array([[float(v) for v in ln[:2]] for ln in chunk], dtype=np.float64)
            if any(len(ln) != 2 for ln in chunk):
                raise ValueError("line does not have exactly two values")
        except ValueError as exc:
            rejected.append((i, f"parse error: {exc}"))
            continue
        errs = rectangle_geometry_errors(pts)
        if not errs and (
            np.any(pts[:, 0] < -bounds_margin)
            or np.any(pts[:, 0] > w - 1 + bounds_margin)
            or np.any(pts[:, 1] < -bounds_margin)
            or np.any(pts[:, 1] > h - 1 + bounds_margin)
        ):
            errs = ["vertex outside image bounds"]
        if errs:
            rejected.append((i, "; ".join(errs)))
            continue
        rects.append(pts)
    arr = np.stack(rects) if rects else np.zeros((0, 4, 2))
    return ParsedRectangles(arr, rejected)


def discover_samples(root: Path) -> tuple[list[CornellSample], list[dict]]:
    """Find and parse every annotated scene under ``root``.

    Returns ``(samples, exclusions)``. A scene is excluded (with a reason) when its RGB image
    or positive-grasp file is missing, the file is malformed, or no valid positive rectangle
    remains. Rejected individual rectangles are listed in ``sample.issues``.
    """
    root = Path(root)
    samples: list[CornellSample] = []
    exclusions: list[dict] = []
    folders = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    if not folders:
        raise FileNotFoundError(f"no numbered scene folders found under {root}")
    for folder in folders:
        pos_files = sorted(folder.glob("pcd*cpos.txt"))
        rgb_files = {p.name[3:7] for p in folder.glob("pcd*r.png")}
        for sid in sorted(rgb_files - {p.name[3:7] for p in pos_files}):
            exclusions.append({"sample_id": sid, "folder": folder.name, "reason": "missing cpos.txt"})
        for pos_path in pos_files:
            sid = pos_path.name[3:7]
            rgb = folder / f"pcd{sid}r.png"
            depth = folder / f"pcd{sid}d.tiff"
            neg = folder / f"pcd{sid}cneg.txt"
            if not rgb.exists():
                exclusions.append({"sample_id": sid, "folder": folder.name, "reason": "missing RGB image"})
                continue
            pos = parse_rectangle_file(pos_path)
            if pos.file_error:
                exclusions.append(
                    {"sample_id": sid, "folder": folder.name, "reason": f"cpos: {pos.file_error}"}
                )
                continue
            negp = parse_rectangle_file(neg) if neg.exists() else ParsedRectangles(np.zeros((0, 4, 2)))
            issues = [f"cpos rect {i}: {r}" for i, r in pos.rejected]
            issues += [f"cneg rect {i}: {r}" for i, r in negp.rejected]
            if negp.file_error:
                issues.append(f"cneg: {negp.file_error}")
            if not depth.exists():
                issues.append("missing depth tiff")
            if len(pos.rects) == 0:
                exclusions.append(
                    {"sample_id": sid, "folder": folder.name, "reason": "no valid positive rectangles"}
                )
                continue
            samples.append(
                CornellSample(
                    sample_id=sid,
                    folder=folder.name,
                    rgb_path=rgb,
                    depth_path=depth if depth.exists() else None,
                    pos_path=pos_path,
                    neg_path=neg if neg.exists() else None,
                    pos=pos.rects,
                    neg=negp.rects,
                    issues=issues,
                )
            )
    samples.sort(key=lambda s: s.index)
    for s in samples:
        for issue in s.issues:
            log.warning("sample %s: %s", s.sample_id, issue)
    for e in exclusions:
        log.warning("excluded sample %s: %s", e["sample_id"], e["reason"])
    return samples, exclusions


def load_rgb(path: Path) -> np.ndarray:
    """Load an RGB image as uint8 (H, W, 3)."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"could not read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


MIN_VALID_DEPTH_M = 0.3  # Kinect-class sensors cannot measure closer than ~0.5 m; the local TIFFs
#                         contain a few near-zero pixels (~0.01%, image corners) that are artefacts


def mask_invalid_depth(depth: np.ndarray) -> np.ndarray:
    """Return a float32 copy with non-finite or implausibly small (< 0.3 m) depths set to NaN."""
    d = np.asarray(depth, dtype=np.float32).copy()
    d[~np.isfinite(d) | (d < MIN_VALID_DEPTH_M)] = np.nan
    return d


def load_depth(path: Path) -> np.ndarray:
    """Load a depth TIFF (metres) as float32 (H, W). Invalid pixels are returned as NaN."""
    import tifffile

    return mask_invalid_depth(tifffile.imread(str(path)))
