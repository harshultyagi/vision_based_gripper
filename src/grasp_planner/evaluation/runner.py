"""Run a model over a (non-augmented) dataset split and score it with the Cornell metric."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import CornellGraspDataset
from ..data.grasp import Grasp, rect_to_grasp
from ..inference.decode import DecodeSpec, decode_grasps, grasp_to_original
from ..models.heads import HEAD_NAMES
from ..training.losses import grasp_loss
from .metrics import ImageResult, evaluate_image, summarise


@torch.no_grad()
def predict_maps(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Forward pass -> numpy (B, 4, H, W) in HEAD_NAMES order (float32)."""
    out = model(x)
    return torch.cat([out[n] for n in HEAD_NAMES], dim=1).float().cpu().numpy()


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataset: CornellGraspDataset,
    decode_spec: DecodeSpec,
    device: torch.device,
    batch_size: int = 8,
    k: int = 5,
    num_workers: int = 0,
    amp: bool = False,
) -> tuple[dict, list[ImageResult], list[list[Grasp]]]:
    """Evaluate ``model`` on every scene of ``dataset`` (which must have ``augment=None``).

    Returns ``(summary, per_image_results, predictions_in_original_coords)``. The summary also
    contains the mean dense loss on this split.
    """
    if dataset.augment is not None:
        raise ValueError("evaluation dataset must not be augmented")
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    results: list[ImageResult] = []
    all_preds: list[list[Grasp]] = []
    loss_sum, n = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            out = model(x)
        loss, _ = grasp_loss(out, y)
        loss_sum += float(loss) * x.shape[0]
        n += x.shape[0]
        maps = torch.cat([out[h] for h in HEAD_NAMES], dim=1).float().cpu().numpy()
        for b in range(x.shape[0]):
            idx = int(batch["index"][b])
            sample = dataset.samples[idx]
            M = batch["M"][b].numpy()
            preds = [grasp_to_original(g, M) for g in decode_grasps(*maps[b], decode_spec)]
            gts = [rect_to_grasp(r) for r in sample.pos]
            results.append(evaluate_image(sample.sample_id, preds, gts, k=k))
            all_preds.append(preds)
    summary = summarise(results, k)
    summary["loss"] = loss_sum / max(n, 1)
    return summary, results, all_preds


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    dataset: CornellGraspDataset,
    decode_spec: DecodeSpec,
    device: torch.device,
    n: int = 100,
    warmup: int = 10,
) -> dict:
    """Median per-image latency (batch size 1) of the forward pass and of forward + decoding.
    Excludes image loading and preprocessing."""
    model.eval()
    x = dataset[0]["x"][None].to(device)
    fwd, total = [], []
    for i in range(warmup + n):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        maps = predict_maps(model, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        decode_grasps(*maps[0], decode_spec)
        t2 = time.perf_counter()
        if i >= warmup:
            fwd.append((t1 - t0) * 1e3)
            total.append((t2 - t0) * 1e3)
    return {
        "forward_ms_median": float(np.median(fwd)),
        "forward_plus_decode_ms_median": float(np.median(total)),
        "num_runs": n,
        "batch_size": 1,
        "note": "excludes image file loading and preprocessing; maps copied to CPU before decoding",
    }
