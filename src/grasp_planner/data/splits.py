"""Reproducible train/validation/test splits saved as JSON files.

Two strategies are implemented:

``image``  (image-wise) -- scenes are shuffled with a fixed seed and cut by ratio. Views of the
    same physical object may land in different splits, so this measures generalisation to
    *new views of seen objects*. It is the protocol most published Cornell numbers use.

``block``  (contiguous-block, an approximation of object-wise) -- the local copy of Cornell
    has no object-identity file (the original ``z.txt``), so object identity cannot be
    established reliably. Cornell scene indices were captured in sessions where an object
    usually appears in a short run of consecutive indices, so the sorted scene list is cut into
    contiguous blocks of ``block_size`` scenes and whole blocks are assigned to splits. A
    ``gap`` of scenes at each block boundary between different splits is dropped to reduce
    leakage from runs that straddle a boundary. This *reduces* but does not *eliminate* object
    overlap (the same object can be re-photographed far apart), so no object-level
    generalisation claim is made from it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def image_wise_split(ids: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, list[str]]:
    """Random scene-level split with fixed seed."""
    ids = sorted(ids)
    rng = np.random.default_rng(seed)
    perm = [ids[i] for i in rng.permutation(len(ids))]
    n_train = int(round(ratios[0] * len(ids)))
    n_val = int(round(ratios[1] * len(ids)))
    return {
        "train": sorted(perm[:n_train]),
        "val": sorted(perm[n_train : n_train + n_val]),
        "test": sorted(perm[n_train + n_val :]),
    }


def block_split(
    ids: list[str], ratios: tuple[float, float, float], seed: int, block_size: int = 20, gap: int = 2
) -> dict[str, list[str]]:
    """Contiguous-block split (see module docstring). Blocks are assigned to splits at random
    (fixed seed) so that each split's share of blocks matches ``ratios``; ``gap`` scenes are
    removed on each side of a boundary between blocks that went to different splits."""
    ids = sorted(ids, key=int)
    blocks = [ids[i : i + block_size] for i in range(0, len(ids), block_size)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    n_train = int(round(ratios[0] * len(blocks)))
    n_val = int(round(ratios[1] * len(blocks)))
    assign = np.empty(len(blocks), dtype=object)
    assign[order[:n_train]] = "train"
    assign[order[n_train : n_train + n_val]] = "val"
    assign[order[n_train + n_val :]] = "test"
    out: dict[str, list[str]] = {"train": [], "val": [], "test": [], "dropped_gap": []}
    for b, block in enumerate(blocks):
        keep = list(block)
        if gap and b > 0 and assign[b - 1] != assign[b]:
            out["dropped_gap"] += keep[:gap]
            keep = keep[gap:]
        if gap and b < len(blocks) - 1 and assign[b + 1] != assign[b]:
            out["dropped_gap"] += keep[-gap:]
            keep = keep[:-gap]
        out[str(assign[b])] += keep
    for k in out:
        out[k] = sorted(out[k], key=int)
    return out


def make_split(
    ids: list[str], strategy: str, ratios: tuple[float, float, float], seed: int, **kwargs
) -> dict:
    if strategy == "image":
        parts = image_wise_split(ids, ratios, seed)
    elif strategy == "block":
        parts = block_split(ids, ratios, seed, **kwargs)
    else:
        raise ValueError(f"unknown split strategy {strategy!r}")
    check_disjoint(parts)
    return {
        "strategy": strategy,
        "seed": seed,
        "ratios": list(ratios),
        "params": kwargs,
        "num_input_ids": len(ids),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {k: len(v) for k, v in parts.items()},
        "splits": parts,
    }


def check_disjoint(parts: dict[str, list[str]]) -> None:
    """Raise if any scene id appears in more than one split."""
    seen: dict[str, str] = {}
    for name, ids in parts.items():
        for i in ids:
            if i in seen:
                raise AssertionError(f"scene {i} in both {seen[i]} and {name}")
            seen[i] = name


def save_split(split: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1), encoding="utf-8")


def load_split(path: Path) -> dict:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    check_disjoint(split["splits"])
    return split
