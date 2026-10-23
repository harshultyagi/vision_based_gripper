"""Evaluate a trained run's best checkpoint on the validation or test split.

Usage:
    python scripts/evaluate.py --run outputs/runs/<name> --split val
    python scripts/evaluate.py --run outputs/runs/<name> --split test   # only after model selection

Writes:
    experiments/results/<name>__<split>.json           summary metrics + provenance
    experiments/results/<name>__<split>_per_image.csv  per-image results
    experiments/figures/<name>/<split>/                success / failure overlays (--figures N)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_planner.config import REPO_ROOT  # noqa: E402
from grasp_planner.data.cornell import load_rgb  # noqa: E402
from grasp_planner.evaluation.runner import evaluate_dataset, measure_latency  # noqa: E402
from grasp_planner.pipeline import decode_spec, load_checkpoint_model, make_dataset, split_samples  # noqa: E402
from grasp_planner.utils import environment_info, select_device, setup_logging  # noqa: E402
from grasp_planner.visualization.plots import render_prediction  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_examples(samples, results, preds, out_dir: Path, n: int) -> list[str]:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort([-r.score if np.isfinite(r.score) else 0 for r in results])
    succ = [i for i in order if results[i].top1_success][:n]
    fail = [i for i in order if not results[i].top1_success][:n]
    written = []
    for tag, idxs in (("success", succ), ("failure", fail)):
        for i in idxs:
            s = samples[i]
            img = render_prediction(load_rgb(s.rgb_path), preds[i], s.pos, None, num_candidates=5)
            p = out_dir / f"{tag}_pcd{s.sample_id}.png"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            written.append(p.name)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory containing checkpoints/best.pt")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--figures", type=int, default=6, help="success/failure examples to save (each)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    setup_logging()
    run = Path(args.run)
    if not run.is_absolute():
        run = REPO_ROOT / run
    ckpt_path = run / "checkpoints" / args.checkpoint
    device = select_device(args.device)
    model, ckpt = load_checkpoint_model(ckpt_path, device)
    cfg = ckpt["config"]
    name = cfg["experiment"]["name"]
    samples = split_samples(cfg)[args.split]
    ds = make_dataset(cfg, samples, train=False)
    dspec = decode_spec(cfg)
    summary, results, preds = evaluate_dataset(model, ds, dspec, device, k=args.k)
    latency = measure_latency(model, ds, dspec, device)

    summary.update(
        {
            "experiment": name,
            "split": args.split,
            "split_file": cfg["data"]["split_file"],
            "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
            "checkpoint_sha256_16": sha256(ckpt_path),
            "checkpoint_epoch": ckpt["epoch"],
            "selection_criterion": "best validation top-1 accuracy (ties: lower val loss)",
            "metric": "Cornell rectangle metric: IoU > 0.25 and |angle diff| < 30 deg (mod 180)",
            "latency": latency,
            "environment": environment_info(device),
            "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    res_dir = REPO_ROOT / "experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out = res_dir / f"{name}__{args.split}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (res_dir / f"{name}__{args.split}_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].to_dict().keys()))
        w.writeheader()
        for r in results:
            w.writerow(r.to_dict())
    if args.figures:
        save_examples(
            samples, results, preds, REPO_ROOT / "experiments" / "figures" / name / args.split, args.figures
        )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "experiment",
                    "split",
                    "num_images",
                    "top1_accuracy",
                    f"top{args.k}_accuracy",
                    "center_error_px",
                    "angle_error_deg",
                    "width_error_px",
                    "latency",
                )
            },
            indent=1,
        )
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
