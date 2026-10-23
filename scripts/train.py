"""Train a grasp model from a YAML config.

Usage:
    python scripts/train.py --config configs/experiment_configs/grconvnet_rgbd_image.yaml
    python scripts/train.py --config ... --resume            # continue from <run>/checkpoints/last.pt
    python scripts/train.py --config ... --smoke              # 2 epochs x 3 batches, run dir *_smoke

Outputs go to outputs/runs/<experiment.name>/ : config.yaml, train.log, history.csv,
env.json, checkpoints/{best,last}.pt, train_result.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_planner.config import REPO_ROOT, load_config  # noqa: E402
from grasp_planner.pipeline import decode_spec, make_dataset, make_model, split_samples  # noqa: E402
from grasp_planner.training.trainer import Trainer  # noqa: E402
from grasp_planner.utils import environment_info, seed_everything, select_device, setup_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline end to end")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"] + ("_smoke" if args.smoke else "")
    if args.smoke:
        cfg["train"]["epochs"] = 2
    run_dir = REPO_ROOT / "outputs" / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    seed_everything(int(cfg["experiment"]["seed"]))
    device = select_device(args.device)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    (run_dir / "env.json").write_text(json.dumps(environment_info(device), indent=2), encoding="utf-8")

    parts = split_samples(cfg)  # test samples are parsed but never given to the trainer
    train_set = make_dataset(cfg, parts["train"], train=True)
    val_set = make_dataset(cfg, parts["val"], train=False)
    if args.smoke:
        val_set.samples = val_set.samples[:16]
    # warm the inpainted-depth disk cache in the main process (avoids worker races)
    if train_set.spec.modality != "rgb":
        for s in parts["train"] + parts["val"]:
            train_set.depth_cache.get(s)

    model = make_model(cfg)
    trainer = Trainer(cfg, model, train_set, val_set, decode_spec(cfg), run_dir, device)
    if args.resume:
        last = run_dir / "checkpoints" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"--resume given but {last} does not exist")
        trainer.resume(last)
    trainer.fit(max_batches=3 if args.smoke else None)


if __name__ == "__main__":
    main()
