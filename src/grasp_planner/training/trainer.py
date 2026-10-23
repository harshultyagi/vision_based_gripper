"""Training loop with validation-based checkpoint selection, early stopping and resume.

Selection criterion (fixed before training): highest validation top-1 rectangle accuracy;
ties broken by lower validation loss. The test split is never loaded here.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import CornellGraspDataset
from ..evaluation.runner import evaluate_dataset
from ..inference.decode import DecodeSpec
from ..models import count_parameters
from .losses import grasp_loss

log = logging.getLogger(__name__)


class NonFiniteLossError(RuntimeError):
    pass


def build_optimizer(model: torch.nn.Module, tcfg: dict) -> torch.optim.Optimizer:
    name = tcfg.get("optimizer", "adam").lower()
    lr, wd = float(tcfg["lr"]), float(tcfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer {name}")


def build_scheduler(opt: torch.optim.Optimizer, tcfg: dict):
    name = tcfg.get("scheduler", "none").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(tcfg["epochs"]), eta_min=float(tcfg.get("min_lr", 1e-6))
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, int(tcfg["step_size"]), float(tcfg.get("gamma", 0.5)))
    if name == "none":
        return None
    raise ValueError(f"unknown scheduler {name}")


def _better(score: tuple[float, float], best: tuple[float, float] | None) -> bool:
    """score = (val_top1, val_loss): higher accuracy wins, then lower loss."""
    if best is None:
        return True
    return score[0] > best[0] or (score[0] == best[0] and score[1] < best[1])


class Trainer:
    def __init__(
        self,
        cfg: dict,
        model: torch.nn.Module,
        train_set: CornellGraspDataset,
        val_set: CornellGraspDataset,
        decode_spec: DecodeSpec,
        run_dir: Path,
        device: torch.device,
    ):
        self.cfg = cfg
        self.tcfg = cfg["train"]
        self.model = model.to(device)
        self.train_set = train_set
        self.val_set = val_set
        self.decode_spec = decode_spec
        self.run_dir = Path(run_dir)
        self.device = device
        self.amp = bool(self.tcfg.get("amp", True)) and device.type == "cuda"
        self.opt = build_optimizer(self.model, self.tcfg)
        self.sched = build_scheduler(self.opt, self.tcfg)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.start_epoch = 0
        self.best: tuple[float, float] | None = None
        self.best_epoch = -1
        self.epochs_since_best = 0
        nw = int(cfg["data"].get("num_workers", 0))
        self.train_loader = DataLoader(
            train_set,
            batch_size=int(self.tcfg["batch_size"]),
            shuffle=True,
            num_workers=nw,
            pin_memory=device.type == "cuda",
            drop_last=True,
            persistent_workers=nw > 0,
            generator=torch.Generator().manual_seed(int(cfg["experiment"]["seed"])),
        )
        self.history_path = self.run_dir / "history.csv"

    # ------------------------------------------------------------------ checkpoints
    def _state(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict() if self.sched else None,
            "scaler": self.scaler.state_dict(),
            "best": self.best,
            "best_epoch": self.best_epoch,
            "epochs_since_best": self.epochs_since_best,
            "config": self.cfg,
            "torch_rng": torch.get_rng_state(),
        }

    def save(self, name: str, epoch: int) -> Path:
        path = self.run_dir / "checkpoints" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(self._state(epoch), tmp)
        tmp.replace(path)  # atomic on the same filesystem: never leaves a half-written checkpoint
        return path

    def resume(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        if self.sched and ck["scheduler"]:
            self.sched.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.best = tuple(ck["best"]) if ck["best"] else None
        self.best_epoch = ck["best_epoch"]
        self.epochs_since_best = ck["epochs_since_best"]
        self.start_epoch = ck["epoch"] + 1
        torch.set_rng_state(ck["torch_rng"])
        log.info(
            "resumed from %s at epoch %d (best %s @ %d)", path, self.start_epoch, self.best, self.best_epoch
        )

    # ------------------------------------------------------------------ loops
    def train_epoch(self, epoch: int, max_batches: int | None = None) -> dict:
        self.model.train()
        self.train_set.set_epoch(epoch)
        sums: dict[str, float] = {}
        n = 0
        clip = float(self.tcfg.get("grad_clip", 0.0))
        for b, batch in enumerate(self.train_loader):
            if max_batches is not None and b >= max_batches:
                break
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
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
            for k, v in {"loss": float(loss.detach()), **parts}.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        return {f"train_{k}": v / max(n, 1) for k, v in sums.items()}

    def validate(self) -> dict:
        summary, _, _ = evaluate_dataset(
            self.model,
            self.val_set,
            self.decode_spec,
            self.device,
            batch_size=int(self.tcfg.get("val_batch_size", 8)),
            k=int(self.cfg.get("eval", {}).get("k", 5)),
            amp=self.amp,
        )
        return {
            "val_loss": summary["loss"],
            "val_top1": summary["top1_accuracy"],
            "val_top5": summary.get("top5_accuracy"),
            "val_center_err": summary["center_error_px"]["mean"],
            "val_angle_err": summary["angle_error_deg"]["mean"],
        }

    def fit(self, max_batches: int | None = None) -> dict:
        epochs = int(self.tcfg["epochs"])
        patience = int(self.tcfg.get("early_stopping_patience", 0))
        log.info(
            "model parameters: %d | device %s | amp %s", count_parameters(self.model), self.device, self.amp
        )
        write_header = not self.history_path.exists() or self.start_epoch == 0
        stopped_early = False
        epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            row = {"epoch": epoch, **self.train_epoch(epoch, max_batches)}
            row.update(self.validate())
            row["lr"] = self.opt.param_groups[0]["lr"]
            row["epoch_time_s"] = round(time.time() - t0, 2)
            if self.sched:
                self.sched.step()
            score = (row["val_top1"], row["val_loss"])
            improved = _better(score, self.best)
            if improved:
                self.best, self.best_epoch, self.epochs_since_best = score, epoch, 0
                self.save("best.pt", epoch)
            else:
                self.epochs_since_best += 1
            row["is_best"] = int(improved)
            self.save("last.pt", epoch)
            self._append_history(row, write_header)
            write_header = False
            log.info(
                "epoch %3d | train %.4f | val loss %.4f top1 %.3f top5 %.3f | lr %.2e | %.1fs%s",
                epoch,
                row["train_loss"],
                row["val_loss"],
                row["val_top1"],
                math.nan if row["val_top5"] is None else row["val_top5"],
                row["lr"],
                row["epoch_time_s"],
                " *best*" if improved else "",
            )
            if patience and self.epochs_since_best >= patience:
                log.info("early stopping: no val improvement for %d epochs", patience)
                stopped_early = True
                break
        result = {
            "best_val_top1": self.best[0] if self.best else None,
            "best_val_loss": self.best[1] if self.best else None,
            "best_epoch": self.best_epoch,
            "last_epoch": epoch,
            "stopped_early": stopped_early,
            "completed": True,
        }
        (self.run_dir / "train_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("TRAINING FINISHED: %s", result)
        return result

    def _append_history(self, row: dict, header: bool) -> None:
        mode = "w" if header else "a"
        with self.history_path.open(mode, newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if header:
                w.writeheader()
            w.writerow(
                {k: (round(v, 6) if isinstance(v, float) and np.isfinite(v) else v) for k, v in row.items()}
            )
