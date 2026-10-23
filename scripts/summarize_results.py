"""Build comparison tables and training-curve figures from saved run histories and result JSONs.
Every number comes from files written by train.py / evaluate.py.

Usage:
    python scripts/summarize_results.py
Writes:
    experiments/results/model_comparison.md and .json
    docs/figures/loss_curves_image_wise.png, loss_curves_block_wise.png
    experiments/results/history_<run>.csv (copies of the training histories, small)
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_planner.config import REPO_ROOT  # noqa: E402
from grasp_planner.visualization.plots import plot_history  # noqa: E402

RUNS = REPO_ROOT / "outputs" / "runs"
RES = REPO_ROOT / "experiments" / "results"
FIG = REPO_ROOT / "docs" / "figures"


def read_history(run: Path) -> list[dict]:
    with (run / "history.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(v, pct=False, nd=1):
    if v is None:
        return "n/a"
    return f"{100 * v:.{nd}f}%" if pct else f"{v:.{nd}f}"


PAIRS = [  # (A, B) compared on the same split's images
    ("grconvnet_rgbd_image", "ggcnn_rgbd_image"),
    ("grconvnet_rgbd_block", "ggcnn_rgbd_block"),
    ("grconvnet_rgbd_image", "grconvnet_rgb_image"),
]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / den
    return centre - half, centre + half


def per_image_success(run: str, split: str) -> dict[str, bool] | None:
    f = RES / f"{run}__{split}_per_image.csv"
    if not f.exists():
        return None
    with f.open(encoding="utf-8") as fh:
        return {r["sample_id"]: r["top1_success"] == "True" for r in csv.DictReader(fh)}


def statistics(rows: list[dict]) -> dict:
    """Wilson CIs per run/split and exact McNemar tests between paired runs."""
    from scipy.stats import binomtest

    intervals, tests = [], []
    for r in rows:
        for split in ("val", "test"):
            s = per_image_success(r["run"], split)
            if s is None:
                continue
            k, n = sum(s.values()), len(s)
            lo, hi = wilson(k, n)
            intervals.append(
                {"run": r["run"], "split": split, "k": k, "n": n, "top1": k / n, "ci_low": lo, "ci_high": hi}
            )
    for a, b in PAIRS:
        for split in ("val", "test"):
            sa, sb = per_image_success(a, split), per_image_success(b, split)
            if sa is None or sb is None or set(sa) != set(sb):
                continue
            a_only = sum(sa[i] and not sb[i] for i in sa)
            b_only = sum(sb[i] and not sa[i] for i in sa)
            n_disc = a_only + b_only
            p = binomtest(a_only, n_disc, 0.5).pvalue if n_disc else 1.0
            tests.append(
                {"split": split, "a": a, "b": b, "a_only": a_only, "b_only": b_only, "p_value": float(p)}
            )
    return {
        "intervals": intervals,
        "paired_tests": tests,
        "method": "Wilson score 95% CI; exact two-sided McNemar (binomial on discordant pairs)",
    }


def main() -> None:
    rows = []
    histories: dict[str, dict[str, list[dict]]] = {"image": {}, "block": {}}
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.name.endswith("_smoke")):
        if not (run / "history.csv").exists():
            continue
        hist = read_history(run)
        shutil.copy(run / "history.csv", RES / f"history_{run.name}.csv")
        split_kind = "block" if "block" in run.name else "image"
        histories[split_kind][run.name] = hist
        tr = (
            json.loads((run / "train_result.json").read_text())
            if (run / "train_result.json").exists()
            else {}
        )
        row = {
            "run": run.name,
            "split": split_kind,
            "epochs_run": len(hist),
            "completed": bool(tr.get("completed")),
            "best_epoch": tr.get("best_epoch"),
            "train_time_min": round(sum(float(h["epoch_time_s"]) for h in hist) / 60, 1),
        }
        for split in ("val", "test"):
            f = RES / f"{run.name}__{split}.json"
            if f.exists():
                r = json.loads(f.read_text())
                row[f"{split}_top1"] = r["top1_accuracy"]
                row[f"{split}_top5"] = r.get("top5_accuracy")
                row[f"{split}_center_err_px"] = r["center_error_px"]["mean"]
                row[f"{split}_angle_err_deg"] = r["angle_error_deg"]["mean"]
                row[f"{split}_width_err_px"] = r["width_error_px"]["mean"]
                row[f"{split}_n"] = r["num_images"]
                row["latency_fwd_ms"] = r["latency"]["forward_ms_median"]
                row["latency_total_ms"] = r["latency"]["forward_plus_decode_ms_median"]
        rows.append(row)

    (RES / "model_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# Model comparison (generated by scripts/summarize_results.py)", ""]
    for split in ("val", "test"):
        lines += [
            f"## {split} split",
            "",
            "| run | split strategy | epochs | best epoch | top-1 | top-5 | centre err (px) "
            "| angle err (deg) | width err (px) | n images |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            if f"{split}_top1" not in r:
                continue
            cells = [
                r["run"],
                r["split"],
                r["epochs_run"],
                r["best_epoch"],
                fmt(r[f"{split}_top1"], True),
                fmt(r[f"{split}_top5"], True),
                fmt(r[f"{split}_center_err_px"]),
                fmt(r[f"{split}_angle_err_deg"]),
                fmt(r[f"{split}_width_err_px"]),
                r[f"{split}_n"],
            ]
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
        lines.append("")
    stats = statistics(rows)
    (RES / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    lines += [
        "## Uncertainty (top-1)",
        "",
        "95% Wilson score intervals:",
        "",
        "| run | split | top-1 | 95% CI |",
        "|---|---|---|---|",
    ]
    for s in stats["intervals"]:
        lines.append(
            f"| {s['run']} | {s['split']} | {fmt(s['top1'], True)} "
            f"| {fmt(s['ci_low'], True)} – {fmt(s['ci_high'], True)} |"
        )
    lines += [
        "",
        "Paired exact McNemar tests (same images, discordant pairs only):",
        "",
        "| split | A | B | A only correct | B only correct | p-value |",
        "|---|---|---|---|---|---|",
    ]
    for t in stats["paired_tests"]:
        lines.append(
            f"| {t['split']} | {t['a']} | {t['b']} | {t['a_only']} | {t['b_only']} | {t['p_value']:.3f} |"
        )
    lines.append("")
    (RES / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    for kind, h in histories.items():
        if h:
            plot_history(h, FIG / f"loss_curves_{kind}_wise.png")


if __name__ == "__main__":
    main()
