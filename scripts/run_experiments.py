"""Run several training configs one after another (each in its own process), logging each run
to outputs/logs/<config-stem>.log. Intended to be launched in the background.

Usage:
    python scripts/run_experiments.py configs/experiment_configs/a.yaml configs/experiment_configs/b.yaml
    python scripts/run_experiments.py --resume ...   # pass --resume to every train.py call
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    log_dir = REPO / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for cfg in args.configs:
        log = log_dir / f"{Path(cfg).stem}.log"
        cmd = [sys.executable, str(REPO / "scripts" / "train.py"), "--config", cfg]
        if args.resume:
            cmd.append("--resume")
        t0 = time.time()
        print(f"[run_experiments] START {cfg} -> {log}", flush=True)
        with log.open("a", encoding="utf-8") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
        print(f"[run_experiments] END {cfg} rc={rc} after {time.time() - t0:.0f}s", flush=True)
    print("[run_experiments] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
