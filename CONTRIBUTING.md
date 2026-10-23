# Contributing

## Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows  (Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # match your CUDA driver
pip install -r requirements.txt
pip install -e . --no-deps
copy configs\local.example.yaml configs\local.yaml   # then set dataset_root
```

Verify the GPU: `python -c "import torch; print(torch.cuda.is_available())"`.

## Code style

* Python ≥ 3.10, type hints and docstrings on public functions.
* Formatting and linting with **ruff** (line length 110):
  `python -m ruff format src scripts tests app` and `python -m ruff check src scripts tests app`.
* Paths via `pathlib`; no machine-specific paths in source; scripts guard their entry point with
  `if __name__ == "__main__":` (required for multi-worker DataLoaders on Windows).
* No magic numbers: put tunables in `configs/default.yaml` or a named module constant.

## Tests

* `python -m pytest -q` must pass before every commit.
* New geometry, transform, decoding or metric code needs a hand-checked unit test.
* Tests marked `dataset` use the real Cornell copy and are skipped if it is not configured.
* Synthetic fixtures must never be used to claim performance.

## Dataset handling

* The dataset folder is **read-only**. Derived data goes to `outputs/cache/` (git-ignored).
* Never commit images, depth maps, point clouds, checkpoints or `configs/local.yaml`.

## Experiment conventions

* One YAML per experiment in `configs/experiment_configs/`, inheriting `configs/default.yaml`.
* Runs write to `outputs/runs/<experiment.name>/` (config copy, env, log, history, checkpoints).
* Model selection uses the **validation** split only. `scripts/evaluate.py --split test` is run once
  per selected model after selection is frozen; record it in `docs/experiment_log.md`.
* Commit the small result files (`experiments/results/*.json|csv|md`) and figures used by the report.

## Branches, commits and pull requests

* `main` is always green. Work on `feature/<topic>` or `fix/<topic>` branches.
* Commit messages: imperative subject ≤ 72 chars, body explaining *why*.
* Review `git status` and the staged diff before each commit.
* A PR must state what changed, how it was tested (commands + outcomes), and update the docs
  (`README.md`, `docs/technical_report.md`, `CHANGELOG.md`, `PROJECT_STATUS.md`) when behaviour or
  results change. Every number quoted in docs must come from a committed result file.
