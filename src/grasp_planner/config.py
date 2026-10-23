"""Configuration loading.

Experiment configs are YAML files. A config may name a ``base`` config (path relative to the
repository root) whose values it overrides recursively, so experiment files only list what
changes. The machine-specific dataset location is *not* stored in experiment configs: it is
resolved by :func:`resolve_dataset_root` from ``GRASP_DATASET_ROOT`` or ``configs/local.yaml``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = REPO_ROOT / "configs" / "local.yaml"
ENV_VAR = "GRASP_DATASET_ROOT"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` recursively updated with ``override``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config, resolving its optional ``base`` chain."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_update(load_config(REPO_ROOT / base), cfg)
    return cfg


def resolve_dataset_root(explicit: str | Path | None = None, local_config: Path = LOCAL_CONFIG) -> Path:
    """Find the Cornell dataset root.

    Priority: explicit argument > ``GRASP_DATASET_ROOT`` environment variable >
    ``dataset_root`` in ``configs/local.yaml``. Raises :class:`ConfigError` with setup
    instructions if none is set or the folder does not exist.
    """
    candidate: str | Path | None = explicit or os.environ.get(ENV_VAR)
    if not candidate and local_config.exists():
        with local_config.open("r", encoding="utf-8") as f:
            candidate = (yaml.safe_load(f) or {}).get("dataset_root")
    if not candidate:
        raise ConfigError(
            f"Dataset path not configured. Set {ENV_VAR} or copy configs/local.example.yaml "
            "to configs/local.yaml and set dataset_root."
        )
    root = Path(candidate).expanduser()
    if not root.is_dir():
        raise ConfigError(f"Configured dataset root does not exist: {root}")
    return root
