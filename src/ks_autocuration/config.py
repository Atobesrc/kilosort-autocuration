"""Configuration loading: default.yaml overlaid with an optional user YAML."""
from __future__ import annotations

import copy
from importlib import resources
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(user_config: str | Path | None = None) -> dict:
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if user_config is not None:
        with open(user_config, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_update(cfg, override)
    return cfg


def dump_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
