"""YAML configuration loading and access helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from utils.project_paths import find_project_root, resolve_project_path


def _default_project_root() -> Path:
    return find_project_root(Path(__file__).resolve())


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path, root=_default_project_root())
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def load_yaml_configs(paths: list[str | Path] | tuple[str | Path, ...]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_update(merged, load_yaml_config(path))
    return merged


def deep_get(config: dict[str, Any], keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
