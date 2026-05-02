"""Data loading and saving helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from utils.project_paths import find_project_root, resolve_project_path


def _default_project_root() -> Path:
    return find_project_root(Path(__file__).resolve())


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_array(path: str | Path, array: np.ndarray) -> Path:
    target = ensure_parent(resolve_project_path(path, root=_default_project_root()))
    np.save(target, np.asarray(array))
    return target


def load_array(path: str | Path) -> np.ndarray:
    return np.load(resolve_project_path(path, root=_default_project_root()))


def save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    target = ensure_parent(resolve_project_path(path, root=_default_project_root()))
    np.savez_compressed(target, **arrays)
    return target


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    loaded = np.load(resolve_project_path(path, root=_default_project_root()))
    return {key: loaded[key] for key in loaded.files}


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = ensure_parent(resolve_project_path(path, root=_default_project_root()))
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target


def save_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    target = ensure_parent(resolve_project_path(path, root=_default_project_root()))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target
