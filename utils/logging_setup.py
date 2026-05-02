"""File and console logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from utils.project_paths import ensure_project_directories, find_project_root


def setup_logger(name: str, *, log_dir: str | Path | None = None) -> logging.Logger:
    root = find_project_root(Path(__file__).resolve())
    directories = ensure_project_directories(root)
    target_dir = Path(log_dir) if log_dir is not None else directories["logs"]
    if not target_dir.is_absolute():
        target_dir = root / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    log_path = target_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Log file: %s", log_path)
    return logger
