"""Project path helpers used by scripts and utilities."""

from __future__ import annotations

from pathlib import Path


PROJECT_MARKERS = ("configs", "papers", "scripts", "utils")


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    raise FileNotFoundError(
        f"Could not find project root from {current}; expected markers {PROJECT_MARKERS}"
    )


def resolve_project_path(path: str | Path, *, root: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or find_project_root()) / p


def ensure_project_directories(root: Path | None = None) -> dict[str, Path]:
    project_root = root or find_project_root()
    directories = {
        "configs": project_root / "configs",
        "data": project_root / "data",
        "logs": project_root / "logs",
        "outputs": project_root / "outputs",
        "figures": project_root / "outputs" / "figures",
        "models": project_root / "outputs" / "models",
        "generated_processes": project_root / "data" / "generated_processes",
        "neural_training": project_root / "data" / "neural_training",
        "classical_outputs": project_root / "outputs" / "classical_spectral_estimation",
        "neural_outputs": project_root / "outputs" / "neural_spectral_estimation",
    }
    directories["logs"].mkdir(parents=True, exist_ok=True)
    return directories
