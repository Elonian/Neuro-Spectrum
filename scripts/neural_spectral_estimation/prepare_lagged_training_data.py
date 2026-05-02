"""Generate x.npy/y.npy training pairs for each configured lag value."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_configs
from utils.data_io import save_array
from utils.logging_setup import setup_logger
from utils.project_paths import ensure_project_directories, resolve_project_path
from utils.random_process import generate_aligned_training_pair


def main() -> None:
    config = load_yaml_configs(
        [
            "configs/system_iir_process.yml",
            "configs/neural_data_generation.yml",
        ]
    )
    ensure_project_directories(PROJECT_ROOT)
    logger = setup_logger("prepare_lagged_training_data")

    output_length = int(config["project"]["sample_length"])
    base_seed = int(config["project"]["seed"])
    lag_values = [int(value) for value in config["lag_experiments"]["values"]]
    training_root = resolve_project_path(config["paths"]["training_data_dir"], root=PROJECT_ROOT)

    for lag in lag_values:
        x, y = generate_aligned_training_pair(
            output_length,
            lag,
            numerator=config["system"]["numerator"],
            denominator=config["system"]["denominator"],
            noise_variance=float(config["system"]["noise_variance"]),
            burnin=int(config["system"].get("burnin_samples", 0)),
            seed=base_seed + lag,
        )
        lag_dir = training_root / f"lag_{lag}"
        save_array(lag_dir / "x.npy", x)
        save_array(lag_dir / "y.npy", y)
        logger.info("Saved lag=%d training pair to %s", lag, lag_dir)


if __name__ == "__main__":
    main()
