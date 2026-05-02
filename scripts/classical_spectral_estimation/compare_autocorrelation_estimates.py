"""Compare sample autocorrelation estimates against the true autocorrelation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_configs
from utils.data_io import load_npz, save_npz
from utils.logging_setup import setup_logger
from utils.project_paths import ensure_project_directories, resolve_project_path
from utils.random_process import sample_autocorrelation
from visualization.spectral_plots import plot_autocorrelation_comparison


def main() -> None:
    config = load_yaml_configs(
        [
            "configs/system_iir_process.yml",
            "configs/classical_reference_process.yml",
        ]
    )
    ensure_project_directories(PROJECT_ROOT)
    logger = setup_logger("compare_autocorrelation_estimates")

    reference_path = resolve_project_path(config["paths"]["reference_process_npz"], root=PROJECT_ROOT)
    reference = load_npz(reference_path)
    process_samples = reference["process_samples"]
    true_lags = reference["autocorrelation_lags"]
    true_autocorrelation = reference["true_autocorrelation"]
    max_lag = int(true_lags[-1])
    data_lengths = [int(value) for value in config["project"]["data_lengths"]]

    arrays = {
        "true_lags": true_lags,
        "true_autocorrelation": true_autocorrelation,
    }
    estimated_by_length = {}
    for length in data_lengths:
        lags, estimate = sample_autocorrelation(process_samples[:length], max_lag=min(max_lag, length - 1))
        arrays[f"lags_{length}"] = lags
        arrays[f"estimate_{length}"] = estimate
        estimated_by_length[length] = (lags, estimate)
        logger.info("Computed autocorrelation estimate for N=%d", length)

    output_path = resolve_project_path(config["paths"]["autocorrelation_results_npz"], root=PROJECT_ROOT)
    save_npz(output_path, **arrays)
    logger.info("Saved autocorrelation estimates to %s", output_path)

    figure_path = resolve_project_path(
        Path(config["paths"]["figures_dir"]) / "autocorrelation_comparison.png",
        root=PROJECT_ROOT,
    )
    try:
        plot_autocorrelation_comparison(
            figure_path,
            true_lags=true_lags,
            true_autocorrelation=true_autocorrelation,
            estimated_by_length=estimated_by_length,
        )
        logger.info("Saved autocorrelation figure to %s", figure_path)
    except ImportError as exc:
        logger.warning("Skipping plot because matplotlib is unavailable: %s", exc)


if __name__ == "__main__":
    main()
