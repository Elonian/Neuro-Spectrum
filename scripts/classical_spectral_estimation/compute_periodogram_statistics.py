"""Compute periodogram estimates and Monte Carlo statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_configs
from utils.data_io import load_npz, save_npz
from utils.logging_setup import setup_logger
from utils.project_paths import ensure_project_directories, resolve_project_path
from utils.random_process import generate_process_realizations
from utils.spectral_estimators import periodogram, periodogram_statistics
from visualization.spectral_plots import plot_spectrum_comparison, plot_statistic_curves


def main() -> None:
    config = load_yaml_configs(
        [
            "configs/system_iir_process.yml",
            "configs/classical_reference_process.yml",
            "configs/classical_periodogram_monte_carlo.yml",
        ]
    )
    ensure_project_directories(PROJECT_ROOT)
    logger = setup_logger("compute_periodogram_statistics")

    reference = load_npz(resolve_project_path(config["paths"]["reference_process_npz"], root=PROJECT_ROOT))
    process_samples = reference["process_samples"]
    omega = reference["omega"]
    true_psd = reference["true_psd"]

    sample_length = int(config["project"]["sample_length"])
    n_fft = int(config["project"]["frequency_grid_size"])
    data_lengths = [int(value) for value in config["project"]["data_lengths"]]
    numerator = config["system"]["numerator"]
    denominator = config["system"]["denominator"]
    noise_variance = float(config["system"]["noise_variance"])
    burnin = int(config["system"].get("burnin_samples", 0))
    num_realizations = int(config["monte_carlo"]["num_realizations"])
    seed = int(config["monte_carlo"]["seed"])

    _, monte_carlo_y = generate_process_realizations(
        sample_length,
        num_realizations=num_realizations,
        numerator=numerator,
        denominator=denominator,
        noise_variance=noise_variance,
        burnin=burnin,
        seed=seed,
    )
    realizations_path = resolve_project_path(config["paths"]["monte_carlo_realizations_npz"], root=PROJECT_ROOT)
    save_npz(realizations_path, process_realizations=monte_carlo_y)
    logger.info("Saved %d Monte Carlo process realizations to %s", num_realizations, realizations_path)

    single_estimates = {
        length: periodogram(process_samples[:length], n_fft=n_fft) for length in data_lengths
    }
    stats = periodogram_statistics(monte_carlo_y, data_lengths=data_lengths, n_fft=n_fft)

    arrays = {
        "omega": omega,
        "true_psd": true_psd,
        "data_lengths": np.asarray(data_lengths, dtype=int),
    }
    for length in data_lengths:
        arrays[f"single_estimate_{length}"] = single_estimates[length]
        arrays[f"mean_{length}"] = stats[length]["mean"]
        arrays[f"variance_{length}"] = stats[length]["variance"]

    output_path = resolve_project_path(config["paths"]["periodogram_results_npz"], root=PROJECT_ROOT)
    save_npz(output_path, **arrays)
    logger.info("Saved periodogram statistics to %s", output_path)

    figures_dir = Path(config["paths"]["figures_dir"])
    try:
        plot_spectrum_comparison(
            resolve_project_path(figures_dir / "periodogram_single_realization.png", root=PROJECT_ROOT),
            omega=omega,
            true_psd=true_psd,
            estimates_by_length=single_estimates,
            title="Periodogram Estimates From One Realization",
        )
        plot_statistic_curves(
            resolve_project_path(figures_dir / "periodogram_monte_carlo_mean.png", root=PROJECT_ROOT),
            omega=omega,
            statistics_by_length=stats,
            statistic_key="mean",
            title="Periodogram Monte Carlo Mean",
            ylabel="mean PSD estimate",
        )
        plot_statistic_curves(
            resolve_project_path(figures_dir / "periodogram_monte_carlo_variance.png", root=PROJECT_ROOT),
            omega=omega,
            statistics_by_length=stats,
            statistic_key="variance",
            title="Periodogram Monte Carlo Variance",
            ylabel="variance",
        )
        logger.info("Saved periodogram figures under %s", resolve_project_path(figures_dir, root=PROJECT_ROOT))
    except ImportError as exc:
        logger.warning("Skipping plots because matplotlib is unavailable: %s", exc)


if __name__ == "__main__":
    main()
