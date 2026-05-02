"""Repeat periodogram mean/variance analysis using learned y_ML realizations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_configs
from utils.data_io import load_array, save_npz
from utils.logging_setup import setup_logger
from utils.project_paths import ensure_project_directories, resolve_project_path
from utils.random_process import true_power_spectrum
from utils.spectral_estimators import periodogram_statistics
from visualization.spectral_plots import plot_statistic_curves


def _selected_models(config: dict, override_lag: int | None) -> list[tuple[str, int]]:
    if override_lag is not None:
        return [("mlp", override_lag), ("lstm", override_lag)]
    selected_path = resolve_project_path(config["paths"]["selected_config_json"], root=PROJECT_ROOT)
    if selected_path.exists():
        with selected_path.open("r", encoding="utf-8") as handle:
            selected = json.load(handle)["selected"]
        return [(model_type, int(result["lag"])) for model_type, result in selected.items()]
    lag = int(config["lag_experiments"]["selected_lag"])
    return [("mlp", lag), ("lstm", lag)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze learned periodogram statistics")
    parser.add_argument("--lag", type=int, default=None, help="override selected lag from YAML")
    args = parser.parse_args()

    config = load_yaml_configs(
        [
            "configs/system_iir_process.yml",
            "configs/neural_data_generation.yml",
            "configs/neural_configuration_search.yml",
            "configs/neural_inference_analysis.yml",
        ]
    )
    ensure_project_directories(PROJECT_ROOT)
    logger = setup_logger("analyze_learned_periodogram_statistics")

    data_lengths = [int(value) for value in config["project"]["data_lengths"]]
    n_fft = int(config["project"]["frequency_grid_size"])
    omega, true_psd = true_power_spectrum(
        n_fft,
        numerator=config["system"]["numerator"],
        denominator=config["system"]["denominator"],
        noise_variance=float(config["system"]["noise_variance"]),
    )

    prediction_dir = resolve_project_path(config["paths"]["prediction_dir"], root=PROJECT_ROOT)
    analysis_dir = resolve_project_path(config["paths"]["analysis_dir"], root=PROJECT_ROOT)
    figures_dir = Path(config["paths"]["figures_dir"])

    for model_name, lag in _selected_models(config, args.lag):
        prediction_path = prediction_dir / f"y_predict_{model_name}_lag_{lag}.npy"
        if not prediction_path.exists():
            logger.warning("Skipping %s analysis; missing %s", model_name, prediction_path)
            continue

        predictions = load_array(prediction_path)
        stats = periodogram_statistics(predictions, data_lengths=data_lengths, n_fft=n_fft)
        arrays = {
            "omega": omega,
            "true_psd": true_psd,
            "data_lengths": np.asarray(data_lengths, dtype=int),
        }
        for length in data_lengths:
            arrays[f"mean_{length}"] = stats[length]["mean"]
            arrays[f"variance_{length}"] = stats[length]["variance"]
        output_path = analysis_dir / f"periodogram_statistics_{model_name}_lag_{lag}.npz"
        save_npz(output_path, **arrays)
        logger.info("Saved learned periodogram statistics for %s to %s", model_name, output_path)

        try:
            plot_statistic_curves(
                resolve_project_path(
                    figures_dir / f"learned_periodogram_mean_{model_name}_lag_{lag}.png",
                    root=PROJECT_ROOT,
                ),
                omega=omega,
                statistics_by_length=stats,
                statistic_key="mean",
                title=f"Learned Periodogram Mean, {model_name.upper()}, lag={lag}",
                ylabel="mean PSD estimate",
            )
            plot_statistic_curves(
                resolve_project_path(
                    figures_dir / f"learned_periodogram_variance_{model_name}_lag_{lag}.png",
                    root=PROJECT_ROOT,
                ),
                omega=omega,
                statistics_by_length=stats,
                statistic_key="variance",
                title=f"Learned Periodogram Variance, {model_name.upper()}, lag={lag}",
                ylabel="variance",
            )
        except ImportError as exc:
            logger.warning("Skipping learned statistic plots because matplotlib is unavailable: %s", exc)


if __name__ == "__main__":
    main()
