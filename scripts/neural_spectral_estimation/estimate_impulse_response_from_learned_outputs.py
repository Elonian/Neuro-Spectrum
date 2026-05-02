"""Estimate system impulse response from x_new and learned y_ML outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_configs
from utils.data_io import load_array, save_npz
from utils.logging_setup import setup_logger
from utils.project_paths import ensure_project_directories, resolve_project_path
from utils.random_process import estimate_impulse_response_from_io, impulse_response
from visualization.spectral_plots import plot_impulse_response_comparison


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
    parser = argparse.ArgumentParser(description="Estimate h[n] from sample cross correlation")
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
    logger = setup_logger("estimate_impulse_response_from_learned_outputs")

    max_lag = int(config["analysis"]["impulse_response_max_lag"])
    input_dir = resolve_project_path(config["paths"]["inference_input_dir"], root=PROJECT_ROOT)
    prediction_dir = resolve_project_path(config["paths"]["prediction_dir"], root=PROJECT_ROOT)
    analysis_dir = resolve_project_path(config["paths"]["analysis_dir"], root=PROJECT_ROOT)
    figures_dir = Path(config["paths"]["figures_dir"])

    true_h = impulse_response(
        max_lag + 1,
        numerator=config["system"]["numerator"],
        denominator=config["system"]["denominator"],
    )

    for model_name, lag in _selected_models(config, args.lag):
        x_path = input_dir / f"x_new_{model_name}_lag_{lag}.npy"
        if not x_path.exists():
            logger.warning("Skipping impulse response estimate for %s; missing %s", model_name, x_path)
            continue
        x_new = load_array(x_path)
        prediction_path = prediction_dir / f"y_predict_{model_name}_lag_{lag}.npy"
        if not prediction_path.exists():
            logger.warning("Skipping impulse response estimate for %s; missing %s", model_name, prediction_path)
            continue

        y_ml = load_array(prediction_path)
        lags, estimated_h = estimate_impulse_response_from_io(
            x_new,
            y_ml,
            max_lag,
            input_pre_history=lag,
            input_variance=float(config["system"]["noise_variance"]),
        )
        output_path = analysis_dir / f"impulse_response_estimate_{model_name}_lag_{lag}.npz"
        save_npz(output_path, lags=lags, estimated_impulse_response=estimated_h, true_impulse_response=true_h)
        logger.info("Saved impulse response estimate for %s to %s", model_name, output_path)

        try:
            plot_impulse_response_comparison(
                resolve_project_path(
                    figures_dir / f"impulse_response_{model_name}_lag_{lag}.png",
                    root=PROJECT_ROOT,
                ),
                true_impulse=true_h,
                estimated_lags=lags,
                estimated_impulse=estimated_h,
            )
        except ImportError as exc:
            logger.warning("Skipping impulse response plot because matplotlib is unavailable: %s", exc)


if __name__ == "__main__":
    main()
