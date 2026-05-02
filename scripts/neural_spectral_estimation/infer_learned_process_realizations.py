"""Generate new x realizations and infer y_ML with trained MLP/LSTM checkpoints."""

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
from utils.data_io import save_array
from utils.logging_setup import setup_logger
from utils.model_training import load_model_from_checkpoint, predict_windows, select_device
from utils.neural_windows import build_inference_windows_numpy
from utils.project_paths import ensure_project_directories, resolve_project_path
from utils.random_process import generate_white_noise


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
    parser = argparse.ArgumentParser(description="Infer learned y_ML realizations from new x samples")
    parser.add_argument("--lag", type=int, default=None, help="override selected lag from YAML")
    args = parser.parse_args()

    config = load_yaml_configs(
        [
            "configs/neural_data_generation.yml",
            "configs/neural_configuration_search.yml",
            "configs/neural_inference_analysis.yml",
        ]
    )
    ensure_project_directories(PROJECT_ROOT)
    logger = setup_logger("infer_learned_process_realizations")

    output_length = int(config["project"]["sample_length"])
    num_realizations = int(config["project"]["inference_realizations"])
    seed = int(config["inference"]["seed"])
    device = select_device(str(config["inference"]["device"]))
    batch_size = int(config["inference"]["batch_size"])

    model_dir = resolve_project_path(config["paths"]["model_dir"], root=PROJECT_ROOT)
    prediction_dir = resolve_project_path(config["paths"]["prediction_dir"], root=PROJECT_ROOT)
    input_dir = resolve_project_path(config["paths"]["inference_input_dir"], root=PROJECT_ROOT)

    for index, (model_name, lag) in enumerate(_selected_models(config, args.lag)):
        x_new = generate_white_noise(
            output_length + lag,
            num_realizations=num_realizations,
            seed=seed + index + lag,
        ).astype(np.float32)
        x_path = input_dir / f"x_new_{model_name}_lag_{lag}.npy"
        save_array(x_path, x_new)
        logger.info("Saved inference inputs shape=%s to %s", x_new.shape, x_path)

        windows = build_inference_windows_numpy(x_new, lag=lag, output_length=output_length)
        checkpoint = model_dir / f"simple_{model_name}_lag_{lag}.pt"
        if not checkpoint.exists():
            logger.warning("Skipping %s inference; checkpoint not found: %s", model_name, checkpoint)
            continue
        model, _ = load_model_from_checkpoint(checkpoint, device=device)
        flat_prediction = predict_windows(model, windows, device=device, batch_size=batch_size)
        prediction = flat_prediction.reshape(num_realizations, output_length)
        output_path = prediction_dir / f"y_predict_{model_name}_lag_{lag}.npy"
        save_array(output_path, prediction)
        logger.info("Saved %s predictions shape=%s to %s", model_name, prediction.shape, output_path)


if __name__ == "__main__":
    main()
