"""Train selected MLP and LSTM neural process generators."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_configs
from utils.data_io import load_array, save_array, save_json, save_npz
from utils.logging_setup import setup_logger
from utils.model_training import (
    build_model,
    checkpoint_payload,
    make_train_test_loaders,
    save_checkpoint,
    select_device,
    train_model_with_test_history,
)
from utils.neural_windows import LagWindowDataset
from utils.project_paths import ensure_project_directories, resolve_project_path
from visualization.training_plots import plot_train_test_loss


def _load_selected(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["selected"]


def _training_config_for_result(result: dict[str, Any]) -> dict[str, Any]:
    if result["model_type"] == "mlp":
        return {
            "hidden": int(result["hidden"]),
            "num_hidden_layers": int(result["num_hidden_layers"]),
        }
    if result["model_type"] == "lstm":
        return {
            "hidden": int(result["hidden"]),
            "num_layers": int(result["num_layers"]),
        }
    raise ValueError(f"Unsupported model type {result['model_type']}")


def main() -> None:
    config = load_yaml_configs(
        [
            "configs/neural_data_generation.yml",
            "configs/neural_configuration_search.yml",
        ]
    )
    ensure_project_directories(PROJECT_ROOT)
    logger = setup_logger("train_selected_neural_generators")
    selected_path = resolve_project_path(config["paths"]["selected_config_json"], root=PROJECT_ROOT)
    selected = _load_selected(selected_path)
    device = select_device("auto")

    for model_type, result in selected.items():
        lag = int(result["lag"])
        data_dir = resolve_project_path(config["paths"]["training_data_dir"], root=PROJECT_ROOT) / f"lag_{lag}"
        x = load_array(data_dir / "x.npy")
        y = load_array(data_dir / "y.npy")
        dataset = LagWindowDataset(x, y, lag)
        seed = int(config["project"]["seed"]) + lag + int(result["hidden"])
        train_loader, test_loader = make_train_test_loaders(
            dataset,
            batch_size=int(config["selected_training"]["batch_size"]),
            test_fraction=float(config["selected_training"]["test_fraction"]),
            seed=seed,
        )

        torch.manual_seed(seed)
        np.random.seed(seed)
        model_config = _training_config_for_result(result)
        learning_rate = float(result.get("learning_rate", config["selected_training"]["learning_rate"]))
        model = build_model(model_type, lag=lag, config=model_config)
        logger.info(
            "Training selected %s lag=%d config=%s learning_rate=%.6g",
            model_type,
            lag,
            model_config,
            learning_rate,
        )
        history = train_model_with_test_history(
            model,
            train_loader,
            test_loader,
            device=device,
            learning_rate=learning_rate,
            epochs=int(config["selected_training"]["epochs"]),
            log_every=100,
            logger=logger,
        )
        best_state = history["best_state_dict"][0]
        if best_state is not None:
            model.load_state_dict(best_state)

        model_dir = resolve_project_path("outputs/models", root=PROJECT_ROOT)
        checkpoint_path = model_dir / f"simple_{model_type}_lag_{lag}.pt"
        payload = checkpoint_payload(
            model_type,
            model,
            lag=lag,
            model_config=model_config,
            losses=history["train_loss"],
        )
        payload["test_losses"] = history["test_loss"]
        save_checkpoint(checkpoint_path, payload)
        save_array(
            model_dir / f"simple_{model_type}_lag_{lag}_train_losses.npy",
            np.asarray(history["train_loss"], dtype=np.float32),
        )
        save_array(
            model_dir / f"simple_{model_type}_lag_{lag}_test_losses.npy",
            np.asarray(history["test_loss"], dtype=np.float32),
        )
        save_npz(
            model_dir / f"simple_{model_type}_lag_{lag}_loss_history.npz",
            train_loss=np.asarray(history["train_loss"], dtype=np.float32),
            test_loss=np.asarray(history["test_loss"], dtype=np.float32),
        )
        save_json(
            model_dir / f"simple_{model_type}_lag_{lag}_summary.json",
            {
                "model": model_type,
                "lag": lag,
                "selected_from_search_train_mse": float(result["final_train_mse"]),
                "selected_from_search_test_mse": float(result["final_test_mse"]),
                "selected_from_search_best_test_mse": float(result.get("best_test_mse", result["final_test_mse"])),
                "learning_rate": learning_rate,
                "final_train_mse": float(history["train_loss"][-1]),
                "final_test_mse": float(history["test_loss"][-1]),
                "saved_checkpoint_epoch": int(history["best_epoch"][0]),
                "saved_checkpoint_test_mse": float(history["best_test_loss"][0]),
                "checkpoint": str(checkpoint_path),
            },
        )
        logger.info(
            "Saved selected %s checkpoint to %s from epoch=%d test_mse=%.6e",
            model_type,
            checkpoint_path,
            int(history["best_epoch"][0]),
            float(history["best_test_loss"][0]),
        )

        try:
            plot_train_test_loss(
                resolve_project_path(
                    Path("outputs/figures/neural_training_loss") / f"selected_{model_type}_lag_{lag}_train_test_loss.png",
                    root=PROJECT_ROOT,
                ),
                train_loss=history["train_loss"],
                test_loss=history["test_loss"],
                title=f"Selected {model_type.upper()} train and test loss, lag={lag}",
            )
        except ImportError as exc:
            logger.warning("Skipping training loss plot because matplotlib is unavailable: %s", exc)


if __name__ == "__main__":
    main()
