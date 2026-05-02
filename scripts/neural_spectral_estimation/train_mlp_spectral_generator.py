"""Train the supplied SimpleMLP model as a learned spectral process generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_configs
from utils.data_io import load_array, save_array, save_json
from utils.logging_setup import setup_logger
from utils.model_training import build_model, checkpoint_payload, save_checkpoint, select_device, train_model
from utils.neural_windows import LagWindowDataset
from utils.project_paths import ensure_project_directories, resolve_project_path
from utils.random_process import generate_aligned_training_pair
from visualization.training_plots import plot_training_loss


def _ensure_training_data(config: dict, lag: int, logger) -> tuple[np.ndarray, np.ndarray, Path]:
    data_dir = resolve_project_path(config["paths"]["training_data_dir"], root=PROJECT_ROOT) / f"lag_{lag}"
    x_path = data_dir / "x.npy"
    y_path = data_dir / "y.npy"
    if x_path.exists() and y_path.exists():
        return load_array(x_path), load_array(y_path), data_dir

    logger.info("Training data for lag=%d was missing; generating it now", lag)
    x, y = generate_aligned_training_pair(
        int(config["project"]["sample_length"]),
        lag,
        numerator=config["system"]["numerator"],
        denominator=config["system"]["denominator"],
        noise_variance=float(config["system"]["noise_variance"]),
        burnin=int(config["system"].get("burnin_samples", 0)),
        seed=int(config["project"]["seed"]) + lag,
    )
    save_array(x_path, x)
    save_array(y_path, y)
    return x, y, data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SimpleMLP on lagged x/y data")
    parser.add_argument("--lag", type=int, default=None, help="override selected lag from YAML")
    parser.add_argument("--epochs", type=int, default=None, help="override training epochs from YAML")
    parser.add_argument("--learning-rate", type=float, default=None, help="override learning rate from YAML")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch size from YAML")
    args = parser.parse_args()

    config = load_yaml_configs(
        [
            "configs/system_iir_process.yml",
            "configs/neural_data_generation.yml",
            "configs/neural_mlp_training.yml",
        ]
    )
    ensure_project_directories(PROJECT_ROOT)
    logger = setup_logger("train_mlp_spectral_generator")

    training = config["training"]
    lag = int(args.lag if args.lag is not None else training.get("lag", config["lag_experiments"]["selected_lag"]))
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.learning_rate is not None:
        training["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    torch.manual_seed(int(training["seed"]))
    np.random.seed(int(training["seed"]))

    x, y, data_dir = _ensure_training_data(config, lag, logger)
    dataset = LagWindowDataset(x, y, lag)
    batch_size = int(training["batch_size"])
    effective_batch_size = len(dataset) if batch_size <= 0 else min(batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=effective_batch_size < len(dataset),
        drop_last=False,
    )

    device = select_device(str(training["device"]))
    model_config = {
        "hidden": int(training["hidden"]),
        "num_hidden_layers": int(training["num_hidden_layers"]),
    }
    model = build_model("mlp", lag=lag, config=model_config)
    logger.info("Training MLP lag=%d data_dir=%s device=%s", lag, data_dir, device)
    losses = train_model(
        model,
        loader,
        device=device,
        learning_rate=float(training["learning_rate"]),
        epochs=int(training["epochs"]),
        log_every=int(training["log_every"]),
        logger=logger,
    )

    model_dir = resolve_project_path(config["paths"]["model_dir"], root=PROJECT_ROOT)
    checkpoint_path = model_dir / f"simple_mlp_lag_{lag}.pt"
    save_checkpoint(
        checkpoint_path,
        checkpoint_payload("mlp", model, lag=lag, model_config=model_config, losses=losses),
    )
    loss_path = model_dir / f"simple_mlp_lag_{lag}_losses.npy"
    save_array(loss_path, np.asarray(losses, dtype=np.float32))
    save_json(
        model_dir / f"simple_mlp_lag_{lag}_summary.json",
        {
            "model": "mlp",
            "lag": lag,
            "final_train_mse": float(losses[-1]),
            "epochs": int(training["epochs"]),
            "data_dir": str(data_dir),
            "checkpoint": str(checkpoint_path),
        },
    )
    logger.info("Saved MLP checkpoint to %s", checkpoint_path)

    try:
        plot_training_loss(
            resolve_project_path(
                Path(config["paths"]["figures_dir"]) / f"simple_mlp_lag_{lag}_training_loss.png",
                root=PROJECT_ROOT,
            ),
            losses=losses,
            title=f"SimpleMLP Training Loss, lag={lag}",
        )
    except ImportError as exc:
        logger.warning("Skipping training loss plot because matplotlib is unavailable: %s", exc)


if __name__ == "__main__":
    main()
