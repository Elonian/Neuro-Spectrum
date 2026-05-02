"""Training and inference helpers for the supplied neural models."""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from modules.Model_LSTM import SimpleLSTM
from modules.Model_MLP import SimpleMLP
from utils.data_io import ensure_parent


def select_device(preference: str = "auto") -> torch.device:
    normalized = preference.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is false")
    return torch.device(normalized)


def build_model(model_type: str, *, lag: int, config: dict[str, Any]) -> nn.Module:
    name = model_type.lower()
    if name == "mlp":
        return SimpleMLP(
            L=lag,
            hidden=int(config.get("hidden", 64)),
            num_hidden_layers=int(config.get("num_hidden_layers", 2)),
        )
    if name == "lstm":
        return SimpleLSTM(
            L=lag,
            hidden=int(config.get("hidden", 64)),
            num_layers=int(config.get("num_layers", 1)),
        )
    raise ValueError(f"Unsupported model_type '{model_type}'")


def train_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    learning_rate: float,
    epochs: int,
    log_every: int = 100,
    logger: Any | None = None,
) -> list[float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses: list[float] = []
    start = time.perf_counter()
    model.train()

    for epoch in range(1, epochs + 1):
        running = 0.0
        batches = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            batches += 1

        epoch_loss = running / max(batches, 1)
        losses.append(epoch_loss)
        if logger is not None and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            elapsed = time.perf_counter() - start
            logger.info(
                "epoch=%d/%d train_mse=%.6e elapsed_seconds=%.1f",
                epoch,
                epochs,
                epoch_loss,
                elapsed,
            )
    return losses


def make_train_test_loaders(
    dataset: Dataset,
    *,
    batch_size: int,
    test_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    total = len(dataset)
    test_count = max(1, int(round(total * test_fraction)))
    train_count = total - test_count
    if train_count <= 0:
        raise ValueError("test_fraction leaves no train samples")
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [train_count, test_count], generator=generator)
    train_batch_size = train_count if batch_size <= 0 else min(batch_size, train_count)
    test_batch_size = test_count if batch_size <= 0 else min(batch_size, test_count)
    return (
        DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=train_batch_size < train_count,
            drop_last=False,
        ),
        DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            drop_last=False,
        ),
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> float:
    criterion = nn.MSELoss(reduction="sum")
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            total_loss += float(criterion(predictions, targets).detach().cpu())
            total_count += int(targets.numel())
    return total_loss / max(total_count, 1)


def train_model_with_test_history(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    learning_rate: float,
    epochs: int,
    log_every: int = 100,
    logger: Any | None = None,
) -> dict[str, list[float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "test_loss": []}
    best_state = None
    best_epoch = 0
    best_test_loss = float("inf")
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            batches += 1

        train_loss = running / max(batches, 1)
        test_loss = evaluate_model(model, test_loader, device=device)
        if not np.isfinite(train_loss) or not np.isfinite(test_loss):
            train_loss = float(train_loss) if np.isfinite(train_loss) else float("inf")
            test_loss = float(test_loss) if np.isfinite(test_loss) else float("inf")
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            if logger is not None:
                logger.warning(
                    "stopping early at epoch=%d/%d train_mse=%.6e test_mse=%.6e",
                    epoch,
                    epochs,
                    train_loss,
                    test_loss,
                )
            break
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        if logger is not None and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            elapsed = time.perf_counter() - start
            logger.info(
                "epoch=%d/%d train_mse=%.6e test_mse=%.6e elapsed_seconds=%.1f",
                epoch,
                epochs,
                train_loss,
                test_loss,
                elapsed,
            )
    history["best_epoch"] = [float(best_epoch)]
    history["best_test_loss"] = [best_test_loss]
    history["best_state_dict"] = [best_state]
    return history


def checkpoint_payload(
    model_type: str,
    model: nn.Module,
    *,
    lag: int,
    model_config: dict[str, Any],
    losses: list[float],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "model_type": model_type.lower(),
        "L": int(lag),
        "lag": int(lag),
        "losses": losses,
    }
    payload.update({key: value for key, value in model_config.items() if isinstance(value, (int, float, str))})
    return payload


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    target = ensure_parent(path)
    torch.save(payload, target)
    return target


def load_model_from_checkpoint(path: str | Path, *, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device)
    lag = int(payload.get("lag", payload["L"]))
    model_type = str(payload.get("model_type", "mlp" if "num_hidden_layers" in payload else "lstm"))
    if model_type == "mlp":
        config = {
            "hidden": int(payload["hidden"]),
            "num_hidden_layers": int(payload["num_hidden_layers"]),
        }
    elif model_type == "lstm":
        config = {
            "hidden": int(payload["hidden"]),
            "num_layers": int(payload["num_layers"]),
        }
    else:
        raise ValueError(f"Unsupported checkpoint model_type '{model_type}'")
    model = build_model(model_type, lag=lag, config=config)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, payload


def predict_windows(
    model: nn.Module,
    windows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(np.asarray(windows, dtype=np.float32))
    predictions = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = x[start : start + batch_size].to(device)
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)
