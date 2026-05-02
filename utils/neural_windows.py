"""Lag window data helpers for the supplied MLP and LSTM models."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def build_lag_windows_numpy(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float32).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
    if lag < 0:
        raise ValueError("lag must not be negative")
    if x_arr.size != y_arr.size + lag:
        raise ValueError(
            f"expected len(x)==len(y)+lag, got len(x)={x_arr.size}, "
            f"len(y)={y_arr.size}, lag={lag}"
        )
    indices = np.arange(y_arr.size)
    columns = [x_arr[indices + lag - offset] for offset in range(lag + 1)]
    return np.stack(columns, axis=1), y_arr


def build_inference_windows_numpy(x_new: np.ndarray, lag: int, output_length: int) -> np.ndarray:
    x_arr = np.asarray(x_new, dtype=np.float32)
    required_length = output_length + lag
    if x_arr.ndim == 1:
        if x_arr.size != required_length:
            raise ValueError(f"expected len(x_new)=={required_length}, got {x_arr.size}")
        indices = np.arange(output_length)
        columns = [x_arr[indices + lag - offset] for offset in range(lag + 1)]
        return np.stack(columns, axis=1)

    if x_arr.ndim != 2:
        raise ValueError(f"x_new must be 1D or 2D, got shape {x_arr.shape}")
    if x_arr.shape[1] == required_length:
        batch = x_arr
    elif x_arr.shape[0] == required_length:
        batch = x_arr.T
    else:
        raise ValueError(
            f"expected x_new shape (B,{required_length}) or ({required_length},B), "
            f"got {x_arr.shape}"
        )
    indices = np.arange(output_length)
    columns = [batch[:, indices + lag - offset] for offset in range(lag + 1)]
    windows = np.stack(columns, axis=2)
    return windows.reshape(batch.shape[0] * output_length, lag + 1)


class LagWindowDataset(Dataset):

    def __init__(self, x: np.ndarray, y: np.ndarray, lag: int) -> None:
        windows, targets = build_lag_windows_numpy(x, y, lag)
        self.windows = torch.from_numpy(windows)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[index], self.targets[index]
