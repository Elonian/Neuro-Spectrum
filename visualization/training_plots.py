"""Plots for neural training logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(
    path: str | Path,
    *,
    losses: list[float] | np.ndarray,
    title: str = "Training Loss",
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    loss_values = np.asarray(losses, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(loss_values) + 1), loss_values, linewidth=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_train_test_loss(
    path: str | Path,
    *,
    train_loss: list[float] | np.ndarray,
    test_loss: list[float] | np.ndarray,
    title: str = "Train and Test Loss",
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    train_values = np.asarray(train_loss, dtype=float)
    test_values = np.asarray(test_loss, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(train_values) + 1), train_values, linewidth=1.4, label="train")
    ax.plot(np.arange(1, len(test_values) + 1), test_values, linewidth=1.4, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target
