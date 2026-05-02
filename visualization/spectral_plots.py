"""Plots for autocorrelation, PSD, periodogram, Welch, and impulse response."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare_output(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def plot_autocorrelation_comparison(
    path: str | Path,
    *,
    true_lags: np.ndarray,
    true_autocorrelation: np.ndarray,
    estimated_by_length: dict[int, tuple[np.ndarray, np.ndarray]],
) -> Path:
    target = _prepare_output(path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(true_lags, true_autocorrelation, color="black", linewidth=2, label="true")
    for length, (lags, estimate) in estimated_by_length.items():
        ax.plot(lags, estimate, linewidth=1.2, label=f"N={length}")
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    ax.set_title("True and Estimated Autocorrelation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_spectrum_comparison(
    path: str | Path,
    *,
    omega: np.ndarray,
    true_psd: np.ndarray,
    estimates_by_length: dict[int, np.ndarray],
    ylabel: str = "PSD",
    title: str = "True Spectrum and Estimates",
) -> Path:
    target = _prepare_output(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    frequency = omega / np.pi
    ax.plot(frequency, true_psd, color="black", linewidth=2, label="true")
    for length, estimate in estimates_by_length.items():
        ax.plot(frequency, estimate, linewidth=1.1, alpha=0.85, label=f"N={length}")
    ax.set_xlabel("normalized frequency (x pi rad/sample)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_statistic_curves(
    path: str | Path,
    *,
    omega: np.ndarray,
    statistics_by_length: dict[int, dict[str, np.ndarray]],
    statistic_key: str,
    title: str,
    ylabel: str,
) -> Path:
    target = _prepare_output(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    frequency = omega / np.pi
    for length, stats in statistics_by_length.items():
        ax.plot(frequency, stats[statistic_key], linewidth=1.1, label=f"N={length}")
    ax.set_xlabel("normalized frequency (x pi rad/sample)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_impulse_response_comparison(
    path: str | Path,
    *,
    true_impulse: np.ndarray,
    estimated_lags: np.ndarray,
    estimated_impulse: np.ndarray,
) -> Path:
    target = _prepare_output(path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stem(
        np.arange(len(true_impulse)),
        true_impulse,
        linefmt="k-",
        markerfmt="ko",
        basefmt=" ",
        label="true",
    )
    ax.plot(estimated_lags, estimated_impulse, "o-", linewidth=1.2, label="estimated")
    ax.set_xlabel("lag")
    ax.set_ylabel("amplitude")
    ax.set_title("Impulse Response Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target
