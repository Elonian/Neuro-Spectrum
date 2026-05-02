"""Periodogram and Welch spectral estimators."""

from __future__ import annotations

import numpy as np


def _as_batch(sequence: np.ndarray) -> np.ndarray:
    arr = np.asarray(sequence, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D sequence array, got shape {arr.shape}")


def periodogram(sequence: np.ndarray, *, n_fft: int | None = None) -> np.ndarray:
    y = _as_batch(sequence)
    n_samples = y.shape[-1]
    fft_len = int(n_fft or n_samples)
    transformed = np.fft.fft(y, n=fft_len, axis=-1)
    estimate = np.abs(transformed) ** 2 / n_samples
    return estimate[0] if np.asarray(sequence).ndim == 1 else estimate


def periodogram_statistics(
    realizations: np.ndarray,
    *,
    data_lengths: list[int],
    n_fft: int,
) -> dict[int, dict[str, np.ndarray]]:
    y = _as_batch(realizations)
    stats: dict[int, dict[str, np.ndarray]] = {}
    for length in data_lengths:
        if length > y.shape[-1]:
            raise ValueError(f"data length {length} exceeds realization length {y.shape[-1]}")
        estimates = periodogram(y[:, :length], n_fft=n_fft)
        stats[int(length)] = {
            "mean": np.mean(estimates, axis=0),
            "variance": np.var(estimates, axis=0, ddof=1 if y.shape[0] > 1 else 0),
        }
    return stats


def make_window(name: str, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("window length must be positive")
    normalized = name.lower()
    if normalized in {"rect", "rectangular", "boxcar"}:
        return np.ones(length, dtype=float)
    if normalized in {"hann", "hanning"}:
        return np.hanning(length)
    if normalized == "hamming":
        return np.hamming(length)
    if normalized == "blackman":
        return np.blackman(length)
    raise ValueError(f"Unsupported window '{name}'")


def welch_periodogram(
    sequence: np.ndarray,
    *,
    segment_length: int,
    overlap: int,
    window: str = "hann",
    n_fft: int | None = None,
) -> np.ndarray:
    y = _as_batch(sequence)
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    if overlap < 0 or overlap >= segment_length:
        raise ValueError("overlap must satisfy 0 <= overlap < segment_length")
    if segment_length > y.shape[-1]:
        raise ValueError("segment_length cannot exceed sequence length")

    step = segment_length - overlap
    starts = list(range(0, y.shape[-1] - segment_length + 1, step))
    if not starts:
        raise ValueError("No Welch segments fit the sequence")

    w = make_window(window, segment_length)
    scale = np.sum(w**2)
    fft_len = int(n_fft or segment_length)
    segment_estimates = []
    for start in starts:
        segment = y[:, start : start + segment_length] * w
        transformed = np.fft.fft(segment, n=fft_len, axis=-1)
        segment_estimates.append(np.abs(transformed) ** 2 / scale)
    estimate = np.mean(np.stack(segment_estimates, axis=0), axis=0)
    return estimate[0] if np.asarray(sequence).ndim == 1 else estimate


def welch_statistics(
    realizations: np.ndarray,
    *,
    data_lengths: list[int],
    welch_plan: dict[int, dict[str, int | str]],
    n_fft: int,
) -> dict[int, dict[str, np.ndarray]]:
    y = _as_batch(realizations)
    stats: dict[int, dict[str, np.ndarray]] = {}
    for length in data_lengths:
        plan = welch_plan[int(length)]
        estimates = welch_periodogram(
            y[:, :length],
            segment_length=int(plan["segment_length"]),
            overlap=int(plan["overlap"]),
            window=str(plan.get("window", "hann")),
            n_fft=n_fft,
        )
        stats[int(length)] = {
            "mean": np.mean(estimates, axis=0),
            "variance": np.var(estimates, axis=0, ddof=1 if y.shape[0] > 1 else 0),
        }
    return stats


def default_welch_plan(data_lengths: list[int], *, window: str = "hann") -> dict[int, dict[str, int | str]]:
    plan: dict[int, dict[str, int | str]] = {}
    for length in data_lengths:
        segment_length = max(16, length // 4)
        segment_length = min(segment_length, length)
        overlap = segment_length // 2
        plan[int(length)] = {
            "segment_length": int(segment_length),
            "overlap": int(overlap),
            "window": window,
        }
    return plan

