"""Random process generation and true system quantities."""

from __future__ import annotations

import numpy as np
from scipy import signal


DEFAULT_NUMERATOR = np.array([1.0, -0.9, 0.81], dtype=float)
DEFAULT_DENOMINATOR = np.array([1.0, -2.76, 3.809, -2.654, 0.924], dtype=float)


def normalize_filter_coefficients(
    numerator: list[float] | np.ndarray | None = None,
    denominator: list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(DEFAULT_NUMERATOR if numerator is None else numerator, dtype=float)
    a = np.asarray(DEFAULT_DENOMINATOR if denominator is None else denominator, dtype=float)
    if b.ndim != 1 or a.ndim != 1:
        raise ValueError("Filter coefficients must be one dimensional")
    if a.size == 0 or a[0] == 0:
        raise ValueError("Denominator must have a nonzero leading coefficient")
    return b / a[0], a / a[0]


def filter_iir(sequence: np.ndarray, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    x = np.asarray(sequence, dtype=float)
    b, a = normalize_filter_coefficients(numerator, denominator)
    if signal is not None:
        return signal.lfilter(b, a, x, axis=-1)
    y = np.zeros_like(x, dtype=float)
    n_samples = x.shape[-1]

    for n in range(n_samples):
        acc = np.zeros(x.shape[:-1], dtype=float)
        for k, coeff in enumerate(b):
            if n - k >= 0:
                acc = acc + coeff * x[..., n - k]
        for k in range(1, len(a)):
            if n - k >= 0:
                acc = acc - a[k] * y[..., n - k]
        y[..., n] = acc
    return y


def generate_white_noise(
    length: int,
    *,
    num_realizations: int = 1,
    variance: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    if num_realizations <= 0:
        raise ValueError("num_realizations must be positive")
    rng = np.random.default_rng(seed)
    shape = (length,) if num_realizations == 1 else (num_realizations, length)
    return rng.normal(loc=0.0, scale=np.sqrt(variance), size=shape)


def generate_process_realizations(
    length: int,
    *,
    num_realizations: int = 1,
    numerator: list[float] | np.ndarray | None = None,
    denominator: list[float] | np.ndarray | None = None,
    noise_variance: float = 1.0,
    burnin: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if burnin < 0:
        raise ValueError("burnin must not be negative")
    b, a = normalize_filter_coefficients(numerator, denominator)
    x = generate_white_noise(
        length + burnin,
        num_realizations=num_realizations,
        variance=noise_variance,
        seed=seed,
    )
    y_full = filter_iir(x, b, a)
    if burnin > 0:
        return x[..., burnin:], y_full[..., burnin:]
    return x, y_full


def generate_aligned_training_pair(
    output_length: int,
    lag: int,
    *,
    numerator: list[float] | np.ndarray | None = None,
    denominator: list[float] | np.ndarray | None = None,
    noise_variance: float = 1.0,
    burnin: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if lag < 0:
        raise ValueError("lag must not be negative")
    if burnin < 0:
        raise ValueError("burnin must not be negative")
    b, a = normalize_filter_coefficients(numerator, denominator)
    x_all = generate_white_noise(output_length + lag + burnin, variance=noise_variance, seed=seed)
    y_full = filter_iir(x_all, b, a)
    x_store = x_all[burnin : burnin + lag + output_length]
    y = y_full[burnin + lag : burnin + lag + output_length]
    return x_store.astype(np.float32), y.astype(np.float32)


def impulse_response(
    length: int,
    *,
    numerator: list[float] | np.ndarray | None = None,
    denominator: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    b, a = normalize_filter_coefficients(numerator, denominator)
    delta = np.zeros(length, dtype=float)
    delta[0] = 1.0
    return filter_iir(delta, b, a)


def true_autocorrelation_from_impulse_response(
    impulse: np.ndarray,
    max_lag: int,
    *,
    noise_variance: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(impulse, dtype=float).reshape(-1)
    if max_lag < 0:
        raise ValueError("max_lag must not be negative")
    if max_lag >= h.size:
        raise ValueError("max_lag must be smaller than impulse response length")
    lags = np.arange(max_lag + 1)
    autocorr = np.array(
        [noise_variance * np.dot(h[: h.size - lag], h[lag:]) for lag in lags],
        dtype=float,
    )
    return lags, autocorr


def true_power_spectrum(
    n_fft: int,
    *,
    numerator: list[float] | np.ndarray | None = None,
    denominator: list[float] | np.ndarray | None = None,
    noise_variance: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    b, a = normalize_filter_coefficients(numerator, denominator)
    omega = 2.0 * np.pi * np.arange(n_fft) / n_fft
    z_inv = np.exp(-1j * omega)
    numerator_eval = sum(coeff * z_inv**k for k, coeff in enumerate(b))
    denominator_eval = sum(coeff * z_inv**k for k, coeff in enumerate(a))
    response = numerator_eval / denominator_eval
    return omega, noise_variance * np.abs(response) ** 2


def sample_autocorrelation(
    sequence: np.ndarray,
    max_lag: int,
    *,
    biased: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(sequence, dtype=float).reshape(-1)
    if max_lag < 0 or max_lag >= x.size:
        raise ValueError("max_lag must satisfy 0 <= max_lag < len(sequence)")
    lags = np.arange(max_lag + 1)
    values = []
    for lag in lags:
        numerator = np.dot(x[: x.size - lag], x[lag:])
        denominator = x.size if biased else x.size - lag
        values.append(numerator / denominator)
    return lags, np.asarray(values, dtype=float)


def estimate_impulse_response_from_io(
    input_sequence: np.ndarray,
    output_sequence: np.ndarray,
    max_lag: int,
    *,
    input_pre_history: int = 0,
    input_variance: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(input_sequence, dtype=float)
    y = np.asarray(output_sequence, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if y.ndim == 1:
        y = y[None, :]
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"batch mismatch: x shape {x.shape}, y shape {y.shape}")
    if max_lag < 0:
        raise ValueError("max_lag must not be negative")

    batch, out_len = y.shape
    lags = np.arange(max_lag + 1)
    estimates = np.zeros(max_lag + 1, dtype=float)
    for lag in lags:
        products: list[np.ndarray] = []
        for n in range(out_len):
            x_idx = n + input_pre_history - lag
            if 0 <= x_idx < x.shape[1]:
                products.append(y[:, n] * x[:, x_idx])
        if not products:
            estimates[lag] = np.nan
            continue
        stacked = np.stack(products, axis=1)
        estimates[lag] = np.mean(stacked) / input_variance
    return lags, estimates
