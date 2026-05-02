"""Assignment figure generation for the project outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_configs
from utils.data_io import load_array, load_npz
from utils.project_paths import resolve_project_path
from utils.random_process import impulse_response, true_power_spectrum


def db(values: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(values), floor))


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return path


def _plot_psd(ax, omega: np.ndarray, psd: np.ndarray, label: str, linewidth: float = 1.5) -> None:
    ax.plot(omega / np.pi, db(psd), label=label, linewidth=linewidth)
    ax.set_xlabel("normalized frequency omega/pi")
    ax.set_ylabel("power dB")


def plot_system_overview(config: dict, figure_dir: Path) -> Path:
    numerator = np.asarray(config["system"]["numerator"], dtype=float)
    denominator = np.asarray(config["system"]["denominator"], dtype=float)
    reference = load_npz("data/generated_processes/reference_process.npz")

    zeros = np.roots(numerator)
    poles = np.roots(denominator)
    h = reference["impulse_response"]
    lags = reference["autocorrelation_lags"]
    acf = reference["true_autocorrelation"]
    omega = reference["omega"]
    true_psd = reference["true_psd"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    theta = np.linspace(0, 2.0 * np.pi, 400)
    axes[0].plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1, label="unit circle")
    axes[0].scatter(np.real(zeros), np.imag(zeros), marker="o", s=80, label="zeros")
    axes[0].scatter(np.real(poles), np.imag(poles), marker="x", s=100, label="poles")
    axes[0].axhline(0, linewidth=1)
    axes[0].axvline(0, linewidth=1)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Pole zero plot")
    axes[0].set_xlabel("real")
    axes[0].set_ylabel("imaginary")
    axes[0].legend()

    axes[1].plot(np.arange(250), h[:250], linewidth=1.4)
    axes[1].set_title("Impulse response h[n]")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("h[n]")

    axes[2].plot(lags, acf, linewidth=1.4)
    axes[2].set_title("True autocorrelation")
    axes[2].set_xlabel("lag")
    axes[2].set_ylabel("r_y[k]")

    _plot_psd(axes[3], omega, true_psd, "true PSD", linewidth=2.0)
    axes[3].set_title("True power spectrum")
    axes[3].legend(fontsize=9)

    return _save(fig, figure_dir / "system_overview.png")


def plot_generated_realization(figure_dir: Path) -> Path:
    reference = load_npz("data/generated_processes/reference_process.npz")
    y = reference["process_samples"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    axes[0].plot(y, linewidth=1.0)
    axes[0].set_title("Generated output realization y[n], N=1024")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("y[n]")
    axes[1].hist(y, bins=40, density=True, alpha=0.8)
    axes[1].set_title("Histogram of generated y[n]")
    axes[1].set_xlabel("value")
    axes[1].set_ylabel("empirical density")
    return _save(fig, figure_dir / "generated_realization.png")


def plot_autocorrelation_grid(config: dict, figure_dir: Path) -> Path:
    data_lengths = [int(value) for value in config["project"]["data_lengths"]]
    results = load_npz("outputs/classical_spectral_estimation/autocorrelation_estimates.npz")
    true_lags = results["true_lags"]
    true_acf = results["true_autocorrelation"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    axes = axes.ravel()
    for index, length in enumerate(data_lengths):
        ax = axes[index]
        lags = results[f"lags_{length}"]
        estimate = results[f"estimate_{length}"]
        ax.plot(true_lags, true_acf, label="true", linewidth=2)
        ax.stem(lags, estimate, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="estimate")
        ax.set_title(f"Autocorrelation estimate, N={length}")
        ax.set_xlabel("lag")
        ax.set_ylabel("r_y[k]")
        ax.legend(fontsize=9)
    axes[-1].axis("off")
    return _save(fig, figure_dir / "autocorrelation_estimates_grid.png")


def plot_periodogram_grid(config: dict, figure_dir: Path) -> Path:
    data_lengths = [int(value) for value in config["project"]["data_lengths"]]
    results = load_npz("outputs/classical_spectral_estimation/periodogram_statistics.npz")
    omega = results["omega"]
    true_psd = results["true_psd"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for index, length in enumerate(data_lengths):
        ax = axes[index]
        _plot_psd(ax, omega, true_psd, "true PSD", linewidth=2)
        _plot_psd(ax, omega, results[f"single_estimate_{length}"], f"periodogram N={length}", linewidth=1)
        ax.set_title(f"Raw periodogram, N={length}")
        ax.legend(fontsize=9)
    axes[-1].axis("off")
    return _save(fig, figure_dir / "raw_periodogram_grid.png")


def plot_periodogram_statistics_grid(config: dict, figure_dir: Path) -> list[Path]:
    data_lengths = [int(value) for value in config["project"]["data_lengths"]]
    results = load_npz("outputs/classical_spectral_estimation/periodogram_statistics.npz")
    omega = results["omega"]
    true_psd = results["true_psd"]
    saved = []

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for index, length in enumerate(data_lengths):
        ax = axes[index]
        _plot_psd(ax, omega, true_psd, "true PSD", linewidth=2)
        _plot_psd(ax, omega, results[f"mean_{length}"], "sample mean", linewidth=1.5)
        ax.set_title(f"Mean periodogram, N={length}")
        ax.legend(fontsize=9)
    axes[-1].axis("off")
    saved.append(_save(fig, figure_dir / "periodogram_mean_grid.png"))

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    axes = axes.ravel()
    for index, length in enumerate(data_lengths):
        ax = axes[index]
        ax.plot(omega / np.pi, db(results[f"variance_{length}"]), linewidth=1.2, label="sample variance")
        ax.plot(omega / np.pi, db(true_psd**2), linestyle="--", linewidth=1.2, label="S_y squared")
        ax.set_title(f"Periodogram variance, N={length}")
        ax.set_xlabel("normalized frequency omega/pi")
        ax.set_ylabel("variance dB")
        ax.legend(fontsize=9)
    axes[-1].axis("off")
    saved.append(_save(fig, figure_dir / "periodogram_variance_grid.png"))
    return saved


def plot_welch_statistics_grid(config: dict, figure_dir: Path) -> list[Path]:
    data_lengths = [int(value) for value in config["project"]["data_lengths"]]
    period = load_npz("outputs/classical_spectral_estimation/periodogram_statistics.npz")
    welch = load_npz("outputs/classical_spectral_estimation/welch_statistics.npz")
    omega = welch["omega"]
    true_psd = welch["true_psd"]
    saved = []

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for index, length in enumerate(data_lengths):
        ax = axes[index]
        _plot_psd(ax, omega, true_psd, "true PSD", linewidth=2)
        _plot_psd(ax, omega, welch[f"mean_{length}"], "Welch mean", linewidth=1.5)
        ax.set_title(f"Mean Welch PSD, N={length}")
        ax.legend(fontsize=9)
    axes[-1].axis("off")
    saved.append(_save(fig, figure_dir / "welch_mean_grid.png"))

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    axes = axes.ravel()
    for index, length in enumerate(data_lengths):
        ax = axes[index]
        ax.plot(omega / np.pi, db(period[f"variance_{length}"]), linewidth=1.0, label="raw variance")
        ax.plot(omega / np.pi, db(welch[f"variance_{length}"]), linewidth=1.5, label="Welch variance")
        ax.set_title(f"Variance comparison, N={length}")
        ax.set_xlabel("normalized frequency omega/pi")
        ax.set_ylabel("variance dB")
        ax.legend(fontsize=9)
    axes[-1].axis("off")
    saved.append(_save(fig, figure_dir / "welch_variance_grid.png"))
    return saved


def plot_welch_single_grid(config: dict, figure_dir: Path) -> Path:
    data_lengths = [int(value) for value in config["project"]["data_lengths"]]
    results = load_npz("outputs/classical_spectral_estimation/welch_statistics.npz")
    omega = results["omega"]
    true_psd = results["true_psd"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for index, length in enumerate(data_lengths):
        ax = axes[index]
        _plot_psd(ax, omega, true_psd, "true PSD", linewidth=2)
        _plot_psd(ax, omega, results[f"single_estimate_{length}"], f"Welch N={length}", linewidth=1.3)
        ax.set_title(f"Welch estimate, N={length}")
        ax.legend(fontsize=9)
    axes[-1].axis("off")
    return _save(fig, figure_dir / "welch_single_grid.png")


def plot_neural_overview(config: dict, figure_dir: Path) -> list[Path]:
    selected_path = resolve_project_path(
        "outputs/neural_spectral_estimation/selected_neural_configurations.json",
        root=PROJECT_ROOT,
    )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected"]
    n_fft = int(config["project"]["frequency_grid_size"])
    omega, true_psd = true_power_spectrum(
        n_fft,
        numerator=config["system"]["numerator"],
        denominator=config["system"]["denominator"],
        noise_variance=float(config["system"]["noise_variance"]),
    )
    saved = []
    data_lengths = [int(value) for value in config["project"]["data_lengths"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    _plot_psd(ax, omega, true_psd, "true PSD", linewidth=2.5)
    for model_name in ("mlp", "lstm"):
        lag = int(selected[model_name]["lag"])
        stats = load_npz(f"outputs/neural_spectral_estimation/periodogram_statistics_{model_name}_lag_{lag}.npz")
        _plot_psd(ax, omega, stats["mean_1024"], f"{model_name.upper()} mean periodogram", linewidth=1.4)
    ax.set_title("Mean PSD comparison for learned outputs, N=1024")
    ax.legend(fontsize=9)
    saved.append(_save(fig, figure_dir / "learned_mean_psd_comparison.png"))

    model_names = ("mlp", "lstm")
    fig, axes = plt.subplots(len(model_names), len(data_lengths), figsize=(22, 8), sharex=True, sharey=True)
    for row, model_name in enumerate(model_names):
        lag = int(selected[model_name]["lag"])
        stats = load_npz(f"outputs/neural_spectral_estimation/periodogram_statistics_{model_name}_lag_{lag}.npz")
        for col, length in enumerate(data_lengths):
            ax = axes[row, col]
            _plot_psd(ax, omega, true_psd, "true PSD", linewidth=1.8)
            _plot_psd(ax, omega, stats[f"mean_{length}"], f"{model_name.upper()} mean", linewidth=1.2)
            ax.set_title(f"{model_name.upper()} learned mean, N={length}")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    saved.append(_save(fig, figure_dir / "learned_periodogram_mean_grid.png"))

    raw_periodogram = load_npz("outputs/classical_spectral_estimation/periodogram_statistics.npz")
    fig, axes = plt.subplots(len(model_names), len(data_lengths), figsize=(22, 8), sharex=True, sharey=True)
    for row, model_name in enumerate(model_names):
        lag = int(selected[model_name]["lag"])
        stats = load_npz(f"outputs/neural_spectral_estimation/periodogram_statistics_{model_name}_lag_{lag}.npz")
        for col, length in enumerate(data_lengths):
            ax = axes[row, col]
            ax.plot(
                omega / np.pi,
                db(raw_periodogram[f"variance_{length}"]),
                linestyle="--",
                linewidth=1.0,
                label="true-process raw variance",
            )
            ax.plot(
                omega / np.pi,
                db(stats[f"variance_{length}"]),
                linewidth=1.2,
                label=f"{model_name.upper()} variance",
            )
            ax.set_title(f"{model_name.upper()} learned variance, N={length}")
            ax.set_xlabel("normalized frequency omega/pi")
            ax.set_ylabel("variance dB")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    saved.append(_save(fig, figure_dir / "learned_periodogram_variance_grid.png"))

    max_lag = int(config["analysis"]["impulse_response_max_lag"])
    true_h = impulse_response(
        max_lag + 1,
        numerator=config["system"]["numerator"],
        denominator=config["system"]["denominator"],
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    lags = np.arange(len(true_h))
    ax.plot(lags, true_h, label="true h[k]", linewidth=2.3)
    for model_name in ("mlp", "lstm"):
        lag = int(selected[model_name]["lag"])
        h = load_npz(f"outputs/neural_spectral_estimation/impulse_response_estimate_{model_name}_lag_{lag}.npz")
        ax.plot(h["lags"], h["estimated_impulse_response"], label=f"{model_name.upper()} estimate", linewidth=1.3)
    ax.set_title("Impulse response estimated from sample cross correlation")
    ax.set_xlabel("lag")
    ax.set_ylabel("h[k]")
    ax.legend(fontsize=9)
    saved.append(_save(fig, figure_dir / "learned_impulse_response_comparison.png"))
    return saved


def main() -> None:
    _style()
    config = load_yaml_configs(
        [
            "configs/system_iir_process.yml",
            "configs/classical_reference_process.yml",
            "configs/neural_inference_analysis.yml",
        ]
    )
    figure_dir = resolve_project_path("outputs/figures/assignment_summary", root=PROJECT_ROOT)
    saved = [
        plot_system_overview(config, figure_dir),
        plot_generated_realization(figure_dir),
        plot_autocorrelation_grid(config, figure_dir),
        plot_periodogram_grid(config, figure_dir),
    ]
    saved.extend(plot_periodogram_statistics_grid(config, figure_dir))
    saved.append(plot_welch_single_grid(config, figure_dir))
    saved.extend(plot_welch_statistics_grid(config, figure_dir))
    saved.extend(plot_neural_overview(config, figure_dir))
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
