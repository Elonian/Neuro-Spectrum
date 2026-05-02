"""Generate the assignment process and its true autocorrelation/PSD."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_configs
from utils.data_io import save_npz
from utils.logging_setup import setup_logger
from utils.project_paths import ensure_project_directories, resolve_project_path
from utils.random_process import (
    generate_aligned_training_pair,
    impulse_response,
    true_autocorrelation_from_impulse_response,
    true_power_spectrum,
)


def main() -> None:
    config = load_yaml_configs(
        [
            "configs/system_iir_process.yml",
            "configs/classical_reference_process.yml",
        ]
    )
    ensure_project_directories(PROJECT_ROOT)
    logger = setup_logger("generate_process_and_truth")

    sample_length = int(config["project"]["sample_length"])
    seed = int(config["project"]["seed"])
    reference_input_lag = int(config["project"].get("reference_input_lag", 0))
    n_fft = int(config["project"]["frequency_grid_size"])
    numerator = config["system"]["numerator"]
    denominator = config["system"]["denominator"]
    noise_variance = float(config["system"]["noise_variance"])
    burnin = int(config["system"].get("burnin_samples", 0))
    impulse_length = int(config["system"]["impulse_response_length"])
    autocorr_lag = int(config["system"]["autocorrelation_max_lag"])

    white_noise, process_samples = generate_aligned_training_pair(
        sample_length,
        reference_input_lag,
        numerator=numerator,
        denominator=denominator,
        noise_variance=noise_variance,
        burnin=burnin,
        seed=seed,
    )
    h = impulse_response(impulse_length, numerator=numerator, denominator=denominator)
    autocorr_lags, autocorrelation = true_autocorrelation_from_impulse_response(
        h,
        autocorr_lag,
        noise_variance=noise_variance,
    )
    omega, true_psd = true_power_spectrum(
        n_fft,
        numerator=numerator,
        denominator=denominator,
        noise_variance=noise_variance,
    )

    output_path = resolve_project_path(config["paths"]["reference_process_npz"], root=PROJECT_ROOT)
    save_npz(
        output_path,
        white_noise=white_noise,
        reference_input_lag=reference_input_lag,
        process_samples=process_samples,
        impulse_response=h,
        autocorrelation_lags=autocorr_lags,
        true_autocorrelation=autocorrelation,
        omega=omega,
        true_psd=true_psd,
    )
    logger.info("Saved reference process and truth arrays to %s", output_path)
    logger.info(
        "Generated sample length=%d, burnin=%d, input lag=%d, FFT grid=%d",
        sample_length,
        burnin,
        reference_input_lag,
        n_fft,
    )


if __name__ == "__main__":
    main()
