"""Search neural model configurations across lag values."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_configs
from utils.data_io import load_array, save_csv, save_json, save_npz
from utils.logging_setup import setup_logger
from utils.model_training import build_model, make_train_test_loaders, select_device, train_model_with_test_history
from utils.neural_windows import LagWindowDataset
from utils.project_paths import ensure_project_directories, resolve_project_path
from visualization.training_plots import plot_train_test_loss


def _model_trials(config: dict[str, Any], model_type: str) -> list[dict[str, Any]]:
    search = config["search"]
    learning_rates = [float(value) for value in search["learning_rates"]]
    if model_type == "mlp":
        return [
            {
                "model_type": "mlp",
                "hidden": int(hidden),
                "num_hidden_layers": int(num_layers),
                "learning_rate": float(learning_rate),
            }
            for hidden, num_layers, learning_rate in itertools.product(
                search["mlp"]["hidden"],
                search["mlp"]["num_hidden_layers"],
                learning_rates,
            )
        ]
    if model_type == "lstm":
        return [
            {
                "model_type": "lstm",
                "hidden": int(hidden),
                "num_layers": int(num_layers),
                "learning_rate": float(learning_rate),
            }
            for hidden, num_layers, learning_rate in itertools.product(
                search["lstm"]["hidden"],
                search["lstm"]["num_layers"],
                learning_rates,
            )
        ]
    raise ValueError(f"Unsupported model type {model_type}")


def _epochs_for_model(config: dict[str, Any], model_type: str) -> int:
    epochs_by_model = config["search"].get("epochs_by_model", {})
    return int(epochs_by_model.get(model_type, config["search"]["epochs"]))


def _select_trials_for_lag(
    trials: list[dict[str, Any]],
    *,
    model_type: str,
    lag: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    limit_config = config["search"].get("max_trials_per_lag", {})
    limit = int(limit_config.get(model_type, 0) or 0)
    if limit <= 0 or len(trials) <= limit:
        return trials

    seed = int(config["search"].get("random_seed", config["project"]["seed"]))
    rng = np.random.default_rng(seed + 1009 * lag + (0 if model_type == "mlp" else 1))

    selected_indices: set[int] = set()
    learning_rates = sorted({float(trial["learning_rate"]) for trial in trials}, reverse=True)

    # Force the subset to touch every LR band before filling the remaining budget.
    for learning_rate in learning_rates:
        candidates = [
            index
            for index, trial in enumerate(trials)
            if float(trial["learning_rate"]) == learning_rate
        ]
        if not candidates:
            continue
        selected_indices.add(int(rng.choice(candidates)))
        if len(selected_indices) >= limit:
            break

    remaining = [index for index in range(len(trials)) if index not in selected_indices]
    if len(selected_indices) < limit and remaining:
        extra = rng.choice(
            remaining,
            size=min(limit - len(selected_indices), len(remaining)),
            replace=False,
        )
        selected_indices.update(int(index) for index in extra)

    return [trials[index] for index in sorted(selected_indices)]


def _load_dataset(config: dict[str, Any], lag: int) -> LagWindowDataset:
    data_dir = resolve_project_path(config["paths"]["training_data_dir"], root=PROJECT_ROOT) / f"lag_{lag}"
    x = load_array(data_dir / "x.npy")
    y = load_array(data_dir / "y.npy")
    return LagWindowDataset(x, y, lag)


def _format_token(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace("+", "")


def _trial_stem(model_type: str, lag: int, trial_config: dict[str, Any]) -> str:
    if model_type == "mlp":
        return (
            f"{model_type}_lag_{lag}_hidden_{trial_config['hidden']}"
            f"_layers_{trial_config['num_hidden_layers']}"
            f"_lr_{_format_token(trial_config['learning_rate'])}"
        )
    return (
        f"{model_type}_lag_{lag}_hidden_{trial_config['hidden']}"
        f"_layers_{trial_config['num_layers']}"
        f"_lr_{_format_token(trial_config['learning_rate'])}"
    )


def _run_trial(
    *,
    model_type: str,
    trial_config: dict[str, Any],
    lag: int,
    dataset: LagWindowDataset,
    config: dict[str, Any],
    device: torch.device,
    logger,
) -> dict[str, Any]:
    seed = int(config["project"]["seed"]) + lag + int(trial_config["hidden"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = make_train_test_loaders(
        dataset,
        batch_size=int(config["search"]["batch_size"]),
        test_fraction=float(config["search"]["test_fraction"]),
        seed=seed,
    )
    model_config = {key: value for key, value in trial_config.items() if key not in {"model_type", "learning_rate"}}
    model = build_model(model_type, lag=lag, config=model_config)
    logger.info("Search trial model=%s lag=%d config=%s", model_type, lag, trial_config)
    history = train_model_with_test_history(
        model,
        train_loader,
        test_loader,
        device=device,
        learning_rate=float(trial_config["learning_rate"]),
        epochs=_epochs_for_model(config, model_type),
        log_every=max(int(config["search"]["epochs"]), 1),
        logger=None,
    )
    history_dir = resolve_project_path(
        Path(config["paths"]["search_dir"]) / "loss_histories",
        root=PROJECT_ROOT,
    )
    history_path = history_dir / f"{_trial_stem(model_type, lag, trial_config)}.npz"
    save_npz(
        history_path,
        train_loss=np.asarray(history["train_loss"], dtype=np.float32),
        test_loss=np.asarray(history["test_loss"], dtype=np.float32),
    )
    result = dict(trial_config)
    result.update(
        {
            "lag": lag,
            "epochs": _epochs_for_model(config, model_type),
            "final_train_mse": float(history["train_loss"][-1]),
            "final_test_mse": float(history["test_loss"][-1]),
            "best_test_mse": float(history["best_test_loss"][0]),
            "best_epoch": int(history["best_epoch"][0]),
            "loss_history_npz": str(history_path.relative_to(PROJECT_ROOT)),
        }
    )
    logger.info(
        "Search result model=%s lag=%d final_train_mse=%.6e final_test_mse=%.6e",
        model_type,
        lag,
        result["final_train_mse"],
        result["final_test_mse"],
    )
    figure_parts = [
        *(f"{key}_{value}" for key, value in model_config.items()),
        f"learning_rate_{str(trial_config['learning_rate']).replace('.', 'p')}",
    ]
    figure_name = "_".join(figure_parts)
    if bool(config["search"].get("save_loss_plots", True)):
        try:
            plot_train_test_loss(
                resolve_project_path(
                    Path("outputs/figures/neural_training_loss")
                    / f"search_{model_type}_lag_{lag}_{figure_name}.png",
                    root=PROJECT_ROOT,
                ),
                train_loss=history["train_loss"],
                test_loss=history["test_loss"],
                title=f"Search {model_type.upper()} lag={lag}, learning rate={trial_config['learning_rate']}",
            )
        except ImportError as exc:
            logger.warning("Skipping loss plot because matplotlib is unavailable: %s", exc)
    return result


def _plot_search_loss_panel(
    *,
    config: dict[str, Any],
    model_type: str,
    results: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    logger,
) -> None:
    rows_for_model = [row for row in results if row["model_type"] == model_type]
    if not rows_for_model:
        return

    rows_for_model = sorted(
        rows_for_model,
        key=lambda row: (
            int(row["lag"]),
            float(row["best_test_mse"]),
            int(row["hidden"]),
            float(row["learning_rate"]),
        ),
    )
    best = selected.get(model_type, {})
    panel_config = config["search"].get("loss_panel", {})
    if panel_config.get("layout", "balanced") == "balanced":
        n_cols = math.ceil(math.sqrt(len(rows_for_model)))
    else:
        n_cols = int(panel_config.get("columns", 6))
    n_cols = max(1, n_cols)
    n_rows = math.ceil(len(rows_for_model) / n_cols)
    cell_size = float(panel_config.get("cell_size", 2.05))
    fig_width = max(6.0, n_cols * cell_size)
    fig_height = max(4.0, n_rows * cell_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    for axis in axes.ravel():
        axis.axis("off")

    for index, row in enumerate(rows_for_model):
        ax = axes[index // n_cols][index % n_cols]
        ax.axis("on")
        history_path = resolve_project_path(row["loss_history_npz"], root=PROJECT_ROOT)
        history = np.load(history_path)
        train_loss = history["train_loss"]
        test_loss = history["test_loss"]
        epochs = np.arange(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, linewidth=0.8, label="train", color="#1f77b4")
        ax.plot(epochs, test_loss, linewidth=0.8, label="test", color="#ff7f0e")
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=5, length=2)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        if model_type == "mlp":
            title = (
                f"P={row['lag']} h={row['hidden']} L={row['num_hidden_layers']}\n"
                f"lr={row['learning_rate']} best={row['best_test_mse']:.3g}"
            )
            is_selected = (
                best
                and int(row["lag"]) == int(best["lag"])
                and int(row["hidden"]) == int(best["hidden"])
                and int(row["num_hidden_layers"]) == int(best["num_hidden_layers"])
                and float(row["learning_rate"]) == float(best["learning_rate"])
            )
        else:
            title = (
                f"P={row['lag']} h={row['hidden']} L={row['num_layers']}\n"
                f"lr={row['learning_rate']} best={row['best_test_mse']:.3g}"
            )
            is_selected = (
                best
                and int(row["lag"]) == int(best["lag"])
                and int(row["hidden"]) == int(best["hidden"])
                and int(row["num_layers"]) == int(best["num_layers"])
                and float(row["learning_rate"]) == float(best["learning_rate"])
            )
        ax.set_title(title, fontsize=6, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_edgecolor("#999999")
        if is_selected:
            for spine in ax.spines.values():
                spine.set_linewidth(3.0)
                spine.set_edgecolor("#00a83b")
        if index == 0:
            ax.legend(fontsize=5, loc="best", frameon=False)

    fig.suptitle(
        f"{model_type.upper()} Search Train Test Loss Curves green border marks selected best",
        fontsize=14,
        y=0.999,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.995), h_pad=0.65, w_pad=0.35)
    panel_path = resolve_project_path(
        Path("outputs/figures/neural_training_loss") / f"search_{model_type}_all_loss_panel.png",
        root=PROJECT_ROOT,
    )
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(panel_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s search loss panel to %s", model_type, panel_path)


def _save_search_progress(
    *,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    best_by_model: dict[str, dict[str, Any]],
    best_by_lag: dict[str, dict[str, dict[str, Any]]],
    logger,
) -> None:
    search_dir = resolve_project_path(config["paths"]["search_dir"], root=PROJECT_ROOT)
    selected_path = resolve_project_path(config["paths"]["selected_config_json"], root=PROJECT_ROOT)
    payload = {
        "search_epochs": int(config["search"]["epochs"]),
        "epochs_by_model": config["search"].get("epochs_by_model", {}),
        "selected_training_epochs": int(config["selected_training"]["epochs"]),
        "selection_metric": "best_test_mse",
        "num_completed_trials": len(results),
        "all_results": results,
        "selected": best_by_model,
        "best_by_lag": best_by_lag,
    }
    save_json(selected_path, payload)
    save_json(search_dir / "all_search_results.json", payload)
    if results:
        fieldnames = sorted({key for row in results for key in row})
        save_csv(search_dir / "all_search_results.csv", results, fieldnames)
    logger.info("Saved search progress after %d completed trials", len(results))


def main() -> None:
    config = load_yaml_configs(
        [
            "configs/neural_data_generation.yml",
            "configs/neural_configuration_search.yml",
            "configs/neural_mlp_training.yml",
            "configs/neural_lstm_training.yml",
        ]
    )
    ensure_project_directories(PROJECT_ROOT)
    logger = setup_logger("search_model_configurations")
    device = select_device("auto")
    lag_values = [int(value) for value in config["lag_experiments"]["values"]]

    results: list[dict[str, Any]] = []
    best_by_model: dict[str, dict[str, Any]] = {}
    best_by_lag: dict[str, dict[str, dict[str, Any]]] = {"mlp": {}, "lstm": {}}
    for model_type in ("mlp", "lstm"):
        trials = _model_trials(config, model_type)
        for lag in lag_values:
            selected_trials = _select_trials_for_lag(
                trials,
                model_type=model_type,
                lag=lag,
                config=config,
            )
            logger.info(
                "Prepared %d/%d %s trials for lag=%d",
                len(selected_trials),
                len(trials),
                model_type,
                lag,
            )
            dataset = _load_dataset(config, lag)
            for trial_config in selected_trials:
                result = _run_trial(
                    model_type=model_type,
                    trial_config=trial_config,
                    lag=lag,
                    dataset=dataset,
                    config=config,
                    device=device,
                    logger=logger,
                )
                results.append(result)
                current_best = best_by_model.get(model_type)
                if current_best is None or result["best_test_mse"] < current_best["best_test_mse"]:
                    best_by_model[model_type] = result
                lag_key = str(lag)
                current_lag_best = best_by_lag[model_type].get(lag_key)
                if current_lag_best is None or result["best_test_mse"] < current_lag_best["best_test_mse"]:
                    best_by_lag[model_type][lag_key] = result
                _save_search_progress(
                    config=config,
                    results=results,
                    best_by_model=best_by_model,
                    best_by_lag=best_by_lag,
                    logger=logger,
                )

    output_path = resolve_project_path(config["paths"]["selected_config_json"], root=PROJECT_ROOT)
    logger.info("Saved selected neural configurations to %s", output_path)
    for model_type, result in best_by_model.items():
        logger.info("Selected %s configuration: %s", model_type, result)
    for model_type in ("mlp", "lstm"):
        _plot_search_loss_panel(
            config=config,
            model_type=model_type,
            results=results,
            selected=best_by_model,
            logger=logger,
        )


if __name__ == "__main__":
    main()
