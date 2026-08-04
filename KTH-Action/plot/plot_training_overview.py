from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================
# JSON file names
# ============================================================

IMAGE_PRIOR_JSON = "image_prior_metrics.json"
NUM_PRIOR_JSON = "num_prior_metrics.json"
GEMININET_JSON = "gemininet_metrics.json"


# ============================================================
# Path utilities
# ============================================================


def get_project_root() -> Path:
    """Get project root regardless of current working directory."""
    current_file = Path(__file__).resolve()

    for parent in [current_file.parent, *current_file.parents]:
        if (parent / "run.py").exists():
            return parent

    if current_file.parent.name == "plot":
        return current_file.parents[1]

    return current_file.parent


def resolve_log_dir(log_dir: str | Path | None = None) -> Path:
    """Resolve metric log directory.

    Default:
        project_root / experiments / kth_action_motion_stats / logs
    """
    if log_dir is None:
        return get_project_root() / "experiments" / "kth_action_motion_stats" / "logs"

    log_dir = Path(log_dir)

    if log_dir.is_absolute():
        return log_dir

    return get_project_root() / log_dir


def resolve_save_dir(
    log_dir: Path,
    save_dir: str | Path | None = None,
) -> Path:
    """Resolve figure output directory.

    Default:
        log_dir / overview_plots
    """
    if save_dir is None:
        return log_dir / "overview_plots"

    save_dir = Path(save_dir)

    if save_dir.is_absolute():
        return save_dir

    return get_project_root() / save_dir


# ============================================================
# Utility functions
# ============================================================


def load_json_if_exists(path: Path) -> dict | None:
    """Load one JSON metric file if it exists."""
    if not path.exists():
        print(f"[Warning] JSON not found: {path}")
        return None

    with open(path, "r", encoding="utf-8") as file:
        history = json.load(file)

    if "epoch" not in history:
        raise ValueError(f"Metric JSON must contain 'epoch': {path}")

    return history


def plot_two_models(
    ax,
    prior_history: dict | None,
    gemini_history: dict | None,
    prior_key: str,
    gemini_key: str,
    prior_label: str,
    gemini_label: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot prior model and GeminiNet on the same metric and same split."""
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)

    if prior_history is not None and prior_key in prior_history:
        ax.plot(
            prior_history["epoch"],
            prior_history[prior_key],
            label=prior_label,
            linewidth=1.8,
        )
    else:
        print(f"[Warning] Prior metric not found: {prior_key}")

    if gemini_history is not None and gemini_key in gemini_history:
        ax.plot(
            gemini_history["epoch"],
            gemini_history[gemini_key],
            label=gemini_label,
            linewidth=1.8,
            linestyle="--",
        )
    else:
        print(f"[Warning] GeminiNet metric not found: {gemini_key}")

    handles, _ = ax.get_legend_handles_labels()

    if len(handles) > 0:
        ax.legend(fontsize=9)


def plot_pair(
    ax,
    history: dict | None,
    train_key: str,
    val_key: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot train and validation curves of one model."""
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)

    if history is None:
        ax.text(
            0.5,
            0.5,
            "JSON not found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    epochs = history["epoch"]

    if train_key in history:
        ax.plot(
            epochs,
            history[train_key],
            label=train_key,
            linewidth=1.8,
        )
    else:
        print(f"[Warning] Metric not found: {train_key}")

    if val_key in history:
        ax.plot(
            epochs,
            history[val_key],
            label=val_key,
            linewidth=1.8,
            linestyle="--",
        )
    else:
        print(f"[Warning] Metric not found: {val_key}")

    handles, _ = ax.get_legend_handles_labels()

    if len(handles) > 0:
        ax.legend(fontsize=9)


def save_figure(fig, save_path: Path) -> None:
    """Save one figure. Existing figures will be overwritten."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")


# ============================================================
# Overview figure 1: GeminiNet loss and reliability
# ============================================================


def plot_gemininet_loss_overview(
    gemini_history: dict | None,
    save_dir: Path,
) -> None:
    """Plot GeminiNet total loss and reliability losses."""
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 9),
    )

    axes = axes.ravel()

    plot_pair(
        ax=axes[0],
        history=gemini_history,
        train_key="train_total_loss",
        val_key="val_total_loss",
        title="GeminiNet total loss",
        ylabel="Loss",
    )

    plot_pair(
        ax=axes[1],
        history=gemini_history,
        train_key="train_rel_loss",
        val_key="val_rel_loss",
        title="GeminiNet total reliability loss",
        ylabel="Loss",
    )

    plot_pair(
        ax=axes[2],
        history=gemini_history,
        train_key="train_rel_img_loss",
        val_key="val_rel_img_loss",
        title="GeminiNet image reliability loss",
        ylabel="Loss",
    )

    plot_pair(
        ax=axes[3],
        history=gemini_history,
        train_key="train_rel_num_loss",
        val_key="val_rel_num_loss",
        title="GeminiNet numerical reliability loss",
        ylabel="Loss",
    )

    fig.suptitle(
        "GeminiNet loss and reliability overview",
        fontsize=15,
    )

    save_figure(
        fig=fig,
        save_path=save_dir / "01_gemininet_loss_reliability_overview.png",
    )


# ============================================================
# Overview figure 2: MSE comparison
# ============================================================


def plot_mse_comparison(
    image_history: dict | None,
    num_history: dict | None,
    gemini_history: dict | None,
    save_dir: Path,
) -> None:
    """Compare unimodal models and GeminiNet using MSE."""
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 10),
    )

    axes = axes.ravel()

    plot_two_models(
        ax=axes[0],
        prior_history=image_history,
        gemini_history=gemini_history,
        prior_key="train_img_mse_norm",
        gemini_key="train_img_mse_norm",
        prior_label="Image prior train MSE (norm)",
        gemini_label="GeminiNet train image MSE (norm)",
        title="Image MSE on training set (normalized)",
        ylabel="Image MSE",
    )

    plot_two_models(
        ax=axes[1],
        prior_history=image_history,
        gemini_history=gemini_history,
        prior_key="val_img_mse_norm",
        gemini_key="val_img_mse_norm",
        prior_label="Image prior val MSE (norm)",
        gemini_label="GeminiNet val image MSE (norm)",
        title="Image MSE on validation set (normalized)",
        ylabel="Image MSE",
    )

    plot_two_models(
        ax=axes[2],
        prior_history=num_history,
        gemini_history=gemini_history,
        prior_key="train_num_mse_norm",
        gemini_key="train_num_mse_norm",
        prior_label="Numerical prior train MSE (norm)",
        gemini_label="GeminiNet train numerical MSE (norm)",
        title="Numerical MSE on training set (normalized)",
        ylabel="Numerical MSE",
    )

    plot_two_models(
        ax=axes[3],
        prior_history=num_history,
        gemini_history=gemini_history,
        prior_key="val_num_mse_norm",
        gemini_key="val_num_mse_norm",
        prior_label="Numerical prior val MSE (norm)",
        gemini_label="GeminiNet val numerical MSE (norm)",
        title="Numerical MSE on validation set (normalized)",
        ylabel="Numerical MSE",
    )

    fig.suptitle(
        "MSE comparison: unimodal prior models vs GeminiNet",
        fontsize=15,
    )

    save_figure(
        fig=fig,
        save_path=save_dir / "02_mse_comparison_train_val.png",
    )


# ============================================================
# Overview figure 3: MAE comparison
# ============================================================


def plot_mae_comparison(
    image_history: dict | None,
    num_history: dict | None,
    gemini_history: dict | None,
    save_dir: Path,
) -> None:
    """Compare unimodal models and GeminiNet using MAE."""
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 10),
    )

    axes = axes.ravel()

    plot_two_models(
        ax=axes[0],
        prior_history=image_history,
        gemini_history=gemini_history,
        prior_key="train_img_mae_norm",
        gemini_key="train_img_mae_norm",
        prior_label="Image prior train MAE (norm)",
        gemini_label="GeminiNet train image MAE (norm)",
        title="Image MAE on training set (normalized)",
        ylabel="Image MAE",
    )

    plot_two_models(
        ax=axes[1],
        prior_history=image_history,
        gemini_history=gemini_history,
        prior_key="val_img_mae_norm",
        gemini_key="val_img_mae_norm",
        prior_label="Image prior val MAE (norm)",
        gemini_label="GeminiNet val image MAE (norm)",
        title="Image MAE on validation set (normalized)",
        ylabel="Image MAE",
    )

    plot_two_models(
        ax=axes[2],
        prior_history=num_history,
        gemini_history=gemini_history,
        prior_key="train_num_mae_norm",
        gemini_key="train_num_mae_norm",
        prior_label="Numerical prior train MAE (norm)",
        gemini_label="GeminiNet train numerical MAE (norm)",
        title="Numerical MAE on training set (normalized)",
        ylabel="Numerical MAE",
    )

    plot_two_models(
        ax=axes[3],
        prior_history=num_history,
        gemini_history=gemini_history,
        prior_key="val_num_mae_norm",
        gemini_key="val_num_mae_norm",
        prior_label="Numerical prior val MAE (norm)",
        gemini_label="GeminiNet val numerical MAE (norm)",
        title="Numerical MAE on validation set (normalized)",
        ylabel="Numerical MAE",
    )

    fig.suptitle(
        "MAE comparison: unimodal prior models vs GeminiNet",
        fontsize=15,
    )

    save_figure(
        fig=fig,
        save_path=save_dir / "03_mae_comparison_train_val.png",
    )


# ============================================================
# Main
# ============================================================


def main(
    log_dir: str | Path | None = None,
    save_dir: str | Path | None = None,
) -> None:
    """Generate overview figures.

    JSON path:
        project_root / experiments / kth_action_motion_stats / logs

    PNG output:
        project_root / experiments / kth_action_motion_stats / logs / overview_plots
    """
    log_dir = resolve_log_dir(log_dir)
    save_dir = resolve_save_dir(log_dir, save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    image_history = load_json_if_exists(log_dir / IMAGE_PRIOR_JSON)
    num_history = load_json_if_exists(log_dir / NUM_PRIOR_JSON)
    gemini_history = load_json_if_exists(log_dir / GEMININET_JSON)

    plot_gemininet_loss_overview(
        gemini_history=gemini_history,
        save_dir=save_dir,
    )

    plot_mse_comparison(
        image_history=image_history,
        num_history=num_history,
        gemini_history=gemini_history,
        save_dir=save_dir,
    )

    plot_mae_comparison(
        image_history=image_history,
        num_history=num_history,
        gemini_history=gemini_history,
        save_dir=save_dir,
    )

    print(f"All overview figures have been saved to: {save_dir}")


if __name__ == "__main__":
    main()