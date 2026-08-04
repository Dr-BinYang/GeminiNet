from __future__ import annotations

import json
import os
from pathlib import Path

# Keep manual plotting and auto plotting alive in Windows Conda environments
# where PyTorch and Matplotlib/NumPy can initialize separate OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
        project_root / experiments / moving_mnist_state / logs
    """
    if log_dir is None:
        return get_project_root() / "experiments" / "moving_mnist_state" / "logs"

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
    """Load a metric JSON file if it exists."""
    if not path.exists():
        print(f"[Warning] JSON not found: {path}")
        return None

    with open(path, "r", encoding="utf-8") as file:
        history = json.load(file)

    if "epoch" not in history:
        raise ValueError(f"Metric JSON must contain 'epoch': {path}")

    return history


def plot_pair(
    ax,
    history: dict | None,
    train_key: str,
    val_key: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot train/validation curves of the same metric in one subplot."""
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

    epochs = history.get("epoch", [])

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
# Plot functions
# ============================================================


def plot_image_prior_metrics(
    image_history: dict | None,
    save_dir: Path,
) -> None:
    """Plot image prior metrics as grouped subplots."""
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 9),
    )
    axes = axes.ravel()

    plot_pair(
        ax=axes[0],
        history=image_history,
        train_key="train_img_mse",
        val_key="val_img_mse",
        title="Image prior MSE",
        ylabel="MSE",
    )

    plot_pair(
        ax=axes[1],
        history=image_history,
        train_key="train_img_mae",
        val_key="val_img_mae",
        title="Image prior MAE",
        ylabel="MAE",
    )

    plot_pair(
        ax=axes[2],
        history=image_history,
        train_key="train_img_psnr_norm",
        val_key="val_img_psnr_norm",
        title="Image prior PSNR",
        ylabel="PSNR (dB)",
    )

    plot_pair(
        ax=axes[3],
        history=image_history,
        train_key="train_img_ms_ssim_norm",
        val_key="val_img_ms_ssim_norm",
        title="Image prior MS-SSIM",
        ylabel="MS-SSIM",
    )

    fig.suptitle(
        "Image prior training and validation metrics",
        fontsize=15,
    )

    save_figure(
        fig=fig,
        save_path=save_dir / "image_prior_metrics_grouped.png",
    )


def plot_num_prior_metrics(
    num_history: dict | None,
    save_dir: Path,
) -> None:
    """Plot numerical prior metrics as grouped subplots."""
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 5),
    )

    plot_pair(
        ax=axes[0],
        history=num_history,
        train_key="train_num_mse",
        val_key="val_num_mse",
        title="Numerical prior MSE",
        ylabel="MSE",
    )

    plot_pair(
        ax=axes[1],
        history=num_history,
        train_key="train_num_mae",
        val_key="val_num_mae",
        title="Numerical prior MAE",
        ylabel="MAE",
    )

    fig.suptitle(
        "Numerical prior training and validation metrics",
        fontsize=15,
    )

    save_figure(
        fig=fig,
        save_path=save_dir / "num_prior_metrics_grouped.png",
    )


def plot_gemininet_task_metrics(
    gemini_history: dict | None,
    save_dir: Path,
) -> None:
    """Plot GeminiNet task metrics in separated subplots."""
    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(22, 10),
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
        train_key="train_image_mse",
        val_key="val_image_mse",
        title="GeminiNet image MSE",
        ylabel="MSE",
    )

    plot_pair(
        ax=axes[2],
        history=gemini_history,
        train_key="train_image_mae",
        val_key="val_image_mae",
        title="GeminiNet image MAE",
        ylabel="MAE",
    )

    plot_pair(
        ax=axes[3],
        history=gemini_history,
        train_key="train_img_psnr_norm",
        val_key="val_img_psnr_norm",
        title="GeminiNet image PSNR",
        ylabel="PSNR (dB)",
    )

    plot_pair(
        ax=axes[4],
        history=gemini_history,
        train_key="train_img_ms_ssim_norm",
        val_key="val_img_ms_ssim_norm",
        title="GeminiNet image MS-SSIM",
        ylabel="MS-SSIM",
    )

    plot_pair(
        ax=axes[5],
        history=gemini_history,
        train_key="train_num_mse",
        val_key="val_num_mse",
        title="GeminiNet numerical MSE",
        ylabel="MSE",
    )

    plot_pair(
        ax=axes[6],
        history=gemini_history,
        train_key="train_num_mae",
        val_key="val_num_mae",
        title="GeminiNet numerical MAE",
        ylabel="MAE",
    )

    plot_pair(
        ax=axes[7],
        history=gemini_history,
        train_key="train_rel_loss",
        val_key="val_rel_loss",
        title="GeminiNet reliability loss",
        ylabel="Loss",
    )

    fig.suptitle(
        "GeminiNet task metrics",
        fontsize=15,
    )

    save_figure(
        fig=fig,
        save_path=save_dir / "gemininet_task_metrics_grouped.png",
    )


def plot_gemininet_reliability_metrics(
    gemini_history: dict | None,
    save_dir: Path,
) -> None:
    """Plot GeminiNet reliability-related losses separately."""
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5),
    )

    plot_pair(
        ax=axes[0],
        history=gemini_history,
        train_key="train_rel_loss",
        val_key="val_rel_loss",
        title="Total reliability loss",
        ylabel="Loss",
    )

    plot_pair(
        ax=axes[1],
        history=gemini_history,
        train_key="train_rel_img_loss",
        val_key="val_rel_img_loss",
        title="Image reliability loss",
        ylabel="Loss",
    )

    plot_pair(
        ax=axes[2],
        history=gemini_history,
        train_key="train_rel_num_loss",
        val_key="val_rel_num_loss",
        title="Numerical reliability loss",
        ylabel="Loss",
    )

    fig.suptitle(
        "GeminiNet reliability metrics",
        fontsize=15,
    )

    save_figure(
        fig=fig,
        save_path=save_dir / "gemininet_reliability_metrics_grouped.png",
    )


# ============================================================
# Main
# ============================================================


def main(
    log_dir: str | Path | None = None,
    save_dir: str | Path | None = None,
) -> None:
    """Generate grouped metric figures.

    JSON path:
        project_root / experiments / moving_mnist_state / logs

    PNG output:
        project_root / experiments / moving_mnist_state / logs / overview_plots
    """
    log_dir = resolve_log_dir(log_dir)
    save_dir = resolve_save_dir(log_dir, save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    image_history = load_json_if_exists(log_dir / IMAGE_PRIOR_JSON)
    num_history = load_json_if_exists(log_dir / NUM_PRIOR_JSON)
    gemini_history = load_json_if_exists(log_dir / GEMININET_JSON)

    plot_image_prior_metrics(
        image_history=image_history,
        save_dir=save_dir,
    )

    plot_num_prior_metrics(
        num_history=num_history,
        save_dir=save_dir,
    )

    plot_gemininet_task_metrics(
        gemini_history=gemini_history,
        save_dir=save_dir,
    )

    plot_gemininet_reliability_metrics(
        gemini_history=gemini_history,
        save_dir=save_dir,
    )

    print(f"All grouped metric figures have been saved to: {save_dir}")


if __name__ == "__main__":
    main()