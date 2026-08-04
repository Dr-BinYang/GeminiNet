from __future__ import annotations

import os
import random
from pathlib import Path

# Windows Conda environments may load separate OpenMP runtimes through
# PyTorch and the Matplotlib/NumPy stack. Visualization is a side effect, so
# prefer keeping training alive over aborting after an epoch has finished.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

STATE_NAMES = [
    "x1",
    "y1",
    "x2",
    "y2",
    "vx1",
    "vy1",
    "vx2",
    "vy2",
    "distance",
    "foreground_area",
    "overlap_area",
]


# Fixed visualization setting.
# When random_samples=False, these fixed examples will be used.
FIXED_VIS_BATCH_INDEX = 0
FIXED_VIS_SAMPLE_INDICES = [0, 1]


def _prepare_image_display(
    arrays: list[np.ndarray],
    img_norm_type: str,
    img_output_activation: str,
) -> list[np.ndarray]:
    """Prepare images for display without changing model or training data."""
    direct_display = img_norm_type in {"scale_01", "minmax"} and img_output_activation == "sigmoid"

    if direct_display:
        return [np.clip(array, 0.0, 1.0) for array in arrays]

    full = np.concatenate(
        [array.reshape(-1) for array in arrays],
        axis=0,
    )
    min_value = float(full.min())
    max_value = float(full.max())
    denominator = max(max_value - min_value, 1e-6)

    return [np.clip((array - min_value) / denominator, 0.0, 1.0) for array in arrays]


def _finite_range(array: np.ndarray, min_span: float = 1e-6) -> tuple[float, float]:
    """Return a finite display range, expanding constant arrays slightly."""
    finite_values = np.asarray(array)[np.isfinite(array)]

    if finite_values.size == 0:
        return 0.0, 1.0

    min_value = float(finite_values.min())
    max_value = float(finite_values.max())

    if max_value - min_value < min_span:
        pad = max(min_span, abs(max_value) * 1e-3)
        return min_value - pad, max_value + pad

    return min_value, max_value


def _joint_finite_range(
    arrays: list[np.ndarray],
    min_span: float = 1e-6,
) -> tuple[float, float]:
    """Return one range shared by several arrays."""
    full = np.concatenate(
        [np.asarray(array).reshape(-1) for array in arrays],
        axis=0,
    )
    return _finite_range(full, min_span=min_span)


def _is_scalar_image_sequence(array: np.ndarray) -> bool:
    """Return whether an image sequence can be shown with a scalar colormap."""
    return array.ndim == 3 or (
        array.ndim == 4 and (array.shape[-1] == 1 or array.shape[-1] not in {3, 4})
    )


def _select_batch(loader, random_samples: bool) -> tuple[dict, int]:
    """Select a fixed or random batch from the loader."""
    total_batches = len(loader)

    if total_batches <= 0:
        raise ValueError("The visualization loader is empty.")

    if random_samples:
        selected_batch_index = random.randint(0, total_batches - 1)
    else:
        selected_batch_index = min(FIXED_VIS_BATCH_INDEX, total_batches - 1)

    for batch_index, batch in enumerate(loader):
        if batch_index == selected_batch_index:
            return batch, selected_batch_index

    batch = next(iter(loader))
    return batch, 0


def _select_sample_indices(
    batch_size: int,
    sample_count: int,
    random_samples: bool,
) -> list[int]:
    """Select fixed or random sample indices inside the selected batch."""
    sample_count = min(sample_count, batch_size)

    if random_samples:
        return random.sample(
            range(batch_size),
            k=sample_count,
        )

    selected_indices = [index for index in FIXED_VIS_SAMPLE_INDICES if 0 <= index < batch_size]

    if len(selected_indices) < sample_count:
        for index in range(batch_size):
            if index not in selected_indices:
                selected_indices.append(index)

            if len(selected_indices) >= sample_count:
                break

    return selected_indices[:sample_count]


def _plot_image_sequence(
    fig,
    grid,
    row_index: int,
    frames: np.ndarray,
    frame_columns: int,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot one row of future image frames."""
    for time_index in range(frame_columns):
        outer_ax = fig.add_subplot(grid[row_index, time_index])
        outer_ax.axis("off")

        ax = outer_ax.inset_axes([0.005, 0.005, 0.99, 0.99])

        if time_index < len(frames):
            frame = frames[time_index]

            if frame.ndim == 3 and frame.shape[-1] == 1:
                ax.imshow(
                    frame[..., 0],
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                )
            elif frame.ndim == 3 and frame.shape[-1] in {3, 4}:
                ax.imshow(np.clip(frame, 0.0, 1.0))
            else:
                ax.imshow(
                    frame,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

        ax.axis("off")


def _add_centered_row_label(
    fig,
    grid,
    row_index: int,
    text: str,
    y_offset: float = -0.10,
) -> None:
    """Place one centered label below a full image row."""
    ax = fig.add_subplot(grid[row_index, :-1])
    ax.patch.set_alpha(0.0)
    ax.axis("off")
    ax.text(
        0.5,
        y_offset,
        text,
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=14,
    )


def _add_grid_colorbar(
    fig,
    grid,
    row_slice,
    label: str,
    vmin: float,
    vmax: float,
    cmap: str,
) -> None:
    """Add a compact vertical colorbar in the reserved GridSpec column."""
    bar_ax = fig.add_subplot(grid[row_slice, -1])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    colorbar = fig.colorbar(scalar_mappable, cax=bar_ax)
    colorbar.set_label(label, fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def _image_abs_bias(y_img: np.ndarray, pred_img: np.ndarray) -> np.ndarray:
    """Return image absolute bias as one heatmap per forecast frame."""
    abs_bias = np.abs(pred_img - y_img)

    if abs_bias.ndim == 4 and abs_bias.shape[-1] in {1, 3, 4}:
        return np.mean(abs_bias, axis=-1)

    return abs_bias


def _numerical_column_slices(frame_columns: int) -> tuple[slice, slice]:
    """Return left curve columns and right heatmap columns."""
    if frame_columns <= 1:
        return slice(0, 1), slice(0, 1)

    if frame_columns >= 6:
        split = frame_columns // 2
        return slice(0, max(2, split - 1)), slice(split + 1, frame_columns)

    if frame_columns >= 3:
        return slice(0, frame_columns - 1), slice(frame_columns - 1, frame_columns)

    return slice(0, 1), slice(max(1, frame_columns - 1), frame_columns)


def _plot_numerical_panel(
    fig,
    grid,
    base_row: int,
    y_num: np.ndarray,
    pred_num: np.ndarray,
    frame_columns: int,
) -> None:
    """Plot numerical target/prediction curves and absolute-bias heatmap."""
    num_abs_bias = np.abs(pred_num - y_num)
    num_left, num_right = _numerical_column_slices(frame_columns)

    ax_target = fig.add_subplot(grid[base_row + 4, num_left])
    ax_pred = fig.add_subplot(grid[base_row + 5, num_left])

    time_axis = np.arange(y_num.shape[0])
    y_min = min(float(y_num.min()), float(pred_num.min()))
    y_max = max(float(y_num.max()), float(pred_num.max()))
    y_pad = max(0.05, 0.08 * (y_max - y_min))

    for var_index in range(y_num.shape[-1]):
        if var_index < len(STATE_NAMES):
            name = STATE_NAMES[var_index]
        else:
            name = f"var_{var_index}"

        ax_target.plot(
            time_axis,
            y_num[:, var_index],
            linewidth=1.15,
            label=name,
        )
        ax_pred.plot(
            time_axis,
            pred_num[:, var_index],
            linewidth=1.15,
            linestyle=(0, (2, 1)),
        )

    for ax in [ax_target, ax_pred]:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlim(0, max(1, y_num.shape[0] - 1))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(axis="both", labelsize=8)

    ax_target.set_xticklabels([])
    ax_target.text(
        0.5,
        -0.22,
        "Num Target",
        ha="center",
        va="top",
        transform=ax_target.transAxes,
        fontsize=10,
    )
    ax_pred.text(
        0.5,
        -0.30,
        "Num Prediction",
        ha="center",
        va="top",
        transform=ax_pred.transAxes,
        fontsize=10,
    )

    ax_target.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.08, 1.0, 0.18),
        mode="expand",
        ncol=min(y_num.shape[-1], 6),
        fontsize=7,
        borderaxespad=0.0,
    )

    ax_heatmap = fig.add_subplot(grid[base_row + 4 : base_row + 6, num_right])

    bias_min, bias_max = 0, 0.4

    heatmap = ax_heatmap.imshow(
        num_abs_bias.T,
        aspect="auto",
        cmap="magma",
        vmin=bias_min,
        vmax=bias_max,
    )
    ax_heatmap.set_title("Num Abs Bias Heatmap", fontsize=11, pad=6)
    ax_heatmap.set_yticks(np.arange(y_num.shape[-1]))
    ax_heatmap.set_yticklabels(
        [
            STATE_NAMES[index] if index < len(STATE_NAMES) else f"var_{index}"
            for index in range(y_num.shape[-1])
        ],
        fontsize=8,
    )
    ax_heatmap.set_xticks(np.arange(y_num.shape[0]))
    ax_heatmap.set_xticklabels(
        [f"t+{index + 1}" for index in range(y_num.shape[0])],
        fontsize=8,
    )
    ax_heatmap.set_xlabel("Forecast horizon")

    num_bar_ax = fig.add_subplot(grid[base_row + 4 : base_row + 6, -1])
    colorbar = fig.colorbar(heatmap, cax=num_bar_ax)
    colorbar.set_label("Num abs bias", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)


@torch.no_grad()
def visualize_multimodal_batch(
    model,
    loader,
    device,
    save_path: str | Path,
    sample_count: int = 2,
    title: str = "Prediction Visualization",
    random_samples: bool = False,
    img_norm_type: str = "scale_01",
    img_output_activation: str = "sigmoid",
) -> None:
    """Visualize future multimodal target, prediction and absolute bias.

    For each sample, this figure contains:
        row 1: future image target;
        row 2: future image prediction;
        row 3: image absolute-bias heatmap;
        row 4-5 left: numerical target and prediction curves;
        row 4-5 right: numerical absolute-bias heatmap.
    """
    model.eval()

    batch, selected_batch_index = _select_batch(
        loader=loader,
        random_samples=random_samples,
    )

    batch_size = batch["x_img"].shape[0]

    sample_indices = _select_sample_indices(
        batch_size=batch_size,
        sample_count=sample_count,
        random_samples=random_samples,
    )

    x_img = batch["x_img"][sample_indices].to(device)
    x_num = batch["x_num"][sample_indices].to(device)
    y_img = batch["y_img"][sample_indices].to(device)
    y_num = batch["y_num"][sample_indices].to(device)

    outputs = model(x_img, x_num)

    if isinstance(outputs, dict):
        pred_img = outputs["y_img"]
        pred_num = outputs["y_num"]
    else:
        pred_img, pred_num = outputs

    y_img_np = y_img.detach().cpu().numpy()
    pred_img_np = pred_img.detach().cpu().numpy()
    y_num_np = y_num.detach().cpu().numpy()
    pred_num_np = pred_num.detach().cpu().numpy()

    sample_count = y_img_np.shape[0]
    pred_len = y_img_np.shape[1]
    frame_columns = pred_len
    grid_columns = frame_columns + 1
    rows_per_sample = 6
    total_rows = rows_per_sample * sample_count

    fig = plt.figure(figsize=(1.30 * frame_columns, 1.55 * total_rows))
    grid = fig.add_gridspec(
        total_rows,
        grid_columns,
        height_ratios=[1, 1, 1, 0.18, 1.05, 1.05] * sample_count,
        width_ratios=[1.0] * frame_columns + [0.08],
    )

    for display_idx in range(sample_count):
        base_row = display_idx * rows_per_sample

        y_img_sample = y_img_np[display_idx]
        pred_img_sample = pred_img_np[display_idx]

        if _is_scalar_image_sequence(y_img_sample):
            y_img_disp = y_img_sample
            pred_img_disp = pred_img_sample
        else:
            y_img_disp, pred_img_disp = _prepare_image_display(
                [y_img_sample, pred_img_sample],
                img_norm_type=img_norm_type,
                img_output_activation=img_output_activation,
            )

        image_value_min, image_value_max = _finite_range(y_img_disp)

        img_abs_bias = _image_abs_bias(
            y_img_sample,
            pred_img_sample,
        )

        img_bias_min, img_bias_max = 0, 1

        _plot_image_sequence(
            fig=fig,
            grid=grid,
            row_index=base_row + 0,
            frames=y_img_disp,
            frame_columns=frame_columns,
            vmin=image_value_min,
            vmax=image_value_max,
        )
        _add_centered_row_label(
            fig=fig,
            grid=grid,
            row_index=base_row + 0,
            text=(f"B{selected_batch_index}/S{sample_indices[display_idx]} " "Image Target"),
        )

        _plot_image_sequence(
            fig=fig,
            grid=grid,
            row_index=base_row + 1,
            frames=pred_img_disp,
            frame_columns=frame_columns,
            vmin=image_value_min,
            vmax=image_value_max,
        )
        _add_centered_row_label(
            fig=fig,
            grid=grid,
            row_index=base_row + 1,
            text="Image Prediction",
        )

        _plot_image_sequence(
            fig=fig,
            grid=grid,
            row_index=base_row + 2,
            frames=img_abs_bias,
            frame_columns=frame_columns,
            cmap="magma",
            vmin=img_bias_min,
            vmax=img_bias_max,
        )
        _add_centered_row_label(
            fig=fig,
            grid=grid,
            row_index=base_row + 2,
            text="Image Abs Bias",
        )

        _add_grid_colorbar(
            fig=fig,
            grid=grid,
            row_slice=slice(base_row + 0, base_row + 2),
            label="Image value",
            vmin=image_value_min,
            vmax=image_value_max,
            cmap="gray",
        )
        _add_grid_colorbar(
            fig=fig,
            grid=grid,
            row_slice=slice(base_row + 2, base_row + 3),
            label="Abs bias",
            vmin=img_bias_min,
            vmax=img_bias_max,
            cmap="magma",
        )

        _plot_numerical_panel(
            fig=fig,
            grid=grid,
            base_row=base_row,
            y_num=y_num_np[display_idx],
            pred_num=pred_num_np[display_idx],
            frame_columns=frame_columns,
        )

    plt.suptitle(title, fontsize=14, y=0.995)
    plt.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.975,
        bottom=0.025,
        wspace=0.08,
        hspace=0.42,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)