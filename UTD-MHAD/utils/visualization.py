from __future__ import annotations

import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

STATE_NAMES = [
    "bt_mean_k",
    "bt_std_k",
    "bt_min_k",
    "bt_p10_k",
    "cold_fraction_235k",
    "cold_fraction_220k",
    "bt_gradient_mean",
]


# Fixed visualization setting.
# When random_samples=False, these fixed examples will be used.
FIXED_VIS_BATCH_INDEX = 0
FIXED_VIS_SAMPLE_INDICES = [0, 1]


def _safe_minmax(array: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    """Return a non-degenerate min/max range for color mapping."""
    min_value = float(np.nanmin(array))
    max_value = float(np.nanmax(array))

    if max_value - min_value < eps:
        max_value = min_value + eps

    return min_value, max_value


def _safe_abs_bias_range(
    array: np.ndarray,
    percentile: float = 99.0,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Return a shared absolute-bias color range.

    The lower bound is fixed at 0. The upper bound uses a high percentile
    instead of the absolute maximum so one extreme pixel/variable does not
    flatten the contrast of all displayed samples.
    """
    max_value = float(
        np.nanpercentile(
            np.asarray(array, dtype=np.float32),
            percentile,
        )
    )
    return 0.0, max(max_value, eps)


def _image_abs_bias(pred_img: np.ndarray, y_img: np.ndarray) -> np.ndarray:
    """Compute image absolute bias for single-channel or RGB-like images."""
    if y_img.shape[-1] in [1, 3]:
        return np.mean(
            np.abs(pred_img - y_img),
            axis=-1,
        )
    return np.abs(pred_img - y_img)


def _scale_image_to_display(
    array: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """Scale image values for display using a fixed target-derived range.

    Prediction outliers are clipped only at the display layer. They never
    change the target-derived color/display range.
    """
    denominator = max(vmax - vmin, 1e-6)
    return np.clip((array - vmin) / denominator, 0.0, 1.0)


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
                    cmap=cmap or "viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
            elif frame.ndim == 3 and frame.shape[-1] == 3:
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


def _plot_numerical_panel(
    fig,
    grid,
    base_row: int,
    y_num: np.ndarray,
    pred_num: np.ndarray,
    frame_columns: int,
    num_bias_vmin: float,
    num_bias_vmax: float,
    state_names: list[str],
) -> None:
    """Plot numerical target/prediction curves and absolute-bias heatmap."""
    num_abs_bias = np.abs(pred_num - y_num)

    # For long horizons, keep a small gap between curves and heatmap to avoid
    # label overlap. For short horizons,
    # split the available image columns directly; otherwise the heatmap slice
    # may become empty.
    if frame_columns <= 5:
        left_end = max(1, frame_columns // 2)
        num_left = slice(0, left_end)
        num_right = slice(left_end, frame_columns)
    else:
        num_left = slice(0, max(3, frame_columns // 2 - 1))
        num_right = slice(max(5, frame_columns // 2 + 1), frame_columns)

    if num_right.start >= num_right.stop:
        num_right = slice(max(0, frame_columns - 1), frame_columns)

    ax_target = fig.add_subplot(grid[base_row + 4, num_left])
    ax_pred = fig.add_subplot(grid[base_row + 5, num_left])

    time_axis = np.arange(y_num.shape[0])
    y_min = min(float(y_num.min()), float(pred_num.min()))
    y_max = max(float(y_num.max()), float(pred_num.max()))
    y_pad = max(0.05, 0.08 * (y_max - y_min))

    for var_index in range(y_num.shape[-1]):
        if var_index < len(state_names):
            name = state_names[var_index]
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
        ax.set_xlim(0, y_num.shape[0] - 1)
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
        ncol=min(y_num.shape[-1], 7),
        fontsize=7,
        borderaxespad=0.0,
    )

    ax_heatmap = fig.add_subplot(grid[base_row + 4 : base_row + 6, num_right])
    heatmap = ax_heatmap.imshow(
        num_abs_bias.T,
        aspect="auto",
        cmap="magma",
        vmin=num_bias_vmin,
        vmax=num_bias_vmax,
    )
    ax_heatmap.set_title("Num Abs Bias Heatmap", fontsize=11, pad=6)
    ax_heatmap.set_yticks(np.arange(y_num.shape[-1]))
    ax_heatmap.set_yticklabels(
        [
            state_names[index] if index < len(state_names) else f"var_{index}"
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
    normalizer=None,
    inverse_transform: bool = True,
) -> None:
    """Visualize multimodal prediction target, prediction and absolute bias.

    For each selected sample, the figure contains:
        row 1: future image target;
        row 2: future image prediction;
        row 3: image absolute-bias heatmap;
        row 4-5 left: numerical target and prediction curves;
        row 4-5 right: numerical absolute-bias heatmap.
    """
    model.eval()

    batch, _ = _select_batch(
        loader=loader,
        random_samples=random_samples,
    )

    batch_size = batch["x_img"].shape[0]
    state_names = getattr(getattr(loader, "dataset", None), "num_feature_names", STATE_NAMES)

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

    if inverse_transform and normalizer is not None:
        y_img = normalizer.image.denormalize(y_img)
        pred_img = normalizer.image.denormalize(pred_img)
        y_num = normalizer.numerical.denormalize(y_num)
        pred_num = normalizer.numerical.denormalize(pred_num)

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

    frame_height = int(y_img_np.shape[2])
    if frame_height >= 192:
        frame_scale = 1.25
        row_scale = 1.55
    else:
        frame_scale = 1.55 if frame_height >= 128 else 1.30
        row_scale = 1.75 if frame_height >= 128 else 1.55

    fig = plt.figure(figsize=(frame_scale * frame_columns, row_scale * total_rows))
    grid = fig.add_gridspec(
        total_rows,
        grid_columns,
        height_ratios=[1, 1, 1, 0.18, 1.05, 1.05] * sample_count,
        width_ratios=[1.0] * frame_columns + [0.08],
    )

    is_rgb_image = y_img_np.shape[-1] == 3
    image_value_min, image_value_max = _safe_minmax(y_img_np)

    all_img_abs_bias = _image_abs_bias(
        pred_img=pred_img_np,
        y_img=y_img_np,
    )
    img_bias_min, img_bias_max = _safe_abs_bias_range(all_img_abs_bias)

    all_num_abs_bias = np.abs(pred_num_np - y_num_np)
    num_bias_min, num_bias_max = _safe_abs_bias_range(all_num_abs_bias)

    for display_index in range(sample_count):
        base_row = display_index * rows_per_sample

        y_img_raw = y_img_np[display_index]
        pred_img_raw = pred_img_np[display_index]

        if is_rgb_image:
            rgb_scale = 255.0 if np.nanmax(y_img_raw) > 2.0 else 1.0
            y_img_disp = np.clip(y_img_raw / rgb_scale, 0.0, 1.0)
            pred_img_disp = np.clip(pred_img_raw / rgb_scale, 0.0, 1.0)
        else:
            y_img_disp = _scale_image_to_display(
                y_img_raw,
                vmin=image_value_min,
                vmax=image_value_max,
            )
            pred_img_disp = _scale_image_to_display(
                pred_img_raw,
                vmin=image_value_min,
                vmax=image_value_max,
            )

        img_abs_bias = all_img_abs_bias[display_index]

        _plot_image_sequence(
            fig=fig,
            grid=grid,
            row_index=base_row + 0,
            frames=y_img_disp,
            frame_columns=frame_columns,
            cmap=None if is_rgb_image else "viridis",
        )
        _add_centered_row_label(
            fig=fig,
            grid=grid,
            row_index=base_row + 0,
            text="Image Target",
        )

        _plot_image_sequence(
            fig=fig,
            grid=grid,
            row_index=base_row + 1,
            frames=pred_img_disp,
            frame_columns=frame_columns,
            cmap=None if is_rgb_image else "viridis",
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

        if not is_rgb_image:
            _add_grid_colorbar(
                fig=fig,
                grid=grid,
                row_slice=slice(base_row + 0, base_row + 2),
                label="Image value",
                vmin=image_value_min,
                vmax=image_value_max,
                cmap="viridis",
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
            y_num=y_num_np[display_index],
            pred_num=pred_num_np[display_index],
            frame_columns=frame_columns,
            num_bias_vmin=num_bias_min,
            num_bias_vmax=num_bias_max,
            state_names=state_names,
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