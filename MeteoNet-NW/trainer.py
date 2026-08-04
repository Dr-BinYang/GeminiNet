from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from models import GeminiNet, ImageUNetPrior, NumericalTransformerPrior
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import JSONLogger, save_json
from utils.metrics import gemininet_loss, image_quality_metric_values

from utils.model_summary import print_model_parameters

PROGRESS_RED = "\033[32m"
PROGRESS_RESET = "\033[0m"


def _format_seconds(seconds: float) -> str:
    """Format seconds as HH:MM:SS for stable PyCharm Run output."""
    return str(timedelta(seconds=int(max(0, seconds))))


def _print_epoch_progress(
    model_name: str,
    epoch: int,
    total_epochs: int,
    stage_start_time: float,
    epoch_durations: list[float],
) -> None:
    """Print one persistent progress line for PyCharm Run/Terminal."""
    elapsed_seconds = time.time() - stage_start_time
    avg_epoch_seconds = sum(epoch_durations) / max(1, len(epoch_durations))
    remaining_epochs = max(0, total_epochs - epoch)
    eta_seconds = avg_epoch_seconds * remaining_epochs
    estimated_total_seconds = elapsed_seconds + eta_seconds
    speed_epoch_per_second = 1.0 / max(avg_epoch_seconds, 1e-6)
    progress_percent = 100.0 * epoch / max(1, total_epochs)

    progress_text = (
        "[Progress] "
        f"{model_name} epoch={epoch:03d}/{total_epochs:03d} "
        f"progress={progress_percent:6.2f}% "
        f"elapsed={_format_seconds(elapsed_seconds)} "
        f"total={_format_seconds(estimated_total_seconds)} "
        f"remaining={_format_seconds(eta_seconds)} "
        f"speed={speed_epoch_per_second:.2f}epoch/s "
        f"epoch_time={_format_seconds(epoch_durations[-1])}"
    )

    print(f"{PROGRESS_RED}{progress_text}{PROGRESS_RESET}")


def _forecast_metric_values(
    pred: torch.Tensor,
    target: torch.Tensor,
    modality: str,
    space: str,
) -> dict:
    """Compute MAE/MSE/RMSE for reporting only."""
    with torch.no_grad():
        pred = pred.detach()
        target = target.detach()
        mse = F.mse_loss(pred, target).item()
        mae = F.l1_loss(pred, target).item()
        rmse = mse**0.5

    return {
        f"{modality}_mse_{space}": mse,
        f"{modality}_mae_{space}": mae,
        f"{modality}_rmse_{space}": rmse,
    }


def _target_derived_image_range(target: torch.Tensor) -> tuple[float, float]:
    """Return target-derived min and non-degenerate value range."""
    with torch.no_grad():
        min_value = float(target.detach().min().cpu())
        max_value = float(target.detach().max().cpu())
    return min_value, max(max_value - min_value, 1e-6)


def _image_raw_data_range(
    normalizer,
    target: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Infer original image min/range for reporting metrics."""
    image_normalizer = getattr(normalizer, "image", None)

    if image_normalizer is None:
        if target is None:
            return 0.0, 1.0
        return _target_derived_image_range(target)

    if getattr(image_normalizer, "norm_type", "") == "scale_01":
        return 0.0, float(getattr(image_normalizer, "value_scale", 255.0))

    if target is None:
        return 0.0, 1.0
    return _target_derived_image_range(target)


def _add_original_scale_metrics(
    metrics: dict,
    normalizer,
    pred_img: torch.Tensor | None = None,
    y_img: torch.Tensor | None = None,
    pred_num: torch.Tensor | None = None,
    y_num: torch.Tensor | None = None,
    include_image_quality: bool = False,
) -> None:
    """Add inverse-transformed metrics for logging/evaluation only."""
    if normalizer is None:
        return

    with torch.no_grad():
        if pred_img is not None and y_img is not None:
            pred_img_raw = normalizer.image.denormalize(pred_img.detach())
            y_img_raw = normalizer.image.denormalize(y_img.detach())
            metrics.update(
                _forecast_metric_values(
                    pred=pred_img_raw,
                    target=y_img_raw,
                    modality="img",
                    space="raw",
                )
            )
            if include_image_quality:
                data_min, data_range = _image_raw_data_range(normalizer, y_img_raw)
                metrics.update(
                    image_quality_metric_values(
                        pred=pred_img_raw,
                        target=y_img_raw,
                        data_min=data_min,
                        data_range=data_range,
                        space="raw",
                    )
                )

        if pred_num is not None and y_num is not None:
            pred_num_raw = normalizer.numerical.denormalize(pred_num.detach())
            y_num_raw = normalizer.numerical.denormalize(y_num.detach())
            metrics.update(
                _forecast_metric_values(
                    pred=pred_num_raw,
                    target=y_num_raw,
                    modality="num",
                    space="raw",
                )
            )


def _average_metrics(total_metrics: dict, total_count: int) -> dict:
    """Average accumulated metric sums."""
    return {key: value / max(1, total_count) for key, value in total_metrics.items()}


def _accumulate_metrics(
    total_metrics: dict,
    metrics: dict,
    batch_size: int,
) -> None:
    """Accumulate numeric metrics by batch size."""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            total_metrics[key] = total_metrics.get(key, 0.0) + float(value) * batch_size


def _format_metrics(metrics: dict, keys: list[str]) -> str:
    """Format a compact metric string."""
    return " ".join(f"{key}={metrics[key]:.8f}" for key in keys if key in metrics)


def _format_prefixed_metrics(
    prefix: str,
    metrics: dict,
    keys: list[str],
) -> str:
    """Format metrics with train/val/test prefixes."""
    return " ".join(f"{prefix}_{key}={metrics[key]:.8f}" for key in keys if key in metrics)


def build_image_prior(args) -> ImageUNetPrior:
    """Create image U-Net tendency model."""
    return ImageUNetPrior(
        input_len=args.input_len_img,
        pred_len=args.pred_len_img,
        img_channels=args.img_channels,
        base_channels=args.unet_base_channels,
        dropout=args.dropout,
        img_output_activation=getattr(args, "img_output_activation", "sigmoid"),
        output_scale=getattr(args, "output_scale", 5.0),
    )


def build_num_prior(args) -> NumericalTransformerPrior:
    """Create numerical Transformer tendency model."""
    return NumericalTransformerPrior(
        input_len=args.input_len_num,
        pred_len=args.pred_len_num,
        num_vars=args.num_vars,
        d_model=args.prior_d_model,
        nhead=args.nhead,
        encoder_layers=args.prior_encoder_layers,
        decoder_layers=args.prior_decoder_layers,
        dim_feedforward=args.prior_dim_feedforward,
        dropout=args.dropout,
        num_output_activation=getattr(args, "num_output_activation", "linear"),
        output_scale=getattr(args, "output_scale", 5.0),
    )


def load_model_weights(model, ckpt_path: str | Path, device):
    """Load checkpoint weights into a model."""
    checkpoint = load_checkpoint(
        path=ckpt_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def train_image_prior(
    args,
    loaders: Dict,
    device,
    normalizer=None,
) -> Path:
    """Train image tendency model on the first 10% data only."""
    model = build_image_prior(args).to(device)

    print_model_parameters(
        model=model,
        model_name="Image Prior",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_prior,
        weight_decay=args.weight_decay,
    )

    save_dir = Path(args.exp_dir) / "checkpoints"
    best_path = save_dir / "image_prior_best.pth"

    logger = JSONLogger(
        path=Path(args.exp_dir) / "logs" / "image_prior_metrics.json",
        reset=True,
    )

    best_val = float("inf")
    stage_start_time = time.time()
    epoch_durations: list[float] = []

    for epoch in range(1, args.epochs_prior + 1):
        epoch_start_time = time.time()

        train_metrics = run_image_prior_epoch(
            model=model,
            loader=loaders["prior_train"],
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            normalizer=normalizer,
            metric_inverse_transform=getattr(
                args,
                "metric_inverse_transform",
                True,
            ),
        )

        val_metrics = run_image_prior_epoch(
            model=model,
            loader=loaders["val"],
            device=device,
            optimizer=None,
            normalizer=normalizer,
            metric_inverse_transform=getattr(
                args,
                "metric_inverse_transform",
                True,
            ),
        )

        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})

        logger.log(row)

        epoch_durations.append(time.time() - epoch_start_time)
        _print_epoch_progress(
            model_name="Image Prior",
            epoch=epoch,
            total_epochs=args.epochs_prior,
            stage_start_time=stage_start_time,
            epoch_durations=epoch_durations,
        )

        print(
            f"[Image Prior] "
            f"epoch={epoch:03d} "
            f"{_format_prefixed_metrics('train', train_metrics, ['img_mse_norm', 'img_mae_norm', 'img_rmse_norm', 'img_psnr_norm', 'img_ms_ssim_norm'])} "
            f"{_format_prefixed_metrics('val', val_metrics, ['img_mse_norm', 'img_mae_norm', 'img_rmse_norm', 'img_psnr_norm', 'img_ms_ssim_norm'])} "
            f"{_format_prefixed_metrics('train', train_metrics, ['img_mse_raw', 'img_mae_raw', 'img_rmse_raw', 'img_psnr_raw', 'img_ms_ssim_raw'])} "
            f"{_format_prefixed_metrics('val', val_metrics, ['img_mse_raw', 'img_mae_raw', 'img_rmse_raw', 'img_psnr_raw', 'img_ms_ssim_raw'])}"
        )

        save_checkpoint(
            path=save_dir / "image_prior_last.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_val,
        )

        if val_metrics["img_mse"] < best_val:
            best_val = val_metrics["img_mse"]

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val,
            )

    model = load_model_weights(
        model=model,
        ckpt_path=best_path,
        device=device,
    )

    test_metrics = run_image_prior_epoch(
        model=model,
        loader=loaders["test"],
        device=device,
        optimizer=None,
        normalizer=normalizer,
        metric_inverse_transform=getattr(
            args,
            "metric_inverse_transform",
            True,
        ),
    )

    print(
        "[Image Prior Test] "
        f"{_format_prefixed_metrics('test', test_metrics, ['img_mse_norm', 'img_mae_norm', 'img_rmse_norm', 'img_psnr_norm', 'img_ms_ssim_norm'])} "
        f"{_format_prefixed_metrics('test', test_metrics, ['img_mse_raw', 'img_mae_raw', 'img_rmse_raw', 'img_psnr_raw', 'img_ms_ssim_raw'])}"
    )

    save_json(
        path=Path(args.exp_dir) / "logs" / "image_prior_test_metrics.json",
        data=test_metrics,
    )

    return save_dir / "image_prior_last.pth"


def run_image_prior_epoch(
    model,
    loader,
    device,
    optimizer,
    grad_clip: float = 0.0,
    normalizer=None,
    metric_inverse_transform: bool = True,
) -> dict:
    """Train or evaluate the image prior model for one epoch."""
    is_training = optimizer is not None
    model.train(is_training)

    total_metrics = {}
    total_count = 0

    for batch in loader:
        x_img = batch["x_img"].to(device)
        y_img = batch["y_img"].to(device)

        if is_training:
            pred_img = model(x_img)
            loss = F.mse_loss(pred_img, y_img)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite image prior loss detected.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                )

            optimizer.step()

        else:
            with torch.no_grad():
                pred_img = model(x_img)

        metrics = _forecast_metric_values(
            pred=pred_img,
            target=y_img,
            modality="img",
            space="norm",
        )
        norm_data_min, norm_data_range = _target_derived_image_range(y_img)
        metrics.update(
            image_quality_metric_values(
                pred=pred_img,
                target=y_img,
                data_min=norm_data_min,
                data_range=norm_data_range,
                space="norm",
            )
        )

        # Backward-compatible aliases, still in normalized space.
        metrics["img_mse"] = metrics["img_mse_norm"]
        metrics["img_mae"] = metrics["img_mae_norm"]

        if metric_inverse_transform:
            _add_original_scale_metrics(
                metrics=metrics,
                normalizer=normalizer,
                pred_img=pred_img,
                y_img=y_img,
                include_image_quality=True,
            )

        batch_size = x_img.size(0)

        _accumulate_metrics(
            total_metrics=total_metrics,
            metrics=metrics,
            batch_size=batch_size,
        )
        total_count += batch_size

    return _average_metrics(total_metrics, total_count)


def train_num_prior(
    args,
    loaders: Dict,
    device,
    normalizer=None,
) -> Path:
    """Train numerical tendency model on the first 10% data only."""
    model = build_num_prior(args).to(device)

    print_model_parameters(
        model=model,
        model_name="Numerical Prior",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_prior,
        weight_decay=args.weight_decay,
    )

    save_dir = Path(args.exp_dir) / "checkpoints"
    best_path = save_dir / "num_prior_best.pth"

    logger = JSONLogger(
        path=Path(args.exp_dir) / "logs" / "num_prior_metrics.json",
        reset=True,
    )

    best_val = float("inf")
    stage_start_time = time.time()
    epoch_durations: list[float] = []

    for epoch in range(1, args.epochs_prior + 1):
        epoch_start_time = time.time()

        train_metrics = run_num_prior_epoch(
            model=model,
            loader=loaders["prior_train"],
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            normalizer=normalizer,
            metric_inverse_transform=getattr(
                args,
                "metric_inverse_transform",
                True,
            ),
        )

        val_metrics = run_num_prior_epoch(
            model=model,
            loader=loaders["val"],
            device=device,
            optimizer=None,
            normalizer=normalizer,
            metric_inverse_transform=getattr(
                args,
                "metric_inverse_transform",
                True,
            ),
        )

        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})

        logger.log(row)

        epoch_durations.append(time.time() - epoch_start_time)
        _print_epoch_progress(
            model_name="Numerical Prior",
            epoch=epoch,
            total_epochs=args.epochs_prior,
            stage_start_time=stage_start_time,
            epoch_durations=epoch_durations,
        )

        print(
            f"[Numerical Prior] "
            f"epoch={epoch:03d} "
            f"{_format_prefixed_metrics('train', train_metrics, ['num_mse_norm', 'num_mae_norm', 'num_rmse_norm'])} "
            f"{_format_prefixed_metrics('val', val_metrics, ['num_mse_norm', 'num_mae_norm', 'num_rmse_norm'])} "
            f"{_format_prefixed_metrics('train', train_metrics, ['num_mse_raw', 'num_mae_raw', 'num_rmse_raw'])} "
            f"{_format_prefixed_metrics('val', val_metrics, ['num_mse_raw', 'num_mae_raw', 'num_rmse_raw'])}"
        )

        save_checkpoint(
            path=save_dir / "num_prior_last.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_val,
        )

        if val_metrics["num_mse"] < best_val:
            best_val = val_metrics["num_mse"]

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val,
            )

    model = load_model_weights(
        model=model,
        ckpt_path=best_path,
        device=device,
    )

    test_metrics = run_num_prior_epoch(
        model=model,
        loader=loaders["test"],
        device=device,
        optimizer=None,
        normalizer=normalizer,
        metric_inverse_transform=getattr(
            args,
            "metric_inverse_transform",
            True,
        ),
    )

    print(
        "[Numerical Prior Test] "
        f"{_format_prefixed_metrics('test', test_metrics, ['num_mse_norm', 'num_mae_norm', 'num_rmse_norm'])} "
        f"{_format_prefixed_metrics('test', test_metrics, ['num_mse_raw', 'num_mae_raw', 'num_rmse_raw'])}"
    )

    save_json(
        path=Path(args.exp_dir) / "logs" / "num_prior_test_metrics.json",
        data=test_metrics,
    )

    return save_dir / "num_prior_last.pth"


def run_num_prior_epoch(
    model,
    loader,
    device,
    optimizer,
    grad_clip: float = 0.0,
    normalizer=None,
    metric_inverse_transform: bool = True,
) -> dict:
    """Train or evaluate the numerical prior model for one epoch."""
    is_training = optimizer is not None
    model.train(is_training)

    total_metrics = {}
    total_count = 0

    for batch in loader:
        x_num = batch["x_num"].to(device)
        y_num = batch["y_num"].to(device)

        if is_training:
            pred_num = model(x_num)
            loss = F.mse_loss(pred_num, y_num)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite numerical prior loss detected.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                )

            optimizer.step()

        else:
            with torch.no_grad():
                pred_num = model(x_num)

        metrics = _forecast_metric_values(
            pred=pred_num,
            target=y_num,
            modality="num",
            space="norm",
        )

        # Backward-compatible aliases, still in normalized space.
        metrics["num_mse"] = metrics["num_mse_norm"]
        metrics["num_mae"] = metrics["num_mae_norm"]

        if metric_inverse_transform:
            _add_original_scale_metrics(
                metrics=metrics,
                normalizer=normalizer,
                pred_num=pred_num,
                y_num=y_num,
            )

        batch_size = x_num.size(0)

        _accumulate_metrics(
            total_metrics=total_metrics,
            metrics=metrics,
            batch_size=batch_size,
        )
        total_count += batch_size

    return _average_metrics(total_metrics, total_count)


def build_gemininet(
    args,
    image_ckpt: str | Path,
    num_ckpt: str | Path,
    device,
) -> GeminiNet:
    """Build GeminiNet and load frozen tendency generators."""
    image_prior = build_image_prior(args).to(device)
    num_prior = build_num_prior(args).to(device)

    image_prior = load_model_weights(image_prior, image_ckpt, device)
    num_prior = load_model_weights(num_prior, num_ckpt, device)

    model = GeminiNet(
        image_prior=image_prior,
        num_prior=num_prior,
        input_len_img=args.input_len_img,
        input_len_num=args.input_len_num,
        pred_len_img=args.pred_len_img,
        pred_len_num=args.pred_len_num,
        img_size=args.img_size,
        img_channels=args.img_channels,
        num_vars=args.num_vars,
        d_img=args.d_img,
        d_num=args.d_num,
        d_shared=args.d_shared,
        nhead=args.nhead,
        dropout=args.dropout,
        use_reliability_estimator=getattr(
            args,
            "use_reliability_estimator",
            True,
        ),
        img_output_activation=getattr(args, "img_output_activation", "sigmoid"),
        num_output_activation=getattr(args, "num_output_activation", "linear"),
        output_scale=getattr(args, "output_scale", 5.0),
    )

    return model.to(device)


def train_gemininet(
    args,
    loaders: Dict,
    device,
    image_ckpt: str | Path,
    num_ckpt: str | Path,
    normalizer=None,
) -> Path:
    """Train GeminiNet on the middle 70% data only."""
    model = build_gemininet(
        args=args,
        image_ckpt=image_ckpt,
        num_ckpt=num_ckpt,
        device=device,
    )

    print_model_parameters(
        model=model,
        model_name="GeminiNet",
    )

    print(
        "[GeminiNet] use_reliability_estimator = "
        f"{getattr(args, 'use_reliability_estimator', True)}"
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr_gemini,
        weight_decay=args.weight_decay,
    )

    save_dir = Path(args.exp_dir) / "checkpoints"
    vis_dir = Path(args.exp_dir) / "visualization"
    best_path = save_dir / "gemininet_best.pth"

    logger = JSONLogger(
        path=Path(args.exp_dir) / "logs" / "gemininet_metrics.json",
        reset=True,
    )

    best_val = float("inf")
    stage_start_time = time.time()
    epoch_durations: list[float] = []

    for epoch in range(1, args.epochs_gemini + 1):
        epoch_start_time = time.time()

        train_metrics = run_gemininet_epoch(
            model=model,
            loader=loaders["gemini_train"],
            device=device,
            optimizer=optimizer,
            args=args,
            normalizer=normalizer,
        )

        val_metrics = run_gemininet_epoch(
            model=model,
            loader=loaders["val"],
            device=device,
            optimizer=None,
            args=args,
            normalizer=normalizer,
        )

        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["use_reliability_estimator"] = train_metrics["use_reliability_estimator"]

        logger.log(row)

        epoch_durations.append(time.time() - epoch_start_time)
        _print_epoch_progress(
            model_name="GeminiNet",
            epoch=epoch,
            total_epochs=args.epochs_gemini,
            stage_start_time=stage_start_time,
            epoch_durations=epoch_durations,
        )

        print(
            f"[GeminiNet] "
            f"epoch={epoch:03d} "
            f"train_total={train_metrics['total_loss']:.8f} "
            f"{_format_prefixed_metrics('train', train_metrics, ['img_mse_norm', 'img_mae_norm', 'img_rmse_norm', 'img_psnr_norm', 'img_ms_ssim_norm', 'num_mse_norm', 'num_mae_norm', 'num_rmse_norm'])} "
            f"{_format_prefixed_metrics('train', train_metrics, ['img_mse_raw', 'img_mae_raw', 'img_rmse_raw', 'img_psnr_raw', 'img_ms_ssim_raw', 'num_mse_raw', 'num_mae_raw', 'num_rmse_raw'])} "
            f"train_rel={train_metrics['rel_loss']:.8f} "
            f"train_r_img={train_metrics['r_img_mean']:.4f} "
            f"train_r_num={train_metrics['r_num_mean']:.4f} "
            f"val_total={val_metrics['total_loss']:.8f} "
            f"{_format_prefixed_metrics('val', val_metrics, ['img_mse_norm', 'img_mae_norm', 'img_rmse_norm', 'img_psnr_norm', 'img_ms_ssim_norm', 'num_mse_norm', 'num_mae_norm', 'num_rmse_norm'])} "
            f"{_format_prefixed_metrics('val', val_metrics, ['img_mse_raw', 'img_mae_raw', 'img_rmse_raw', 'img_psnr_raw', 'img_ms_ssim_raw', 'num_mse_raw', 'num_mae_raw', 'num_rmse_raw'])} "
            f"val_rel={val_metrics['rel_loss']:.8f} "
            f"val_r_img={val_metrics['r_img_mean']:.4f} "
            f"val_r_num={val_metrics['r_num_mean']:.4f}"
        )

        save_checkpoint(
            path=save_dir / "gemininet_last.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_val,
            extra={"args": vars(args)},
        )

        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val,
                extra={"args": vars(args)},
            )

        visualize_train = bool(
            getattr(
                args,
                "visualize_train",
                getattr(args, "visualize", True),
            )
        )
        visualize_vali = bool(
            getattr(
                args,
                "visualize_vali",
                getattr(args, "visualize", True),
            )
        )

        if (visualize_train or visualize_vali) and epoch % args.vis_interval == 0:
            from utils.visualization import visualize_multimodal_batch

            if visualize_train:
                visualize_multimodal_batch(
                    model=model,
                    loader=loaders["gemini_train"],
                    device=device,
                    save_path=vis_dir / "train" / f"gemininet_train_epoch_{epoch:04d}.png",
                    sample_count=2,
                    title=f"GeminiNet Train Prediction Epoch {epoch}",
                    random_samples=args.vis_random_samples,
                    normalizer=normalizer,
                    inverse_transform=getattr(args, "vis_inverse_transform", True),
                )

            if visualize_vali:
                visualize_multimodal_batch(
                    model=model,
                    loader=loaders["val"],
                    device=device,
                    save_path=vis_dir / "val" / f"gemininet_val_epoch_{epoch:04d}.png",
                    sample_count=2,
                    title=f"GeminiNet Validation Prediction Epoch {epoch}",
                    random_samples=args.vis_random_samples,
                    normalizer=normalizer,
                    inverse_transform=getattr(args, "vis_inverse_transform", True),
                )

    return best_path


def run_gemininet_epoch(
    model,
    loader,
    device,
    optimizer,
    args,
    normalizer=None,
) -> dict:
    """Run one epoch for GeminiNet training or evaluation."""
    is_training = optimizer is not None
    model.train(is_training)

    total_metrics = {}
    total_count = 0

    for batch in loader:
        x_img = batch["x_img"].to(device)
        x_num = batch["x_num"].to(device)
        y_img = batch["y_img"].to(device)
        y_num = batch["y_num"].to(device)

        if is_training:
            outputs = model(x_img, x_num)

            loss, metrics = gemininet_loss(
                outputs=outputs,
                y_img=y_img,
                y_num=y_num,
                lambda_img=args.lambda_img,
                lambda_num=args.lambda_num,
                lambda_rel=args.lambda_rel,
                tau_img=args.tau_img,
                tau_num=args.tau_num,
                lambda_img_mae=getattr(args, "lambda_img_mae", 0.0),
                num_loss_type=getattr(args, "num_loss_type", "mae"),
                num_huber_delta=getattr(args, "num_huber_delta", 1.0),
                reliability_label_mode=getattr(args, "reliability_label_mode", "mixed"),
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite GeminiNet loss detected. " f"Metrics: {metrics}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.grad_clip,
                )

            optimizer.step()

        else:
            with torch.no_grad():
                outputs = model(x_img, x_num)

                _, metrics = gemininet_loss(
                    outputs=outputs,
                    y_img=y_img,
                    y_num=y_num,
                    lambda_img=args.lambda_img,
                    lambda_num=args.lambda_num,
                    lambda_rel=args.lambda_rel,
                    tau_img=args.tau_img,
                    tau_num=args.tau_num,
                    lambda_img_mae=getattr(args, "lambda_img_mae", 0.0),
                    num_loss_type=getattr(args, "num_loss_type", "mae"),
                    num_huber_delta=getattr(args, "num_huber_delta", 1.0),
                    reliability_label_mode=getattr(args, "reliability_label_mode", "mixed"),
                )

        if getattr(args, "metric_inverse_transform", True):
            _add_original_scale_metrics(
                metrics=metrics,
                normalizer=normalizer,
                pred_img=outputs["y_img"],
                y_img=y_img,
                pred_num=outputs["y_num"],
                y_num=y_num,
                include_image_quality=True,
            )

        norm_data_min, norm_data_range = _target_derived_image_range(y_img)
        metrics.update(
            image_quality_metric_values(
                pred=outputs["y_img"],
                target=y_img,
                data_min=norm_data_min,
                data_range=norm_data_range,
                space="norm",
            )
        )

        batch_size = x_img.size(0)

        _accumulate_metrics(
            total_metrics=total_metrics,
            metrics=metrics,
            batch_size=batch_size,
        )

        total_count += batch_size

    return _average_metrics(total_metrics, total_count)


@torch.no_grad()
def test_gemininet(
    args,
    loaders: Dict,
    device,
    image_ckpt: str | Path,
    num_ckpt: str | Path,
    gemini_ckpt: str | Path,
    normalizer=None,
) -> dict:
    """Evaluate GeminiNet on the final test set and save full JSON metrics."""
    model = build_gemininet(
        args=args,
        image_ckpt=image_ckpt,
        num_ckpt=num_ckpt,
        device=device,
    )

    model = load_model_weights(model, gemini_ckpt, device)

    test_metrics = run_gemininet_epoch(
        model=model,
        loader=loaders["test"],
        device=device,
        optimizer=None,
        args=args,
        normalizer=normalizer,
    )

    print(
        "[GeminiNet Test] "
        f"total={test_metrics['total_loss']:.8f} "
        f"{_format_prefixed_metrics('test', test_metrics, ['img_mse_norm', 'img_mae_norm', 'img_rmse_norm', 'img_psnr_norm', 'img_ms_ssim_norm', 'num_mse_norm', 'num_mae_norm', 'num_rmse_norm'])} "
        f"{_format_prefixed_metrics('test', test_metrics, ['img_mse_raw', 'img_mae_raw', 'img_rmse_raw', 'img_psnr_raw', 'img_ms_ssim_raw', 'num_mse_raw', 'num_mae_raw', 'num_rmse_raw'])}"
    )

    save_json(
        path=Path(args.exp_dir) / "logs" / "gemininet_test_metrics.json",
        data=test_metrics,
    )

    if getattr(args, "visualize_train", getattr(args, "visualize", True)) or getattr(
        args, "visualize_vali", getattr(args, "visualize", True)
    ):
        from utils.visualization import visualize_multimodal_batch

        visualize_multimodal_batch(
            model=model,
            loader=loaders["test"],
            device=device,
            save_path=Path(args.exp_dir)
            / "visualization"
            / "test"
            / "gemininet_test_prediction.png",
            sample_count=2,
            title="GeminiNet Test Prediction",
            random_samples=args.vis_random_samples,
            normalizer=normalizer,
            inverse_transform=getattr(args, "vis_inverse_transform", True),
        )

    return test_metrics