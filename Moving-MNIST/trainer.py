from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from models import GeminiNet, ImageUNetPrior, NumericalTransformerPrior
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import JSONLogger, save_json
from utils.metrics import (
    build_reliability_labels,
    gemininet_loss,
    image_quality_metrics,
)

from utils.model_summary import print_model_parameters

SEMANTIC_CONFIG_KEYS = [
    "input_len_img",
    "input_len_num",
    "pred_len_img",
    "pred_len_num",
    "img_size",
    "img_channels",
    "num_vars",
    "img_norm_type",
    "num_norm_type",
    "img_output_activation",
    "num_output_activation",
    "output_scale",
]


def _checkpoint_metadata(
    args,
    normalizers=None,
    runtime: dict | None = None,
) -> dict:
    metadata = {
        "args": vars(args),
        "semantic_config": {
            key: getattr(args, key) for key in SEMANTIC_CONFIG_KEYS if hasattr(args, key)
        },
    }

    if normalizers is not None:
        metadata["normalizers"] = {
            name: normalizer.to_dict() for name, normalizer in normalizers.items()
        }

    if runtime is not None:
        metadata.update(runtime)

    return metadata


def _validate_checkpoint_config(
    checkpoint: dict,
    args,
    config_keys: tuple[str, ...] | None = None,
) -> None:
    saved = checkpoint.get("semantic_config")

    if saved is None:
        return

    mismatches = []

    for key, saved_value in saved.items():
        if config_keys is not None and key not in config_keys:
            continue

        if hasattr(args, key) and getattr(args, key) != saved_value:
            mismatches.append(
                f"{key}: checkpoint={saved_value!r}, " f"current={getattr(args, key)!r}"
            )

    if mismatches:
        raise ValueError("Checkpoint configuration is incompatible:\n- " + "\n- ".join(mismatches))


def _validate_checkpoint_normalizers(
    checkpoint: dict,
    normalizers,
) -> None:
    if normalizers is None or "normalizers" not in checkpoint:
        return

    current = {name: normalizer.to_dict() for name, normalizer in normalizers.items()}

    if checkpoint["normalizers"] != current:
        raise ValueError(
            "Checkpoint normalization statistics do not match the "
            "normalizers loaded for this run."
        )


def _forecast_basic_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """Return MSE/MAE/RMSE for tensors with the same shape."""
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()

    return {
        "mse": mse,
        "mae": mae,
        "rmse": mse**0.5,
    }


def _add_weighted_forecast_metrics(
    totals: dict,
    prefix: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
) -> None:
    """Accumulate weighted forecasting metrics into totals."""
    metrics = _forecast_basic_metrics(pred, target)

    totals[f"{prefix}_mse"] += metrics["mse"] * batch_size
    totals[f"{prefix}_mae"] += metrics["mae"] * batch_size


def _image_data_range(
    target: torch.Tensor,
    normalizer=None,
    raw: bool = False,
) -> float:
    """Choose a stable image data range for PSNR/MS-SSIM evaluation."""
    if normalizer is not None:
        norm_type = getattr(normalizer, "norm_type", "")
        if raw and norm_type == "scale_01":
            return float(getattr(normalizer, "scale_01_divisor", 255.0))
        if not raw and norm_type in {"scale_01", "minmax"}:
            return 1.0

    with torch.no_grad():
        finite = target.detach()[torch.isfinite(target.detach())]
        if finite.numel() == 0:
            return 1.0
        value_range = float((finite.max() - finite.min()).clamp_min(1e-6).cpu())
        return max(value_range, 1.0)


def _add_weighted_image_quality_metrics(
    totals: dict,
    prefix: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    data_range: float,
) -> None:
    """Accumulate PSNR/MS-SSIM image quality metrics."""
    metrics = image_quality_metrics(
        pred=pred.detach(),
        target=target.detach(),
        data_range=data_range,
    )
    totals[f"{prefix}_psnr"] += metrics["psnr"] * batch_size
    totals[f"{prefix}_ms_ssim"] += metrics["ms_ssim"] * batch_size


def _finalize_forecast_metrics(
    totals: dict,
    prefixes: list[str],
    total_count: int,
) -> dict:
    """Average MSE/MAE and derive RMSE from averaged MSE."""
    output = {}
    denominator = max(1, total_count)

    for prefix in prefixes:
        if prefix.endswith("_norm"):
            base = prefix.removesuffix("_norm")
            suffix = "norm"
        elif prefix.endswith("_raw"):
            base = prefix.removesuffix("_raw")
            suffix = "raw"
        else:
            base = prefix
            suffix = ""

        metric_prefix = f"{base}_" if base else ""
        metric_suffix = f"_{suffix}" if suffix else ""
        if f"{prefix}_mse" in totals:
            mse = totals[f"{prefix}_mse"] / denominator
            mae = totals[f"{prefix}_mae"] / denominator
            output[f"{metric_prefix}mse{metric_suffix}"] = mse
            output[f"{metric_prefix}mae{metric_suffix}"] = mae
            output[f"{metric_prefix}rmse{metric_suffix}"] = mse**0.5
        if f"{prefix}_psnr" in totals:
            output[f"{metric_prefix}psnr{metric_suffix}"] = totals[f"{prefix}_psnr"] / denominator
            output[f"{metric_prefix}ms_ssim{metric_suffix}"] = (
                totals[f"{prefix}_ms_ssim"] / denominator
            )

    return output


def build_image_prior(args) -> ImageUNetPrior:
    """Create image U-Net tendency model."""
    return ImageUNetPrior(
        input_len=args.input_len_img,
        pred_len=args.pred_len_img,
        img_channels=args.img_channels,
        base_channels=args.unet_base_channels,
        dropout=args.dropout,
        img_output_activation=args.img_output_activation,
        output_scale=args.output_scale,
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
        num_output_activation=args.num_output_activation,
        output_scale=args.output_scale,
    )


def load_model_weights(
    model,
    ckpt_path: str | Path,
    device,
    args=None,
    strict: bool = True,
    allowed_missing_prefixes: tuple[str, ...] = (),
    config_keys: tuple[str, ...] | None = None,
    normalizers=None,
):
    """Load checkpoint weights into a model."""
    checkpoint = load_checkpoint(
        path=ckpt_path,
        map_location=device,
    )

    if args is not None:
        _validate_checkpoint_config(
            checkpoint,
            args,
            config_keys=config_keys,
        )

    _validate_checkpoint_normalizers(checkpoint, normalizers)

    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )

    if not strict:
        invalid_missing = [
            key for key in incompatible.missing_keys if not key.startswith(allowed_missing_prefixes)
        ]

        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint state mismatch. "
                f"Missing={invalid_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    return model


def train_image_prior(
    args,
    loaders: Dict,
    device,
    normalizers=None,
) -> Path:
    """Train the image tendency model without accessing the test split."""
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

    for epoch in range(1, args.epochs_prior + 1):
        train_metrics = run_image_prior_epoch(
            model=model,
            loader=loaders["prior_train"],
            device=device,
            optimizer=optimizer,
            normalizers=normalizers,
        )

        val_metrics = run_image_prior_epoch(
            model=model,
            loader=loaders["val"],
            device=device,
            optimizer=None,
            normalizers=normalizers,
        )

        row = {
            "epoch": epoch,
            "train_img_mse": train_metrics["img_mse_norm"],
            "train_img_mae": train_metrics["img_mae_norm"],
            "val_img_mse": val_metrics["img_mse_norm"],
            "val_img_mae": val_metrics["img_mae_norm"],
            "train_img_mse_norm": train_metrics["img_mse_norm"],
            "train_img_mae_norm": train_metrics["img_mae_norm"],
            "train_img_rmse_norm": train_metrics["img_rmse_norm"],
            "train_img_psnr_norm": train_metrics["img_psnr_norm"],
            "train_img_ms_ssim_norm": train_metrics["img_ms_ssim_norm"],
            "val_img_mse_norm": val_metrics["img_mse_norm"],
            "val_img_mae_norm": val_metrics["img_mae_norm"],
            "val_img_rmse_norm": val_metrics["img_rmse_norm"],
            "val_img_psnr_norm": val_metrics["img_psnr_norm"],
            "val_img_ms_ssim_norm": val_metrics["img_ms_ssim_norm"],
            "train_img_mse_raw": train_metrics["img_mse_raw"],
            "train_img_mae_raw": train_metrics["img_mae_raw"],
            "train_img_rmse_raw": train_metrics["img_rmse_raw"],
            "train_img_psnr_raw": train_metrics["img_psnr_raw"],
            "train_img_ms_ssim_raw": train_metrics["img_ms_ssim_raw"],
            "val_img_mse_raw": val_metrics["img_mse_raw"],
            "val_img_mae_raw": val_metrics["img_mae_raw"],
            "val_img_rmse_raw": val_metrics["img_rmse_raw"],
            "val_img_psnr_raw": val_metrics["img_psnr_raw"],
            "val_img_ms_ssim_raw": val_metrics["img_ms_ssim_raw"],
        }

        logger.log(row)

        print(
            f"[Image Prior] "
            f"epoch={epoch:03d} "
            f"train_img_mse_norm={train_metrics['img_mse_norm']:.8f} "
            f"train_img_mae_norm={train_metrics['img_mae_norm']:.8f} "
            f"train_img_rmse_norm={train_metrics['img_rmse_norm']:.8f} "
            f"train_img_psnr_norm={train_metrics['img_psnr_norm']:.8f} "
            f"train_img_ms_ssim_norm={train_metrics['img_ms_ssim_norm']:.8f} "
            f"val_img_mse_norm={val_metrics['img_mse_norm']:.8f} "
            f"val_img_mae_norm={val_metrics['img_mae_norm']:.8f} "
            f"val_img_rmse_norm={val_metrics['img_rmse_norm']:.8f} "
            f"val_img_psnr_norm={val_metrics['img_psnr_norm']:.8f} "
            f"val_img_ms_ssim_norm={val_metrics['img_ms_ssim_norm']:.8f} "
            f"train_img_mse_raw={train_metrics['img_mse_raw']:.8f} "
            f"train_img_mae_raw={train_metrics['img_mae_raw']:.8f} "
            f"train_img_rmse_raw={train_metrics['img_rmse_raw']:.8f} "
            f"train_img_psnr_raw={train_metrics['img_psnr_raw']:.8f} "
            f"train_img_ms_ssim_raw={train_metrics['img_ms_ssim_raw']:.8f} "
            f"val_img_mse_raw={val_metrics['img_mse_raw']:.8f} "
            f"val_img_mae_raw={val_metrics['img_mae_raw']:.8f} "
            f"val_img_rmse_raw={val_metrics['img_rmse_raw']:.8f} "
            f"val_img_psnr_raw={val_metrics['img_psnr_raw']:.8f} "
            f"val_img_ms_ssim_raw={val_metrics['img_ms_ssim_raw']:.8f}"
        )

        if val_metrics["img_mse_norm"] < best_val:
            best_val = val_metrics["img_mse_norm"]

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=None,
                epoch=epoch,
                best_metric=best_val,
                extra=_checkpoint_metadata(args, normalizers),
            )

        save_checkpoint(
            path=save_dir / "image_prior_last.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_val,
            extra=_checkpoint_metadata(args, normalizers),
        )

    return save_dir / "image_prior_last.pth"


def run_image_prior_epoch(
    model,
    loader,
    device,
    optimizer,
    normalizers=None,
) -> dict:
    """Train or evaluate the image prior model for one epoch."""
    is_training = optimizer is not None
    model.train(is_training)

    totals = {
        "img_norm_mse": 0.0,
        "img_norm_mae": 0.0,
        "img_norm_psnr": 0.0,
        "img_norm_ms_ssim": 0.0,
        "img_raw_mse": 0.0,
        "img_raw_mae": 0.0,
        "img_raw_psnr": 0.0,
        "img_raw_ms_ssim": 0.0,
    }
    total_count = 0

    for batch in loader:
        x_img = batch["x_img"].to(device)
        y_img = batch["y_img"].to(device)
        y_img_raw = batch["y_img_raw"].to(device)

        if is_training:
            pred_img = model(x_img)
            loss = F.mse_loss(pred_img, y_img)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                pred_img = model(x_img)

        batch_size = x_img.size(0)

        _add_weighted_forecast_metrics(
            totals,
            "img_norm",
            pred_img,
            y_img,
            batch_size,
        )
        _add_weighted_image_quality_metrics(
            totals,
            "img_norm",
            pred_img,
            y_img,
            batch_size,
            data_range=_image_data_range(
                y_img,
                normalizers["image"] if normalizers is not None else None,
                raw=False,
            ),
        )

        if normalizers is not None:
            with torch.no_grad():
                pred_img_raw = normalizers["image"].inverse_transform(pred_img.detach())
            _add_weighted_forecast_metrics(
                totals,
                "img_raw",
                pred_img_raw,
                y_img_raw,
                batch_size,
            )
            _add_weighted_image_quality_metrics(
                totals,
                "img_raw",
                pred_img_raw,
                y_img_raw,
                batch_size,
                data_range=_image_data_range(
                    y_img_raw,
                    normalizers["image"],
                    raw=True,
                ),
            )
        else:
            _add_weighted_forecast_metrics(
                totals,
                "img_raw",
                pred_img.detach(),
                y_img,
                batch_size,
            )
            _add_weighted_image_quality_metrics(
                totals,
                "img_raw",
                pred_img.detach(),
                y_img,
                batch_size,
                data_range=_image_data_range(y_img, raw=True),
            )

        total_count += batch_size

    return _finalize_forecast_metrics(
        totals,
        prefixes=["img_norm", "img_raw"],
        total_count=total_count,
    )


def train_num_prior(
    args,
    loaders: Dict,
    device,
    normalizers=None,
) -> Path:
    """Train the numerical tendency model without accessing the test split."""
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

    for epoch in range(1, args.epochs_prior + 1):
        train_metrics = run_num_prior_epoch(
            model=model,
            loader=loaders["prior_train"],
            device=device,
            optimizer=optimizer,
            normalizers=normalizers,
        )

        val_metrics = run_num_prior_epoch(
            model=model,
            loader=loaders["val"],
            device=device,
            optimizer=None,
            normalizers=normalizers,
        )

        row = {
            "epoch": epoch,
            "train_num_mse": train_metrics["num_mse_norm"],
            "train_num_mae": train_metrics["num_mae_norm"],
            "val_num_mse": val_metrics["num_mse_norm"],
            "val_num_mae": val_metrics["num_mae_norm"],
            "train_num_mse_norm": train_metrics["num_mse_norm"],
            "train_num_mae_norm": train_metrics["num_mae_norm"],
            "train_num_rmse_norm": train_metrics["num_rmse_norm"],
            "val_num_mse_norm": val_metrics["num_mse_norm"],
            "val_num_mae_norm": val_metrics["num_mae_norm"],
            "val_num_rmse_norm": val_metrics["num_rmse_norm"],
            "train_num_mse_raw": train_metrics["num_mse_raw"],
            "train_num_mae_raw": train_metrics["num_mae_raw"],
            "train_num_rmse_raw": train_metrics["num_rmse_raw"],
            "val_num_mse_raw": val_metrics["num_mse_raw"],
            "val_num_mae_raw": val_metrics["num_mae_raw"],
            "val_num_rmse_raw": val_metrics["num_rmse_raw"],
        }

        logger.log(row)

        print(
            f"[Numerical Prior] "
            f"epoch={epoch:03d} "
            f"train_num_mse_norm={train_metrics['num_mse_norm']:.8f} "
            f"train_num_mae_norm={train_metrics['num_mae_norm']:.8f} "
            f"train_num_rmse_norm={train_metrics['num_rmse_norm']:.8f} "
            f"val_num_mse_norm={val_metrics['num_mse_norm']:.8f} "
            f"val_num_mae_norm={val_metrics['num_mae_norm']:.8f} "
            f"val_num_rmse_norm={val_metrics['num_rmse_norm']:.8f} "
            f"train_num_mse_raw={train_metrics['num_mse_raw']:.8f} "
            f"train_num_mae_raw={train_metrics['num_mae_raw']:.8f} "
            f"train_num_rmse_raw={train_metrics['num_rmse_raw']:.8f} "
            f"val_num_mse_raw={val_metrics['num_mse_raw']:.8f} "
            f"val_num_mae_raw={val_metrics['num_mae_raw']:.8f} "
            f"val_num_rmse_raw={val_metrics['num_rmse_raw']:.8f}"
        )

        if val_metrics["num_mse_norm"] < best_val:
            best_val = val_metrics["num_mse_norm"]

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=None,
                epoch=epoch,
                best_metric=best_val,
                extra=_checkpoint_metadata(args, normalizers),
            )

        save_checkpoint(
            path=save_dir / "num_prior_last.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_val,
            extra=_checkpoint_metadata(args, normalizers),
        )

    return save_dir / "num_prior_last.pth"


def run_num_prior_epoch(
    model,
    loader,
    device,
    optimizer,
    normalizers=None,
) -> dict:
    """Train or evaluate the numerical prior model for one epoch."""
    is_training = optimizer is not None
    model.train(is_training)

    totals = {
        "num_norm_mse": 0.0,
        "num_norm_mae": 0.0,
        "num_raw_mse": 0.0,
        "num_raw_mae": 0.0,
    }
    total_count = 0

    for batch in loader:
        x_num = batch["x_num"].to(device)
        y_num = batch["y_num"].to(device)
        y_num_raw = batch["y_num_raw"].to(device)

        if is_training:
            pred_num = model(x_num)
            loss = F.mse_loss(pred_num, y_num)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                pred_num = model(x_num)

        batch_size = x_num.size(0)

        _add_weighted_forecast_metrics(
            totals,
            "num_norm",
            pred_num,
            y_num,
            batch_size,
        )

        if normalizers is not None:
            with torch.no_grad():
                pred_num_raw = normalizers["numerical"].inverse_transform(pred_num.detach())
            _add_weighted_forecast_metrics(
                totals,
                "num_raw",
                pred_num_raw,
                y_num_raw,
                batch_size,
            )
        else:
            _add_weighted_forecast_metrics(
                totals,
                "num_raw",
                pred_num.detach(),
                y_num,
                batch_size,
            )

        total_count += batch_size

    return _finalize_forecast_metrics(
        totals,
        prefixes=["num_norm", "num_raw"],
        total_count=total_count,
    )


def build_gemininet(
    args,
    image_ckpt: str | Path,
    num_ckpt: str | Path,
    device,
    normalizers=None,
) -> GeminiNet:
    """Build GeminiNet and load frozen tendency generators."""
    image_prior = build_image_prior(args).to(device)
    num_prior = build_num_prior(args).to(device)

    image_prior = load_model_weights(
        image_prior,
        image_ckpt,
        device,
        args=args,
        normalizers=normalizers,
        config_keys=(
            "input_len_img",
            "pred_len_img",
            "img_channels",
            "img_norm_type",
            "img_output_activation",
            "output_scale",
        ),
    )
    num_prior = load_model_weights(
        num_prior,
        num_ckpt,
        device,
        args=args,
        normalizers=normalizers,
        config_keys=(
            "input_len_num",
            "pred_len_num",
            "num_vars",
            "num_norm_type",
            "num_output_activation",
            "output_scale",
        ),
    )

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
        img_output_activation=args.img_output_activation,
        num_output_activation=args.num_output_activation,
        output_scale=args.output_scale,
    )

    return model.to(device)


def _gemininet_loss_kwargs(args) -> dict:
    """Collect GeminiNet loss settings from args.

    Keeping this in one helper makes training, validation, and testing use
    exactly the same reliability-label and loss configuration.
    """
    return {
        "lambda_img": args.lambda_img,
        "lambda_num": args.lambda_num,
        "lambda_rel": args.lambda_rel,
        "tau_img": args.tau_img,
        "tau_num": args.tau_num,
        "lambda_img_mae": getattr(args, "lambda_img_mae", 0.05),
        "image_loss_type": getattr(args, "image_loss_type", "mse"),
        "image_huber_delta": getattr(args, "image_huber_delta", 1.0),
        "num_loss_type": getattr(args, "num_loss_type", "huber"),
        "huber_delta": getattr(args, "huber_delta", 1.0),
        "reliability_label_mode": getattr(
            args,
            "reliability_label_mode",
            "mixed",
        ),
        "reliability_label_mix": getattr(
            args,
            "reliability_label_mix",
            0.5,
        ),
        "reliability_label_floor": getattr(
            args,
            "reliability_label_floor",
            0.05,
        ),
        "rel_loss_type": getattr(args, "rel_loss_type", "smooth_l1"),
        "lambda_prior_consistency": getattr(
            args,
            "lambda_prior_consistency",
            0.01,
        ),
    }


def train_gemininet(
    args,
    loaders: Dict,
    device,
    image_ckpt: str | Path,
    num_ckpt: str | Path,
    normalizers=None,
) -> Path:
    """Train GeminiNet without accessing the official test split."""
    model = build_gemininet(
        args=args,
        image_ckpt=image_ckpt,
        num_ckpt=num_ckpt,
        device=device,
        normalizers=normalizers,
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

    lr_reliability_scale = getattr(args, "lr_reliability_scale", 1.0)

    if lr_reliability_scale != 1.0:
        reliability_parameters = []
        other_parameters = []

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            if "reliability_estimator" in name:
                reliability_parameters.append(parameter)
            else:
                other_parameters.append(parameter)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": other_parameters,
                    "lr": args.lr_gemini,
                },
                {
                    "params": reliability_parameters,
                    "lr": args.lr_gemini * lr_reliability_scale,
                },
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.lr_gemini,
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    save_dir = Path(args.exp_dir) / "checkpoints"
    vis_dir = Path(args.exp_dir) / "visualization"
    best_path = save_dir / "gemininet_best.pth"
    start_epoch = 1
    best_val = float("inf")
    epochs_without_improvement = 0

    if args.resume_gemininet_ckpt:
        resume_checkpoint = load_checkpoint(
            args.resume_gemininet_ckpt,
            map_location=device,
        )
        _validate_checkpoint_config(resume_checkpoint, args)
        _validate_checkpoint_normalizers(
            resume_checkpoint,
            normalizers,
        )
        incompatible = model.load_state_dict(
            resume_checkpoint["model_state_dict"],
            strict=False,
        )
        invalid_missing = [
            key for key in incompatible.missing_keys if not key.startswith("tendency_generator.")
        ]

        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Cannot resume GeminiNet due to state mismatch. "
                f"Missing={invalid_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

        if "optimizer_state_dict" not in resume_checkpoint:
            raise ValueError(
                "The resume checkpoint has no optimizer state. "
                "Use gemininet_last.pth rather than the compact best "
                "checkpoint."
            )

        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])

        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_val = float(resume_checkpoint.get("best_metric", float("inf")))
        epochs_without_improvement = int(resume_checkpoint.get("epochs_without_improvement", 0))

    logger = JSONLogger(
        path=Path(args.exp_dir) / "logs" / "gemininet_metrics.json",
        reset=not bool(args.resume_gemininet_ckpt),
    )

    for epoch in range(start_epoch, args.epochs_gemini + 1):
        train_metrics = run_gemininet_epoch(
            model=model,
            loader=loaders["gemini_train"],
            device=device,
            optimizer=optimizer,
            args=args,
            normalizers=normalizers,
        )

        val_metrics = run_gemininet_epoch(
            model=model,
            loader=loaders["val"],
            device=device,
            optimizer=None,
            args=args,
            normalizers=normalizers,
        )
        selection_metric = (
            args.lambda_img * val_metrics["image_task_loss"]
            + args.lambda_num * val_metrics["num_task_loss"]
        )

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_selection_metric": selection_metric,
            "train_total_loss": train_metrics["total_loss"],
            "train_image_mse": train_metrics["image_mse"],
            "train_image_mae": train_metrics["image_mae"],
            "train_image_rmse": train_metrics["image_rmse"],
            "train_img_mse_norm": train_metrics["img_mse_norm"],
            "train_img_mae_norm": train_metrics["img_mae_norm"],
            "train_img_rmse_norm": train_metrics["img_rmse_norm"],
            "train_img_psnr_norm": train_metrics["img_psnr_norm"],
            "train_img_ms_ssim_norm": train_metrics["img_ms_ssim_norm"],
            "train_img_mse_raw": train_metrics["img_mse_raw"],
            "train_img_mae_raw": train_metrics["img_mae_raw"],
            "train_img_rmse_raw": train_metrics["img_rmse_raw"],
            "train_img_psnr_raw": train_metrics["img_psnr_raw"],
            "train_img_ms_ssim_raw": train_metrics["img_ms_ssim_raw"],
            "train_image_task_loss": train_metrics["image_task_loss"],
            "train_num_mse": train_metrics["num_mse"],
            "train_num_mae": train_metrics["num_mae"],
            "train_num_rmse": train_metrics["num_rmse"],
            "train_num_mse_norm": train_metrics["num_mse_norm"],
            "train_num_mae_norm": train_metrics["num_mae_norm"],
            "train_num_rmse_norm": train_metrics["num_rmse_norm"],
            "train_num_mse_raw": train_metrics["num_mse_raw"],
            "train_num_mae_raw": train_metrics["num_mae_raw"],
            "train_num_rmse_raw": train_metrics["num_rmse_raw"],
            "train_num_task_loss": train_metrics["num_task_loss"],
            "train_rel_loss": train_metrics["rel_loss"],
            "train_rel_img_loss": train_metrics["rel_img_loss"],
            "train_rel_num_loss": train_metrics["rel_num_loss"],
            "train_prior_consistency_loss": train_metrics["prior_consistency_loss"],
            "train_r_img_mean": train_metrics["r_img_mean"],
            "train_r_num_mean": train_metrics["r_num_mean"],
            "train_r_img_label_mean": train_metrics["r_img_label_mean"],
            "train_r_num_label_mean": train_metrics["r_num_label_mean"],
            "val_total_loss": val_metrics["total_loss"],
            "val_image_mse": val_metrics["image_mse"],
            "val_image_mae": val_metrics["image_mae"],
            "val_image_rmse": val_metrics["image_rmse"],
            "val_img_mse_norm": val_metrics["img_mse_norm"],
            "val_img_mae_norm": val_metrics["img_mae_norm"],
            "val_img_rmse_norm": val_metrics["img_rmse_norm"],
            "val_img_psnr_norm": val_metrics["img_psnr_norm"],
            "val_img_ms_ssim_norm": val_metrics["img_ms_ssim_norm"],
            "val_img_mse_raw": val_metrics["img_mse_raw"],
            "val_img_mae_raw": val_metrics["img_mae_raw"],
            "val_img_rmse_raw": val_metrics["img_rmse_raw"],
            "val_img_psnr_raw": val_metrics["img_psnr_raw"],
            "val_img_ms_ssim_raw": val_metrics["img_ms_ssim_raw"],
            "val_image_task_loss": val_metrics["image_task_loss"],
            "val_num_mse": val_metrics["num_mse"],
            "val_num_mae": val_metrics["num_mae"],
            "val_num_rmse": val_metrics["num_rmse"],
            "val_num_mse_norm": val_metrics["num_mse_norm"],
            "val_num_mae_norm": val_metrics["num_mae_norm"],
            "val_num_rmse_norm": val_metrics["num_rmse_norm"],
            "val_num_mse_raw": val_metrics["num_mse_raw"],
            "val_num_mae_raw": val_metrics["num_mae_raw"],
            "val_num_rmse_raw": val_metrics["num_rmse_raw"],
            "val_num_task_loss": val_metrics["num_task_loss"],
            "val_rel_loss": val_metrics["rel_loss"],
            "val_rel_img_loss": val_metrics["rel_img_loss"],
            "val_rel_num_loss": val_metrics["rel_num_loss"],
            "val_prior_consistency_loss": val_metrics["prior_consistency_loss"],
            "val_r_img_mean": val_metrics["r_img_mean"],
            "val_r_num_mean": val_metrics["r_num_mean"],
            "val_r_img_label_mean": val_metrics["r_img_label_mean"],
            "val_r_num_label_mean": val_metrics["r_num_label_mean"],
            "use_reliability_estimator": train_metrics["use_reliability_estimator"],
        }

        logger.log(row)

        print(
            f"[GeminiNet] "
            f"epoch={epoch:03d} "
            f"train_total={train_metrics['total_loss']:.8f} "
            f"train_img_mse_norm={train_metrics['img_mse_norm']:.8f} "
            f"train_img_mae_norm={train_metrics['img_mae_norm']:.8f} "
            f"train_img_rmse_norm={train_metrics['img_rmse_norm']:.8f} "
            f"train_img_psnr_norm={train_metrics['img_psnr_norm']:.8f} "
            f"train_img_ms_ssim_norm={train_metrics['img_ms_ssim_norm']:.8f} "
            f"train_num_mse_norm={train_metrics['num_mse_norm']:.8f} "
            f"train_num_mae_norm={train_metrics['num_mae_norm']:.8f} "
            f"train_num_rmse_norm={train_metrics['num_rmse_norm']:.8f} "
            f"train_img_mse_raw={train_metrics['img_mse_raw']:.8f} "
            f"train_img_mae_raw={train_metrics['img_mae_raw']:.8f} "
            f"train_img_rmse_raw={train_metrics['img_rmse_raw']:.8f} "
            f"train_img_psnr_raw={train_metrics['img_psnr_raw']:.8f} "
            f"train_img_ms_ssim_raw={train_metrics['img_ms_ssim_raw']:.8f} "
            f"train_num_mse_raw={train_metrics['num_mse_raw']:.8f} "
            f"train_num_mae_raw={train_metrics['num_mae_raw']:.8f} "
            f"train_num_rmse_raw={train_metrics['num_rmse_raw']:.8f} "
            f"train_rel={train_metrics['rel_loss']:.8f} "
            f"train_r_img={train_metrics['r_img_mean']:.4f} "
            f"train_r_num={train_metrics['r_num_mean']:.4f} "
            f"val_total={val_metrics['total_loss']:.8f} "
            f"val_img_mse_norm={val_metrics['img_mse_norm']:.8f} "
            f"val_img_mae_norm={val_metrics['img_mae_norm']:.8f} "
            f"val_img_rmse_norm={val_metrics['img_rmse_norm']:.8f} "
            f"val_img_psnr_norm={val_metrics['img_psnr_norm']:.8f} "
            f"val_img_ms_ssim_norm={val_metrics['img_ms_ssim_norm']:.8f} "
            f"val_num_mse_norm={val_metrics['num_mse_norm']:.8f} "
            f"val_num_mae_norm={val_metrics['num_mae_norm']:.8f} "
            f"val_num_rmse_norm={val_metrics['num_rmse_norm']:.8f} "
            f"val_img_mse_raw={val_metrics['img_mse_raw']:.8f} "
            f"val_img_mae_raw={val_metrics['img_mae_raw']:.8f} "
            f"val_img_rmse_raw={val_metrics['img_rmse_raw']:.8f} "
            f"val_img_psnr_raw={val_metrics['img_psnr_raw']:.8f} "
            f"val_img_ms_ssim_raw={val_metrics['img_ms_ssim_raw']:.8f} "
            f"val_num_mse_raw={val_metrics['num_mse_raw']:.8f} "
            f"val_num_mae_raw={val_metrics['num_mae_raw']:.8f} "
            f"val_num_rmse_raw={val_metrics['num_rmse_raw']:.8f} "
            f"val_rel={val_metrics['rel_loss']:.8f} "
            f"val_r_img={val_metrics['r_img_mean']:.4f} "
            f"val_r_num={val_metrics['r_num_mean']:.4f}"
        )

        improved = selection_metric < best_val

        if improved:
            best_val = selection_metric
            epochs_without_improvement = 0

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=None,
                epoch=epoch,
                best_metric=best_val,
                extra=_checkpoint_metadata(
                    args,
                    normalizers,
                    runtime={
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epochs_without_improvement": 0,
                    },
                ),
                exclude_prefixes=("tendency_generator.",),
            )
        else:
            epochs_without_improvement += 1

        scheduler.step(selection_metric)

        save_checkpoint(
            path=save_dir / "gemininet_last.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_val,
            extra=_checkpoint_metadata(
                args,
                normalizers,
                runtime={
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epochs_without_improvement": (epochs_without_improvement),
                },
            ),
            exclude_prefixes=("tendency_generator.",),
        )

        if args.visualize and epoch % args.vis_interval == 0:
            from utils.visualization import visualize_multimodal_batch

            visualize_multimodal_batch(
                model=model,
                loader=loaders["gemini_train"],
                device=device,
                save_path=vis_dir / f"gemininet_epoch_{epoch:04d}.png",
                sample_count=2,
                title=f"GeminiNet Prediction Epoch {epoch}",
                random_samples=args.vis_random_samples,
                img_norm_type=args.img_norm_type,
                img_output_activation=args.img_output_activation,
            )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "[GeminiNet] Early stopping after "
                f"{epochs_without_improvement} unimproved epochs."
            )
            break

    if best_path.exists():
        return best_path

    if args.resume_gemininet_ckpt:
        return Path(args.resume_gemininet_ckpt)

    return best_path


def run_gemininet_epoch(
    model,
    loader,
    device,
    optimizer,
    args,
    normalizers=None,
) -> dict:
    """Run one epoch for GeminiNet training or evaluation."""
    is_training = optimizer is not None
    model.train(is_training)

    metric_keys = [
        "total_loss",
        "image_mse",
        "image_mae",
        "image_task_loss",
        "num_mse",
        "num_mae",
        "num_task_loss",
        "rel_loss",
        "rel_img_loss",
        "rel_num_loss",
        "prior_consistency_loss",
        "r_img_mean",
        "r_num_mean",
        "r_img_label_mean",
        "r_num_label_mean",
        "use_reliability_estimator",
    ]

    total_metrics = {key: 0.0 for key in metric_keys}
    forecast_totals = {
        "img_norm_mse": 0.0,
        "img_norm_mae": 0.0,
        "img_norm_psnr": 0.0,
        "img_norm_ms_ssim": 0.0,
        "img_raw_mse": 0.0,
        "img_raw_mae": 0.0,
        "img_raw_psnr": 0.0,
        "img_raw_ms_ssim": 0.0,
        "num_norm_mse": 0.0,
        "num_norm_mae": 0.0,
        "num_raw_mse": 0.0,
        "num_raw_mae": 0.0,
    }

    total_count = 0

    for batch in loader:
        x_img = batch["x_img"].to(device)
        x_num = batch["x_num"].to(device)
        y_img = batch["y_img"].to(device)
        y_num = batch["y_num"].to(device)
        y_img_raw = batch["y_img_raw"].to(device)
        y_num_raw = batch["y_num_raw"].to(device)

        if is_training:
            outputs = model(x_img, x_num)

            loss, metrics = gemininet_loss(
                outputs=outputs,
                y_img=y_img,
                y_num=y_num,
                **_gemininet_loss_kwargs(args),
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
                    **_gemininet_loss_kwargs(args),
                )

        batch_size = x_img.size(0)

        for key in metric_keys:
            total_metrics[key] += metrics[key] * batch_size

        pred_img = outputs["y_img"]
        pred_num = outputs["y_num"]
        _add_weighted_forecast_metrics(
            forecast_totals,
            "img_norm",
            pred_img.detach(),
            y_img,
            batch_size,
        )
        _add_weighted_image_quality_metrics(
            forecast_totals,
            "img_norm",
            pred_img,
            y_img,
            batch_size,
            data_range=_image_data_range(
                y_img,
                normalizers["image"] if normalizers is not None else None,
                raw=False,
            ),
        )
        _add_weighted_forecast_metrics(
            forecast_totals,
            "num_norm",
            pred_num.detach(),
            y_num,
            batch_size,
        )

        if normalizers is not None:
            with torch.no_grad():
                pred_img_raw = normalizers["image"].inverse_transform(pred_img.detach())
                pred_num_raw = normalizers["numerical"].inverse_transform(pred_num.detach())
            _add_weighted_forecast_metrics(
                forecast_totals,
                "img_raw",
                pred_img_raw,
                y_img_raw,
                batch_size,
            )
            _add_weighted_image_quality_metrics(
                forecast_totals,
                "img_raw",
                pred_img_raw,
                y_img_raw,
                batch_size,
                data_range=_image_data_range(
                    y_img_raw,
                    normalizers["image"],
                    raw=True,
                ),
            )
            _add_weighted_forecast_metrics(
                forecast_totals,
                "num_raw",
                pred_num_raw,
                y_num_raw,
                batch_size,
            )
        else:
            _add_weighted_forecast_metrics(
                forecast_totals,
                "img_raw",
                pred_img.detach(),
                y_img,
                batch_size,
            )
            _add_weighted_image_quality_metrics(
                forecast_totals,
                "img_raw",
                pred_img.detach(),
                y_img,
                batch_size,
                data_range=_image_data_range(y_img, raw=True),
            )
            _add_weighted_forecast_metrics(
                forecast_totals,
                "num_raw",
                pred_num.detach(),
                y_num,
                batch_size,
            )

        total_count += batch_size

    output = {key: value / max(1, total_count) for key, value in total_metrics.items()}
    output["image_rmse"] = output["image_mse"] ** 0.5
    output["num_rmse"] = output["num_mse"] ** 0.5
    output.update(
        _finalize_forecast_metrics(
            forecast_totals,
            prefixes=["img_norm", "img_raw", "num_norm", "num_raw"],
            total_count=total_count,
        )
    )
    return output


@torch.no_grad()
def evaluate_forecasts(
    model,
    loader,
    device,
    normalizers,
    args,
) -> dict:
    """Compute horizon-wise and original-scale forecasting metrics."""
    model.eval()

    sums = {}
    total_count = 0
    reliability_records = {
        "r_img": [],
        "r_num": [],
        "img_error": [],
        "num_error": [],
        "r_img_label": [],
        "r_num_label": [],
    }

    for batch in loader:
        x_img = batch["x_img"].to(device)
        x_num = batch["x_num"].to(device)
        y_img = batch["y_img"].to(device)
        y_num = batch["y_num"].to(device)
        y_img_raw = batch["y_img_raw"].to(device)
        y_num_raw = batch["y_num_raw"].to(device)
        x_img_raw = batch["x_img_raw"].to(device)
        x_num_raw = batch["x_num_raw"].to(device)

        outputs = model(x_img, x_num)
        pred_img = outputs["y_img"]
        pred_num = outputs["y_num"]

        pred_img_raw = normalizers["image"].inverse_transform(pred_img)
        pred_num_raw = normalizers["numerical"].inverse_transform(pred_num)
        persistence_img = x_img[:, -1:].expand_as(y_img)
        persistence_num = x_num[:, -1:].expand_as(y_num)
        persistence_img_raw = x_img_raw[:, -1:].expand_as(y_img_raw)
        persistence_num_raw = x_num_raw[:, -1:].expand_as(y_num_raw)
        r_img_label, r_num_label = build_reliability_labels(
            p_img=outputs["p_img"],
            y_img=y_img,
            p_num=outputs["p_num"],
            y_num=y_num,
            tau_img=args.tau_img,
            tau_num=args.tau_num,
            mode=args.reliability_label_mode,
            mix=args.reliability_label_mix,
            floor=args.reliability_label_floor,
        )
        image_prior_error = (outputs["p_img"] - y_img).square().mean(dim=(2, 3, 4))
        num_prior_error = (outputs["p_num"] - y_num).abs().mean(dim=-1)
        img_norm_range = _image_data_range(
            y_img,
            normalizers["image"],
            raw=False,
        )
        img_raw_range = _image_data_range(
            y_img_raw,
            normalizers["image"],
            raw=True,
        )
        image_mse_by_horizon = (pred_img - y_img).square().mean(dim=(2, 3, 4))
        image_raw_mse_by_horizon = (pred_img_raw - y_img_raw).square().mean(dim=(2, 3, 4))
        image_psnr_by_horizon = 20.0 * torch.log10(
            pred_img.new_tensor(img_norm_range)
        ) - 10.0 * torch.log10(image_mse_by_horizon.clamp_min(1e-8))
        image_raw_psnr_by_horizon = 20.0 * torch.log10(
            pred_img_raw.new_tensor(img_raw_range)
        ) - 10.0 * torch.log10(image_raw_mse_by_horizon.clamp_min(1e-8))
        image_ms_ssim_by_horizon = (
            torch.stack(
                [
                    pred_img.new_tensor(
                        image_quality_metrics(
                            pred=pred_img[:, horizon : horizon + 1],
                            target=y_img[:, horizon : horizon + 1],
                            data_range=img_norm_range,
                        )["ms_ssim"]
                    )
                    for horizon in range(pred_img.shape[1])
                ]
            )
            .unsqueeze(0)
            .expand_as(image_mse_by_horizon)
        )
        image_raw_ms_ssim_by_horizon = (
            torch.stack(
                [
                    pred_img_raw.new_tensor(
                        image_quality_metrics(
                            pred=pred_img_raw[:, horizon : horizon + 1],
                            target=y_img_raw[:, horizon : horizon + 1],
                            data_range=img_raw_range,
                        )["ms_ssim"]
                    )
                    for horizon in range(pred_img_raw.shape[1])
                ]
            )
            .unsqueeze(0)
            .expand_as(image_raw_mse_by_horizon)
        )

        current = {
            "image_mse_by_horizon": image_mse_by_horizon,
            "image_mae_by_horizon": ((pred_img - y_img).abs().mean(dim=(2, 3, 4))),
            "image_psnr_by_horizon": image_psnr_by_horizon,
            "image_ms_ssim_by_horizon": image_ms_ssim_by_horizon,
            "num_mse_by_horizon": ((pred_num - y_num).square().mean(dim=-1)),
            "num_mae_by_horizon": ((pred_num - y_num).abs().mean(dim=-1)),
            "image_raw_mse_by_horizon": image_raw_mse_by_horizon,
            "image_raw_mae_by_horizon": ((pred_img_raw - y_img_raw).abs().mean(dim=(2, 3, 4))),
            "image_raw_psnr_by_horizon": image_raw_psnr_by_horizon,
            "image_raw_ms_ssim_by_horizon": image_raw_ms_ssim_by_horizon,
            "num_raw_mse_by_horizon": ((pred_num_raw - y_num_raw).square().mean(dim=-1)),
            "num_raw_mae_by_horizon": ((pred_num_raw - y_num_raw).abs().mean(dim=-1)),
            "num_raw_mae_by_variable": ((pred_num_raw - y_num_raw).abs().mean(dim=1)),
            "num_raw_mse_by_variable": ((pred_num_raw - y_num_raw).square().mean(dim=1)),
            "persistence_image_raw_mse_by_horizon": (
                (persistence_img_raw - y_img_raw).square().mean(dim=(2, 3, 4))
            ),
            "persistence_image_raw_mae_by_horizon": (
                (persistence_img_raw - y_img_raw).abs().mean(dim=(2, 3, 4))
            ),
            "persistence_num_raw_mse_by_horizon": (
                (persistence_num_raw - y_num_raw).square().mean(dim=-1)
            ),
            "persistence_num_raw_mae_by_horizon": (
                (persistence_num_raw - y_num_raw).abs().mean(dim=-1)
            ),
            "persistence_image_mse_by_horizon": (
                (persistence_img - y_img).square().mean(dim=(2, 3, 4))
            ),
            "persistence_num_mse_by_horizon": ((persistence_num - y_num).square().mean(dim=-1)),
        }

        batch_size = x_img.size(0)

        for key, value in current.items():
            batch_sum = value.sum(dim=0).detach().cpu()
            sums[key] = sums.get(key, 0) + batch_sum

        reliability_records["r_img"].append(outputs["r_img"].detach().cpu().flatten())
        reliability_records["r_num"].append(outputs["r_num"].detach().cpu().flatten())
        reliability_records["img_error"].append(image_prior_error.detach().cpu().flatten())
        reliability_records["num_error"].append(num_prior_error.detach().cpu().flatten())
        reliability_records["r_img_label"].append(r_img_label.detach().cpu().flatten())
        reliability_records["r_num_label"].append(r_num_label.detach().cpu().flatten())

        total_count += batch_size

    result = {}

    for key, value in sums.items():
        averaged = value / max(1, total_count)

        if "mse" in key:
            result[key] = averaged.tolist()
            result[key.replace("mse", "rmse")] = averaged.sqrt().tolist()
        else:
            result[key] = averaged.tolist()

    def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.float()
        y = y.float()
        x = x - x.mean()
        y = y - y.mean()
        denominator = (x.square().sum().sqrt() * y.square().sum().sqrt()).clamp_min(1e-12)
        return float((x * y).sum() / denominator)

    records = {key: torch.cat(values) for key, values in reliability_records.items()}
    result["reliability"] = {
        "image_reliability_vs_negative_error_corr": _correlation(
            records["r_img"],
            -records["img_error"],
        ),
        "num_reliability_vs_negative_error_corr": _correlation(
            records["r_num"],
            -records["num_error"],
        ),
        "image_reliability_label_mae": float(
            (records["r_img"] - records["r_img_label"]).abs().mean()
        ),
        "num_reliability_label_mae": float(
            (records["r_num"] - records["r_num_label"]).abs().mean()
        ),
    }

    return result


@torch.no_grad()
def test_gemininet(
    args,
    loaders: Dict,
    device,
    image_ckpt: str | Path,
    num_ckpt: str | Path,
    gemini_ckpt: str | Path,
    normalizers=None,
) -> dict:
    """Evaluate GeminiNet on the final test set and save full JSON metrics."""
    model = build_gemininet(
        args=args,
        image_ckpt=image_ckpt,
        num_ckpt=num_ckpt,
        device=device,
        normalizers=normalizers,
    )

    model = load_model_weights(
        model,
        gemini_ckpt,
        device,
        args=args,
        strict=False,
        allowed_missing_prefixes=("tendency_generator.",),
        normalizers=normalizers,
    )

    test_metrics = run_gemininet_epoch(
        model=model,
        loader=loaders["test"],
        device=device,
        optimizer=None,
        args=args,
        normalizers=normalizers,
    )
    test_metrics["detailed"] = evaluate_forecasts(
        model=model,
        loader=loaders["test"],
        device=device,
        normalizers=normalizers,
        args=args,
    )

    print("[GeminiNet Test]", test_metrics)

    save_json(
        path=Path(args.exp_dir) / "logs" / "gemininet_test_metrics.json",
        data=test_metrics,
    )

    if args.visualize:
        from utils.visualization import visualize_multimodal_batch

        visualize_multimodal_batch(
            model=model,
            loader=loaders["test"],
            device=device,
            save_path=Path(args.exp_dir) / "visualization" / "gemininet_test_prediction.png",
            sample_count=2,
            title="GeminiNet Test Prediction",
            random_samples=args.vis_random_samples,
            img_norm_type=args.img_norm_type,
            img_output_activation=args.img_output_activation,
        )

    return test_metrics