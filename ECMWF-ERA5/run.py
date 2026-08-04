from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Windows + Conda environments may load Intel OpenMP through multiple scientific packages.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from data_provider import build_dataloaders
from trainer import (
    test_gemininet,
    train_gemininet,
    train_image_prior,
    train_num_prior,
)
from utils.auto_plot import run_auto_plots
from utils.common import ensure_dir, get_device, set_seed

import warnings

warnings.filterwarnings("ignore")


RED = "\033[31m"
RESET = "\033[0m"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("GeminiNet project runner")

    # Path settings
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"./datasets/ERA5_Land_ChengYu/processed",
        help="Folder containing processed ERA5-Land arrays and metadata.",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=r"./experiments/era5_land_chengyu",
        help="Folder for logs, checkpoints, and visualizations.",
    )

    # Running mode
    parser.add_argument(
        "--train_image_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to train image tendency model.",
    )
    parser.add_argument(
        "--train_num_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to train numerical tendency model.",
    )
    parser.add_argument(
        "--train_gemininet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to train GeminiNet.",
    )
    parser.add_argument(
        "--test_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to only test GeminiNet.",
    )
    parser.add_argument(
        "--visualize_train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save GeminiNet training-set prediction visualizations.",
    )
    parser.add_argument(
        "--visualize_vali",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save GeminiNet validation-set prediction visualizations.",
    )

    # Checkpoint paths
    parser.add_argument(
        "--image_prior_ckpt",
        type=str,
        default="",
        help="Path to existing image tendency model checkpoint.",
    )
    parser.add_argument(
        "--num_prior_ckpt",
        type=str,
        default="",
        help="Path to existing numerical tendency model checkpoint.",
    )
    parser.add_argument(
        "--gemininet_ckpt",
        type=str,
        default="",
        help="Path to existing GeminiNet checkpoint for test-only mode.",
    )

    # Data shape and usage
    parser.add_argument(
        "--input_len_img",
        type=int,
        default=12,
        help="Length of historical ERA5-Land temperature-field observation sequence.",
    )
    parser.add_argument(
        "--input_len_num",
        type=int,
        default=12,
        help="Length of historical ERA5-Land regional-mean state sequence.",
    )
    parser.add_argument(
        "--pred_len_img",
        type=int,
        default=12,
        help="Length of future ERA5-Land temperature-field sequence to predict.",
    )
    parser.add_argument(
        "--pred_len_num",
        type=int,
        default=12,
        help="Length of future ERA5-Land regional-mean state sequence to predict.",
    )
    parser.add_argument(
        "--img_height", type=int, default=101, help="Height of native ERA5-Land ChengYu grid."
    )
    parser.add_argument(
        "--img_width", type=int, default=101, help="Width of native ERA5-Land ChengYu grid."
    )
    parser.add_argument(
        "--img_channels",
        type=int,
        default=1,
        help="Number of ERA5-Land temperature-field channels.",
    )
    parser.add_argument(
        "--num_vars",
        type=int,
        default=9,
        help="ERA5-Land numerical variables: regional-mean meteorological and land-surface states.",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=1,
        help="Sliding-window stride over the hourly ERA5-Land time series.",
    )

    parser.add_argument(
        "--data_usage_ratio",
        type=float,
        default=1.0,
        help="Fraction of processed ERA5-Land time steps to use.",
    )
    parser.add_argument(
        "--test_usage_ratio",
        type=float,
        default=1.0,
        help="Fraction of ERA5-Land test windows to use. Use 1.0 for final evaluation.",
    )

    # Normalization and output range settings
    parser.add_argument(
        "--img_norm_type",
        type=str,
        default="zscore",
        choices=["none", "scale_01", "minmax", "zscore", "robust", "log1p_zscore"],
        help="Image normalization type. zscore is recommended for ERA5-Land physical fields.",
    )
    parser.add_argument(
        "--num_norm_type",
        type=str,
        default="zscore",
        choices=["none", "minmax", "zscore", "robust", "log1p_zscore"],
        help="Numerical normalization type. zscore is recommended for most multivariate time-series variables.",
    )
    parser.add_argument(
        "--img_value_scale",
        type=float,
        default=1.0,
        help="Scale value used by img_norm_type='scale_01'. Not used for zscore ERA5-Land fields.",
    )
    parser.add_argument(
        "--img_output_activation",
        type=str,
        default="linear",
        choices=["sigmoid", "tanh", "linear"],
        help="Image output activation. Use linear for z-score-normalized physical fields.",
    )
    parser.add_argument(
        "--num_output_activation",
        type=str,
        default="linear",
        choices=["linear", "tanh"],
        help="Numerical output activation. linear is recommended for normalized numerical variables.",
    )
    parser.add_argument(
        "--output_scale",
        type=float,
        default=5.0,
        help="Scale used when an output activation is tanh.",
    )

    # Unimodal tendency model settings
    parser.add_argument(
        "--unet_base_channels",
        type=int,
        default=32,
        help="Base channel number of the U-Net image tendency model.",
    )
    parser.add_argument(
        "--prior_d_model",
        type=int,
        default=128,
        help="Hidden dimension of the Transformer numerical tendency model.",
    )
    parser.add_argument(
        "--prior_encoder_layers",
        type=int,
        default=2,
        help="Number of Transformer encoder layers in the numerical tendency model.",
    )
    parser.add_argument(
        "--prior_decoder_layers",
        type=int,
        default=2,
        help="Number of Transformer decoder layers in the numerical tendency model.",
    )
    parser.add_argument(
        "--prior_dim_feedforward",
        type=int,
        default=256,
        help="Feed-forward hidden dimension in the numerical tendency Transformer.",
    )

    # GeminiNet model settings
    parser.add_argument(
        "--d_img",
        type=int,
        default=256,
        help="Hidden representation dimension of the image branch in GeminiNet.",
    )
    parser.add_argument(
        "--d_num",
        type=int,
        default=128,
        help="Hidden representation dimension of the numerical branch in GeminiNet.",
    )
    parser.add_argument(
        "--d_shared",
        type=int,
        default=256,
        help="Shared hidden dimension used in cross-modal interaction.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Number of attention heads in Transformer and cross-attention modules.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate used in Transformer, MLP and related modules.",
    )

    # Training settings
    parser.add_argument(
        "--epochs_prior",
        type=int,
        default=200,
        help="Number of training epochs for the two unimodal tendency models.",
    )
    parser.add_argument(
        "--epochs_gemini", type=int, default=200, help="Number of training epochs for GeminiNet."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for prior training, GeminiNet training, validation and testing.",
    )
    parser.add_argument(
        "--lr_prior",
        type=float,
        default=1e-4,
        help="Learning rate for image and numerical tendency model training.",
    )
    parser.add_argument(
        "--lr_gemini", type=float, default=1e-4, help="Learning rate for GeminiNet training."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay coefficient used by AdamW optimizer.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Set to 0 or a negative value to disable clipping.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes. Use 0 on Windows.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device. Use auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed. Set to -1 to disable fixed seed.",
    )

    # Loss settings
    parser.add_argument(
        "--lambda_img", type=float, default=1.0, help="Loss weight for image forecasting loss."
    )
    parser.add_argument(
        "--lambda_num", type=float, default=1.0, help="Loss weight for numerical forecasting loss."
    )
    parser.add_argument(
        "--lambda_rel",
        type=float,
        default=0.1,
        help="Loss weight for tendency reliability supervision loss.",
    )
    parser.add_argument(
        "--lambda_img_mae",
        type=float,
        default=0.05,
        help="Auxiliary image MAE loss weight. Set to 0.0 to recover beta2 image loss.",
    )
    parser.add_argument(
        "--num_loss_type",
        type=str,
        default="huber",
        choices=["mae", "mse", "huber", "smooth_l1"],
        help="Numerical task loss used for training GeminiNet. Metrics still log both MSE and MAE.",
    )
    parser.add_argument(
        "--num_huber_delta",
        type=float,
        default=1.0,
        help="Beta/delta value for Huber numerical loss when num_loss_type='huber'.",
    )
    parser.add_argument(
        "--tau_img",
        type=float,
        default=0.05,
        help="Temperature coefficient for converting image tendency error into reliability label.",
    )
    parser.add_argument(
        "--tau_num",
        type=float,
        default=1.0,
        help="Temperature coefficient for converting numerical tendency error into reliability label.",
    )
    parser.add_argument(
        "--reliability_label_mode",
        type=str,
        default="mixed",
        choices=["absolute", "relative", "mixed"],
        help="Reliability-label construction mode. mixed is more robust across normalization types.",
    )
    parser.add_argument(
        "--use_reliability_estimator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use TendencyReliabilityEstimator.",
    )

    # Visualization settings
    parser.add_argument(
        "--vis_interval",
        type=int,
        default=1,
        help="Visualization interval during GeminiNet training. For example, 1 means visualize every epoch.",
    )
    parser.add_argument(
        "--vis_random_samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to randomly select visualization samples.",
    )
    parser.add_argument(
        "--vis_inverse_transform",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to inverse-transform data for visualization.",
    )
    parser.add_argument(
        "--metric_inverse_transform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to compute additional original-scale metrics with inverse_transform.",
    )
    parser.add_argument(
        "--auto_plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to automatically plot metrics after training/evaluation.",
    )

    args = parser.parse_args()
    return args


def print_key_args(args) -> None:
    """Print key switches before training starts."""
    print("=" * 80)
    print("Key Running Arguments")
    print("=" * 80)
    print(f"train_image_prior: {args.train_image_prior}")
    print(f"train_num_prior: {args.train_num_prior}")
    print(f"train_gemininet: {args.train_gemininet}")
    print(f"visualize_train: {args.visualize_train}")
    print(f"visualize_vali: {args.visualize_vali}")
    print(f"epochs_prior: {args.epochs_prior}")
    print(f"epochs_gemini: {args.epochs_gemini}")
    print(f"batch_size: {args.batch_size}")
    print(f"window_stride: {args.window_stride}")
    print(f"vis_interval: {args.vis_interval}")
    print(f"vis_inverse_transform: {args.vis_inverse_transform}")
    print(f"metric_inverse_transform: {args.metric_inverse_transform}")
    print("=" * 80)


def print_runtime_args(args) -> None:
    """Print selected runtime parameters in red."""
    lines = [
        "=" * 80,
        "Runtime Arguments Check",
        "=" * 80,
        f"epochs_prior: {args.epochs_prior}",
        f"epochs_gemini: {args.epochs_gemini}",
        f"data_usage_ratio: {args.data_usage_ratio}",
        f"window_stride: {args.window_stride}",
        f"num_vars: {args.num_vars}",
        f"train_image_prior: {args.train_image_prior}",
        f"train_num_prior: {args.train_num_prior}",
        f"visualize_train: {args.visualize_train}",
        f"visualize_vali: {args.visualize_vali}",
        f"vis_interval: {args.vis_interval}",
        f"batch_size: {args.batch_size}",
        "=" * 80,
    ]

    print(f"{RED}" + "\n".join(lines) + f"{RESET}")


def print_dataset_summary(loaders) -> None:
    """Print sample counts and tensor shapes before training starts."""
    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)

    for split in ["prior_train", "gemini_train", "val", "test"]:
        loader = loaders[split]
        print(f"{split}: samples={len(loader.dataset)}, batches={len(loader)}")

    batch = next(iter(loaders["prior_train"]))

    print("-" * 80)
    print("One prior_train batch shape:")
    print(f"x_img: {tuple(batch['x_img'].shape)}")
    print(f"y_img: {tuple(batch['y_img'].shape)}")
    print(f"x_num: {tuple(batch['x_num'].shape)}")
    print(f"y_num: {tuple(batch['y_num'].shape)}")
    print("=" * 80)


def validate_dataset_shapes(args, loaders) -> None:
    """Fail early when processed arrays do not match run.py settings."""
    batch = next(iter(loaders["prior_train"]))

    actual_input_len_img = batch["x_img"].shape[1]
    actual_pred_len_img = batch["y_img"].shape[1]
    actual_input_len_num = batch["x_num"].shape[1]
    actual_pred_len_num = batch["y_num"].shape[1]
    actual_img_shape = batch["x_img"].shape[2:4]
    actual_img_channels = batch["x_img"].shape[4]

    expected = (
        args.input_len_img,
        args.pred_len_img,
        args.input_len_num,
        args.pred_len_num,
    )
    actual = (
        actual_input_len_img,
        actual_pred_len_img,
        actual_input_len_num,
        actual_pred_len_num,
    )

    if actual != expected:
        raise ValueError(
            "Processed data sequence lengths do not match run.py settings. "
            f"Expected (input_img, pred_img, input_num, pred_num)={expected}, "
            f"but got {actual}. Re-run preprocessing, for example: "
            "python scripts/preprocess_era5_land_chengyu.py --force"
        )

    expected_img_shape = (args.img_height, args.img_width)
    if actual_img_shape != expected_img_shape or actual_img_channels != args.img_channels:
        raise ValueError(
            "Processed image shape does not match run.py settings. "
            f"Expected image shape (H, W, C)=({args.img_height}, {args.img_width}, {args.img_channels}), "
            f"but got ({actual_img_shape[0]}, {actual_img_shape[1]}, {actual_img_channels}). "
            "Re-run preprocessing, for example: "
            "python scripts/preprocess_era5_land_chengyu.py --force"
        )


def check_mode(args) -> None:
    """Check whether the current running mode is valid."""
    if args.test_only and args.train_gemininet:
        raise ValueError(
            "Invalid mode: test_only=True and train_gemininet=True " "cannot be used together."
        )

    if args.test_only:
        if not args.gemininet_ckpt:
            raise ValueError("test_only=True requires gemininet_ckpt.")

        if not args.image_prior_ckpt:
            raise ValueError("test_only=True requires image_prior_ckpt.")

        if not args.num_prior_ckpt:
            raise ValueError("test_only=True requires num_prior_ckpt.")

    if args.train_gemininet:
        default_image_ckpt = Path(args.exp_dir) / "checkpoints" / "image_prior_best.pth"

        default_num_ckpt = Path(args.exp_dir) / "checkpoints" / "num_prior_best.pth"

        if (
            not args.train_image_prior
            and not args.image_prior_ckpt
            and not default_image_ckpt.exists()
        ):
            raise ValueError(
                "GeminiNet training requires an image tendency model. "
                "Set train_image_prior=True or provide image_prior_ckpt."
            )

        if not args.train_num_prior and not args.num_prior_ckpt and not default_num_ckpt.exists():
            raise ValueError(
                "GeminiNet training requires a numerical tendency model. "
                "Set train_num_prior=True or provide num_prior_ckpt."
            )


def main() -> None:
    args = parse_args()
    check_mode(args)

    if args.seed >= 0:
        set_seed(args.seed)

    device = get_device(args.device)

    exp_dir = ensure_dir(args.exp_dir)
    ensure_dir(exp_dir / "logs")
    ensure_dir(exp_dir / "checkpoints")
    ensure_dir(exp_dir / "visualization")

    # Save current configuration for later checking.
    with open(exp_dir / "args.json", "w", encoding="utf-8") as file:
        json.dump(
            vars(args),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 80)
    print("GeminiNet Running Configuration")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Experiment directory: {args.exp_dir}")
    print(f"Train image prior: {args.train_image_prior}")
    print(f"Train numerical prior: {args.train_num_prior}")
    print(f"Train GeminiNet: {args.train_gemininet}")
    print(f"Test only: {args.test_only}")
    print(f"Visualize train: {args.visualize_train}")
    print(f"Visualize validation: {args.visualize_vali}")
    print(f"Visualization inverse transform: {args.vis_inverse_transform}")
    print(f"Metric inverse transform: {args.metric_inverse_transform}")
    print(f"Image normalization: {args.img_norm_type}")
    print(f"Numerical normalization: {args.num_norm_type}")
    print(f"Image output activation: {args.img_output_activation}")
    print(f"Numerical output activation: {args.num_output_activation}")
    print("=" * 80)
    print_key_args(args)
    print_runtime_args(args)

    loaders, normalizer = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_norm_type=args.img_norm_type,
        num_norm_type=args.num_norm_type,
        img_value_scale=args.img_value_scale,
        data_usage_ratio=args.data_usage_ratio,
        test_usage_ratio=args.test_usage_ratio,
        num_vars=args.num_vars,
        input_len_img=args.input_len_img,
        input_len_num=args.input_len_num,
        pred_len_img=args.pred_len_img,
        pred_len_num=args.pred_len_num,
        img_height=args.img_height,
        img_width=args.img_width,
        img_channels=args.img_channels,
        window_stride=args.window_stride,
    )
    print_dataset_summary(loaders)
    validate_dataset_shapes(args, loaders)

    with open(exp_dir / "normalizer.json", "w", encoding="utf-8") as file:
        json.dump(
            normalizer.to_dict(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    image_ckpt = args.image_prior_ckpt
    num_ckpt = args.num_prior_ckpt

    if args.train_image_prior:
        image_ckpt = str(
            train_image_prior(
                args=args,
                loaders=loaders,
                device=device,
                normalizer=normalizer,
            )
        )

    if args.train_num_prior:
        num_ckpt = str(
            train_num_prior(
                args=args,
                loaders=loaders,
                device=device,
                normalizer=normalizer,
            )
        )

    if args.train_gemininet:
        if not image_ckpt:
            image_ckpt = str(exp_dir / "checkpoints" / "image_prior_best.pth")

        if not num_ckpt:
            num_ckpt = str(exp_dir / "checkpoints" / "num_prior_best.pth")

        train_gemininet(
            args=args,
            loaders=loaders,
            device=device,
            image_ckpt=image_ckpt,
            num_ckpt=num_ckpt,
            normalizer=normalizer,
        )

    if args.test_only:
        test_gemininet(
            args=args,
            loaders=loaders,
            device=device,
            image_ckpt=args.image_prior_ckpt,
            num_ckpt=args.num_prior_ckpt,
            gemini_ckpt=args.gemininet_ckpt,
            normalizer=normalizer,
        )

    run_auto_plots(args)


if __name__ == "__main__":
    main()