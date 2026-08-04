from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_provider import build_dataloaders, normalizers_from_dict
from trainer import (
    test_gemininet,
    train_gemininet,
    train_image_prior,
    train_num_prior,
)
from utils.common import ensure_dir, get_device, set_seed

import warnings

warnings.filterwarnings("ignore")
from utils.auto_plot import run_auto_plots


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("GeminiNet project runner")

    # ------------------------------
    # Path settings
    # ------------------------------
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"./datasets/Moving_MNIST_State",
        help="Folder containing train.npz, val.npz and test.npz.",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=r"./experiments/moving_mnist_state",
        help="Folder for logs, checkpoints and visualization results.",
    )

    # ------------------------------
    # Running mode
    # ------------------------------
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
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save prediction visualizations.",
    )

    # ------------------------------
    # Checkpoint paths
    # ------------------------------
    parser.add_argument(
        "--image_prior_ckpt",
        type=str,
        default="",
        help="Path to an image tendency model checkpoint.",
    )
    parser.add_argument(
        "--num_prior_ckpt",
        type=str,
        default="",
        help="Path to a numerical tendency model checkpoint.",
    )
    parser.add_argument(
        "--gemininet_ckpt",
        type=str,
        default="",
        help="Path to a GeminiNet checkpoint for test-only mode.",
    )
    parser.add_argument(
        "--resume_gemininet_ckpt",
        type=str,
        default=r"",
        help="Optional GeminiNet checkpoint used to resume training.",
    )

    # ------------------------------
    # Data shape
    # ------------------------------
    parser.add_argument(
        "--input_len_img",
        type=int,
        default=15,
        help="Length of historical image observation sequence.",
    )
    parser.add_argument(
        "--input_len_num",
        type=int,
        default=15,
        help="Length of historical numerical observation sequence.",
    )
    parser.add_argument(
        "--pred_len_img", type=int, default=15, help="Length of future image sequence to predict."
    )
    parser.add_argument(
        "--pred_len_num",
        type=int,
        default=15,
        help="Length of future numerical sequence to predict.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="Height and width of input/output image frames. Current Moving MNIST-State uses 64.",
    )
    parser.add_argument(
        "--img_channels",
        type=int,
        default=1,
        help="Number of image channels. Grayscale images use 1.",
    )
    parser.add_argument(
        "--num_vars", type=int, default=11, help="Number of numerical variables in each time step."
    )
    parser.add_argument(
        "--prior_fraction",
        type=float,
        default=0.125,
        help="Ordered fraction of the official training file reserved for the frozen unimodal prior models.",
    )
    parser.add_argument(
        "--train_split_gap",
        type=int,
        default=0,
        help="Optional number of ordered training samples excluded between prior and GeminiNet partitions. Use a purge gap for overlapping sliding-window datasets.",
    )

    parser.add_argument(
        "--data_usage_ratio",
        type=float,
        default=1.0,
        help="Fraction of prior_train, gemini_train and validation samples to use.",
    )
    parser.add_argument(
        "--test_usage_ratio",
        type=float,
        default=1.0,
        help="Fraction of test samples to use. Use 1.0 for final evaluation.",
    )

    # ------------------------------
    # Normalization and output range
    # ------------------------------
    parser.add_argument(
        "--img_norm_type",
        type=str,
        default="scale_01",
        choices=["none", "scale_01", "minmax", "zscore", "robust", "log1p_zscore"],
        help="Image normalization type.",
    )
    parser.add_argument(
        "--num_norm_type",
        type=str,
        default="zscore",
        choices=["none", "zscore", "robust", "minmax", "log1p_zscore"],
        help="Numerical normalization type.",
    )
    parser.add_argument(
        "--img_output_activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "tanh", "linear"],
        help="Output activation for image forecasting models.",
    )
    parser.add_argument(
        "--num_output_activation",
        type=str,
        default="linear",
        choices=["linear", "tanh"],
        help="Output activation for numerical forecasting models.",
    )
    parser.add_argument(
        "--output_scale",
        type=float,
        default=5.0,
        help="Scale used when an output activation is tanh.",
    )

    # ------------------------------
    # Unimodal tendency model settings
    # ------------------------------
    parser.add_argument(
        "--unet_base_channels",
        type=int,
        default=32,
        help="Base channel number of the U-Net image tendency model.",
    )
    parser.add_argument(
        "--prior_d_model",
        type=int,
        default=512,
        help="Hidden dimension of the Transformer numerical tendency model.",
    )
    parser.add_argument(
        "--prior_encoder_layers",
        type=int,
        default=3,
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
        default=512,
        help="Feed-forward hidden dimension in the numerical tendency Transformer.",
    )

    # ------------------------------
    # GeminiNet model settings
    # ------------------------------
    parser.add_argument(
        "--d_img",
        type=int,
        default=1024,
        help="Hidden representation dimension of the image branch in GeminiNet.",
    )
    parser.add_argument(
        "--d_num",
        type=int,
        default=512,
        help="Hidden representation dimension of the numerical branch in GeminiNet.",
    )
    parser.add_argument(
        "--d_shared",
        type=int,
        default=1024,
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

    # ------------------------------
    # Training settings
    # ------------------------------
    parser.add_argument(
        "--epochs_prior",
        type=int,
        default=600,
        help="Number of training epochs for the two unimodal tendency models.",
    )
    parser.add_argument(
        "--epochs_gemini", type=int, default=600, help="Number of training epochs for GeminiNet."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
        default=42,
        help="Random seed. Set to -1 to disable fixed seed.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic PyTorch algorithms when possible.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop after this many unimproved GeminiNet epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=10,
        help="Validation epochs before reducing the GeminiNet learning rate.",
    )
    parser.add_argument(
        "--lr_factor", type=float, default=0.5, help="Learning-rate reduction factor."
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-7, help="Minimum GeminiNet learning rate."
    )

    # ------------------------------
    # Loss settings
    # ------------------------------
    parser.add_argument(
        "--lambda_img", type=float, default=1.0, help="Loss weight for image forecasting loss."
    )
    parser.add_argument(
        "--lambda_num", type=float, default=1.0, help="Loss weight for numerical forecasting loss."
    )
    parser.add_argument(
        "--lambda_rel",
        type=float,
        default=0.02,
        help="Loss weight for tendency reliability supervision loss.",
    )
    parser.add_argument(
        "--tau_img",
        type=float,
        default=0.1,
        help="Temperature coefficient for converting image tendency error into reliability label.",
    )
    parser.add_argument(
        "--tau_num",
        type=float,
        default=1.0,
        help="Temperature coefficient for converting numerical tendency error into reliability label.",
    )
    parser.add_argument(
        "--use_reliability_estimator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use the calibrated tendency reliability estimator. The default beta5 setting keeps it enabled.",
    )
    parser.add_argument(
        "--lambda_img_mae",
        type=float,
        default=0.05,
        help="Auxiliary image MAE weight added to image MSE.",
    )
    parser.add_argument(
        "--image_loss_type",
        type=str,
        default="mse",
        choices=["mae", "mse", "huber"],
        help="Primary image forecasting loss type.",
    )
    parser.add_argument(
        "--image_huber_delta", type=float, default=1.0, help="Delta value used by Huber image loss."
    )
    parser.add_argument(
        "--num_loss_type",
        type=str,
        default="huber",
        choices=["mae", "mse", "huber"],
        help="Numerical forecasting loss type.",
    )
    parser.add_argument(
        "--huber_delta", type=float, default=1.0, help="Delta value used by Huber numerical loss."
    )
    parser.add_argument(
        "--reliability_label_mode",
        type=str,
        default="mixed",
        choices=["absolute", "relative", "mixed"],
        help="How to build reliability supervision labels from tendency errors.",
    )
    parser.add_argument(
        "--reliability_label_mix",
        type=float,
        default=0.5,
        help="Mix ratio of absolute and relative reliability labels when mode=mixed.",
    )
    parser.add_argument(
        "--reliability_label_floor",
        type=float,
        default=0.05,
        help="Lower/upper clipping floor for reliability labels.",
    )
    parser.add_argument(
        "--rel_loss_type",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "mse", "bce"],
        help="Reliability supervision loss type.",
    )
    parser.add_argument(
        "--lambda_prior_consistency",
        type=float,
        default=0.01,
        help="Consistency weight that anchors high-reliability predictions to prior tendencies.",
    )
    parser.add_argument(
        "--lr_reliability_scale",
        type=float,
        default=0.5,
        help="Learning-rate multiplier for the reliability estimator parameters.",
    )
    # ------------------------------
    # Visualization settings
    # ------------------------------
    parser.add_argument(
        "--vis_interval",
        type=int,
        default=2,
        help="Visualization interval during GeminiNet training. For example, 1 means visualize every epoch.",
    )
    parser.add_argument(
        "--vis_random_samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to randomly select visualization samples.",
    )
    parser.add_argument(
        "--auto_plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to automatically plot metrics after training/evaluation.",
    )
    args = parser.parse_args(args=argv)

    return args


def check_mode(args) -> None:
    """Check whether the current running mode is valid."""
    if args.output_scale <= 0:
        raise ValueError("output_scale must be greater than 0.")

    if not 0.0 < args.prior_fraction < 1.0:
        raise ValueError("prior_fraction must be between 0 and 1.")
    if args.train_split_gap < 0:
        raise ValueError("train_split_gap must be non-negative.")

    if not 0.0 < args.data_usage_ratio <= 1.0:
        raise ValueError("data_usage_ratio must be in the interval (0, 1].")
    if not 0.0 < args.test_usage_ratio <= 1.0:
        raise ValueError("test_usage_ratio must be in the interval (0, 1].")

    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError("lr_factor must be between 0 and 1.")

    if args.lr_patience < 0 or args.early_stopping_patience < 0:
        raise ValueError("patience values must be non-negative.")

    if args.epochs_prior <= 0 or args.epochs_gemini <= 0:
        raise ValueError("Training epoch counts must be positive.")

    dimensions = {
        "prior_d_model": args.prior_d_model,
        "d_img": args.d_img,
        "d_num": args.d_num,
        "d_shared": args.d_shared,
    }
    invalid_dimensions = [name for name, value in dimensions.items() if value % args.nhead != 0]
    if invalid_dimensions:
        raise ValueError(
            "Attention dimensions must be divisible by nhead: " + ", ".join(invalid_dimensions)
        )

    sequence_lengths = [
        args.input_len_img,
        args.input_len_num,
        args.pred_len_img,
        args.pred_len_num,
    ]
    if any(length <= 0 for length in sequence_lengths):
        raise ValueError("All input and prediction lengths must be positive.")

    if args.resume_gemininet_ckpt and not args.train_gemininet:
        raise ValueError("resume_gemininet_ckpt requires train_gemininet=True.")

    if args.test_only and (args.train_image_prior or args.train_num_prior or args.train_gemininet):
        raise ValueError(
            "test_only=True cannot be combined with any training stage. "
            "Disable all three training flags."
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


def _print_all_args(args) -> None:
    """Print parsed arguments."""
    print("=" * 80)
    print("All Arguments")
    print("=" * 80)
    for key in sorted(vars(args)):
        print(f"{key}: {getattr(args, key)}")
    print("=" * 80)


def _red(text: str) -> str:
    """Return red terminal text when ANSI colors are supported."""
    return f"\033[91m{text}\033[0m"


def _print_runtime_args(args) -> None:
    """Highlight selected runtime settings before training starts."""
    sensitive_keys = [
        "epochs_prior",
        "epochs_gemini",
        "data_usage_ratio",
        "test_usage_ratio",
        "train_image_prior",
        "train_num_prior",
        "train_gemininet",
        "test_only",
        "visualize",
        "vis_interval",
        "batch_size",
        "num_workers",
        "img_norm_type",
        "num_norm_type",
        "img_output_activation",
        "num_output_activation",
    ]

    print(_red("=" * 80))
    print(_red("Runtime Arguments Check"))
    print(_red("=" * 80))
    for key in sensitive_keys:
        print(_red(f"{key}: {getattr(args, key)}"))
    print(_red("=" * 80))


def _print_dataset_summary(loaders) -> None:
    """Print split sizes, batch counts and one normalized batch shape."""
    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    for split in ["prior_train", "gemini_train", "val", "test"]:
        loader = loaders[split]
        print(f"{split}: samples={len(loader.dataset)}, " f"batches={len(loader)}")

    print("-" * 80)
    first_batch = next(iter(loaders["prior_train"]))
    print("One prior_train batch shape:")
    for key in ["x_img", "y_img", "x_num", "y_num"]:
        print(f"{key}: {tuple(first_batch[key].shape)}")
    print("=" * 80)


def main() -> None:
    args = parse_args()
    check_mode(args)

    if args.seed >= 0:
        set_seed(args.seed, deterministic=args.deterministic)

    device = get_device(args.device)

    exp_dir = ensure_dir(args.exp_dir)
    ensure_dir(exp_dir / "logs")
    ensure_dir(exp_dir / "checkpoints")
    ensure_dir(exp_dir / "visualization")

    # Keep the training configuration immutable during later test-only runs.
    args_file_name = "test_args.json" if args.test_only else "args.json"
    with open(exp_dir / args_file_name, "w", encoding="utf-8") as file:
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
    print("=" * 80)
    _print_all_args(args)
    _print_runtime_args(args)

    normalizer_path = exp_dir / "normalizer.json"
    restored_normalizers = None

    if args.test_only:
        if not normalizer_path.exists():
            raise FileNotFoundError(
                "test_only requires the normalizer fitted during training: " f"{normalizer_path}"
            )

        with open(normalizer_path, "r", encoding="utf-8") as file:
            restored_normalizers = normalizers_from_dict(json.load(file))

        expected_norm_types = {
            "image": args.img_norm_type,
            "numerical": args.num_norm_type,
        }
        for name, expected in expected_norm_types.items():
            actual = restored_normalizers[name].norm_type
            if actual != expected:
                raise ValueError(
                    f"{name} normalizer mismatch: saved={actual}, " f"current={expected}."
                )

    loaders, normalizers = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_norm_type=args.img_norm_type,
        num_norm_type=args.num_norm_type,
        prior_fraction=args.prior_fraction,
        train_split_gap=args.train_split_gap,
        data_usage_ratio=args.data_usage_ratio,
        test_usage_ratio=args.test_usage_ratio,
        input_len_img=args.input_len_img,
        input_len_num=args.input_len_num,
        pred_len_img=args.pred_len_img,
        pred_len_num=args.pred_len_num,
        img_channels=args.img_channels,
        num_vars=args.num_vars,
        normalizers=restored_normalizers,
    )
    _print_dataset_summary(loaders)

    if not args.test_only:
        with open(normalizer_path, "w", encoding="utf-8") as file:
            json.dump(
                {name: normalizer.to_dict() for name, normalizer in normalizers.items()},
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
                normalizers=normalizers,
            )
        )

    if args.train_num_prior:
        num_ckpt = str(
            train_num_prior(
                args=args,
                loaders=loaders,
                device=device,
                normalizers=normalizers,
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
            normalizers=normalizers,
        )

    if args.test_only:
        test_gemininet(
            args=args,
            loaders=loaders,
            device=device,
            image_ckpt=args.image_prior_ckpt,
            num_ckpt=args.num_prior_ckpt,
            gemini_ckpt=args.gemininet_ckpt,
            normalizers=normalizers,
        )
    run_auto_plots(args)


if __name__ == "__main__":
    main()