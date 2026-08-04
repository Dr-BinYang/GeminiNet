from __future__ import annotations

import argparse
import json
import ssl
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

import numpy as np

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate Moving MNIST-State windows")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR / "processed"),
        help="Directory where train.npz, val.npz, test.npz and metadata.json will be written.",
    )
    parser.add_argument(
        "--mnist_root",
        type=str,
        default=str(SCRIPT_DIR / "raw"),
        help="Directory containing or receiving torchvision MNIST files.",
    )
    parser.add_argument(
        "--train_size", type=int, default=10000, help="Number of generated training samples."
    )
    parser.add_argument(
        "--val_size", type=int, default=2000, help="Number of generated validation samples."
    )
    parser.add_argument(
        "--test_size", type=int, default=2000, help="Number of generated test samples."
    )
    parser.add_argument("--input_len", type=int, default=15, help="Historical sequence length.")
    parser.add_argument("--pred_len", type=int, default=15, help="Forecast sequence length.")
    parser.add_argument("--canvas_size", type=int, default=64, help="Square canvas size.")
    parser.add_argument(
        "--digit_size", type=int, default=28, help="MNIST digit size. Standard MNIST uses 28."
    )
    parser.add_argument(
        "--min_speed", type=float, default=1.0, help="Minimum digit speed in pixels per frame."
    )
    parser.add_argument(
        "--max_speed", type=float, default=3.0, help="Maximum digit speed in pixels per frame."
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--download_mnist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow torchvision to download MNIST if it is missing.",
    )
    parser.add_argument(
        "--insecure_mnist_download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use an unverified HTTPS context for MNIST download when Windows/Python certificate loading fails. Torchvision still checks MNIST file md5 hashes.",
    )
    parser.add_argument(
        "--compressed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use np.savez_compressed. Disable for faster but larger files.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing split files in output_dir.",
    )
    return parser.parse_args()


def load_mnist_arrays(
    mnist_root: str | Path, download: bool, insecure_download: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from torchvision.datasets import MNIST
    except ImportError as error:
        raise ImportError(
            "This script needs torchvision. Install it or provide a project environment that already has it."
        ) from error

    if download and insecure_download:
        ssl._create_default_https_context = ssl._create_unverified_context

    train = MNIST(root=str(mnist_root), train=True, download=download)
    test = MNIST(root=str(mnist_root), train=False, download=download)
    train_images = train.data.numpy().astype(np.uint8)
    train_labels = train.targets.numpy().astype(np.int64)
    test_images = test.data.numpy().astype(np.uint8)
    test_labels = test.targets.numpy().astype(np.int64)
    return train_images, train_labels, test_images, test_labels


def sample_velocity(rng: np.random.Generator, min_speed: float, max_speed: float) -> np.ndarray:
    speed = rng.uniform(min_speed, max_speed)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([np.cos(angle) * speed, np.sin(angle) * speed], dtype=np.float32)


def render_digit(canvas: np.ndarray, digit: np.ndarray, position: np.ndarray) -> np.ndarray:
    x = int(round(float(position[0])))
    y = int(round(float(position[1])))
    local_mask = digit > 0
    patch = canvas[y : y + digit.shape[0], x : x + digit.shape[1]]
    np.maximum(patch, digit, out=patch)
    mask = np.zeros_like(canvas, dtype=bool)
    mask[y : y + digit.shape[0], x : x + digit.shape[1]] = local_mask
    return mask


def advance_position(position: np.ndarray, velocity: np.ndarray, max_position: float) -> None:
    position += velocity
    for axis in range(2):
        if position[axis] < 0.0:
            position[axis] = -position[axis]
            velocity[axis] = abs(velocity[axis])
        elif position[axis] > max_position:
            position[axis] = 2.0 * max_position - position[axis]
            velocity[axis] = -abs(velocity[axis])


def generate_one_sample(
    rng: np.random.Generator,
    images: np.ndarray,
    labels: np.ndarray,
    total_len: int,
    canvas_size: int,
    digit_size: int,
    min_speed: float,
    max_speed: float,
) -> dict:
    max_position = float(canvas_size - digit_size)
    indices = rng.integers(0, len(images), size=2)
    digits = images[indices]
    digit_labels = labels[indices]
    positions = rng.uniform(0.0, max_position, size=(2, 2)).astype(np.float32)
    velocities = np.stack(
        [sample_velocity(rng, min_speed, max_speed), sample_velocity(rng, min_speed, max_speed)],
        axis=0,
    )
    initial_positions = positions.copy()
    initial_velocities = velocities.copy()
    full_img = np.zeros((total_len, 1, canvas_size, canvas_size), dtype=np.uint8)
    full_num = np.zeros((total_len, len(STATE_NAMES)), dtype=np.float32)

    for time_index in range(total_len):
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        masks = []

        for digit_index in range(2):
            mask = render_digit(canvas, digits[digit_index], positions[digit_index])
            masks.append((mask, positions[digit_index].copy()))

        full_img[time_index, 0] = canvas
        mask_1, pos_1 = masks[0]
        mask_2, pos_2 = masks[1]
        x1, y1 = pos_1
        x2, y2 = pos_2
        center_1 = pos_1 + digit_size / 2.0
        center_2 = pos_2 + digit_size / 2.0
        distance = float(np.linalg.norm(center_1 - center_2))
        foreground_area = float(np.count_nonzero(canvas))
        overlap_area = float(np.count_nonzero(mask_1 & mask_2))
        full_num[time_index] = np.array(
            [
                x1,
                y1,
                x2,
                y2,
                velocities[0, 0],
                velocities[0, 1],
                velocities[1, 0],
                velocities[1, 1],
                distance,
                foreground_area,
                overlap_area,
            ],
            dtype=np.float32,
        )

        if time_index < total_len - 1:
            for digit_index in range(2):
                advance_position(positions[digit_index], velocities[digit_index], max_position)

    return {
        "full_img": full_img,
        "full_num": full_num,
        "digit_labels": digit_labels.astype(np.int64),
        "digit_indices": indices.astype(np.int64),
        "initial_positions": initial_positions,
        "initial_velocities": initial_velocities,
    }


def generate_split(
    split: str,
    count: int,
    images: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    seed_offset: int,
) -> dict:
    rng = np.random.default_rng(args.seed + seed_offset)
    total_len = args.input_len + args.pred_len
    full_img = np.zeros((count, total_len, 1, args.canvas_size, args.canvas_size), dtype=np.uint8)
    full_num = np.zeros((count, total_len, len(STATE_NAMES)), dtype=np.float32)
    digit_labels = np.zeros((count, 2), dtype=np.int64)
    digit_indices = np.zeros((count, 2), dtype=np.int64)
    initial_positions = np.zeros((count, 2, 2), dtype=np.float32)
    initial_velocities = np.zeros((count, 2, 2), dtype=np.float32)

    for sample_index in range(count):
        sample = generate_one_sample(
            rng,
            images,
            labels,
            total_len,
            args.canvas_size,
            args.digit_size,
            args.min_speed,
            args.max_speed,
        )
        full_img[sample_index] = sample["full_img"]
        full_num[sample_index] = sample["full_num"]
        digit_labels[sample_index] = sample["digit_labels"]
        digit_indices[sample_index] = sample["digit_indices"]
        initial_positions[sample_index] = sample["initial_positions"]
        initial_velocities[sample_index] = sample["initial_velocities"]

        if (sample_index + 1) % 500 == 0 or sample_index + 1 == count:
            print(f"[{split}] generated {sample_index + 1}/{count}")

    return {
        "x_img": full_img[:, : args.input_len],
        "x_num": full_num[:, : args.input_len],
        "y_img": full_img[:, args.input_len :],
        "y_num": full_num[:, args.input_len :],
        "full_img": full_img,
        "full_num": full_num,
        "digit_labels": digit_labels,
        "digit_indices": digit_indices,
        "initial_positions": initial_positions,
        "initial_velocities": initial_velocities,
        "state_names": np.asarray(STATE_NAMES),
    }


def save_split(path: Path, data: dict, compressed: bool) -> None:
    if compressed:
        np.savez_compressed(path, **data)
    else:
        np.savez(path, **data)
    print(f"Saved: {path}")


def write_metadata(path: Path, args: argparse.Namespace) -> None:
    metadata = {
        "dataset_name": "Moving MNIST-State",
        "description": "Paired image-state forecasting dataset generated from two moving MNIST digits.",
        "normalization": {
            "image": "No normalization in saved files. Images are uint8 raw pixels in [0, 255].",
            "numerical_state": "No normalization. Raw pixel coordinates, pixel/frame velocities, pixel distances, and pixel counts are saved.",
        },
        "state_names": STATE_NAMES,
        "shape": {
            "x_img": "[N, input_len, 1, canvas_size, canvas_size]",
            "x_num": "[N, input_len, 11]",
            "y_img": "[N, pred_len, 1, canvas_size, canvas_size]",
            "y_num": "[N, pred_len, 11]",
            "full_img": "[N, input_len + pred_len, 1, canvas_size, canvas_size]",
            "full_num": "[N, input_len + pred_len, 11]",
        },
        "config": vars(args),
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    if args.input_len <= 0 or args.pred_len <= 0:
        raise ValueError("input_len and pred_len must be positive.")
    if args.canvas_size < args.digit_size:
        raise ValueError("canvas_size must be greater than or equal to digit_size.")
    if args.min_speed <= 0 or args.max_speed < args.min_speed:
        raise ValueError("Require 0 < min_speed <= max_speed.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_files = [
        output_dir / "train.npz",
        output_dir / "val.npz",
        output_dir / "test.npz",
        output_dir / "metadata.json",
    ]
    existing = [path for path in target_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist. Use --overwrite after backing up old 10->10 data: "
            + ", ".join(str(path) for path in existing)
        )

    train_images, train_labels, test_images, test_labels = load_mnist_arrays(
        args.mnist_root, args.download_mnist, args.insecure_mnist_download
    )
    split_specs = [
        ("train", args.train_size, train_images, train_labels, 0),
        ("val", args.val_size, train_images, train_labels, 10_000),
        ("test", args.test_size, test_images, test_labels, 20_000),
    ]

    for split, count, images, labels, seed_offset in split_specs:
        data = generate_split(split, count, images, labels, args, seed_offset)
        save_split(output_dir / f"{split}.npz", data, compressed=args.compressed)

    write_metadata(output_dir / "metadata.json", args)
    print(
        "Done. Generated Moving MNIST-State data with "
        f"input_len={args.input_len}, pred_len={args.pred_len}."
    )


if __name__ == "__main__":
    main()
