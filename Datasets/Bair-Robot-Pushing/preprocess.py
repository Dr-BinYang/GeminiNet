from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

import numpy as np

NUM_FEATURE_NAMES = {
    "ee_only": [
        "ee_x",
        "ee_y",
        "ee_z",
    ],
    "ee_action": [
        "ee_x",
        "ee_y",
        "ee_z",
        "action_1",
        "action_2",
        "action_3",
        "action_4",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert BAIR Robot Pushing Small TFRecords into GeminiNet-ready "
            "memory-mapped .npy arrays."
        )
    )

    parser.add_argument(
        "--raw_root",
        type=str,
        default=str(
            SCRIPT_DIR
            / "raw"
            / "bair_robot_pushing_dataset_v0"
            / "softmotion30_44k"
        ),
        help="Folder containing BAIR train/ and test/ TFRecord files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR / "processed"),
        help="Folder where converted .npy arrays and metadata.json are saved.",
    )
    parser.add_argument("--input_len", type=int, default=15)
    parser.add_argument("--pred_len", type=int, default=15)
    parser.add_argument("--raw_sequence_len", type=int, default=30)
    parser.add_argument(
        "--camera",
        type=str,
        default="image_main",
        choices=["image_main", "image_aux1"],
        help="BAIR camera stream used as the image modality.",
    )
    parser.add_argument(
        "--num_mode",
        type=str,
        default="ee_action",
        choices=["ee_only", "ee_action"],
        help="Numerical modality construction.",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=0,
        help=(
            "0 means one sample per trajectory using frames [0:input+pred]. "
            "A positive value enables sliding windows over the 30-step sequence."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Optional processing cap. 0 means use all train samples.",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=0,
        help="Optional processing cap. 0 means use all test samples.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing converted arrays.",
    )

    return parser.parse_args()


def _tfrecord_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"traj_(\d+)_to_(\d+)\.tfrecords$", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def _list_tfrecords(raw_root: Path, split: str) -> list[Path]:
    split_dir = raw_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"BAIR split folder not found: {split_dir}")

    files = sorted(split_dir.glob("*.tfrecords"), key=_tfrecord_sort_key)
    if not files:
        raise FileNotFoundError(f"No .tfrecords files found in {split_dir}")

    return files


def _build_feature_spec(tf, raw_sequence_len: int, camera: str) -> dict:
    feature_spec = {}
    for step in range(raw_sequence_len):
        feature_spec[f"{step}/{camera}/encoded"] = tf.io.FixedLenFeature([], tf.string)
        feature_spec[f"{step}/endeffector_pos"] = tf.io.FixedLenFeature([3], tf.float32)
        feature_spec[f"{step}/action"] = tf.io.FixedLenFeature([4], tf.float32)
    return feature_spec


def _window_starts(
    raw_sequence_len: int,
    input_len: int,
    pred_len: int,
    window_stride: int,
) -> list[int]:
    total_len = input_len + pred_len
    if total_len > raw_sequence_len:
        raise ValueError(
            f"input_len + pred_len = {total_len} exceeds " f"raw_sequence_len={raw_sequence_len}."
        )

    if window_stride <= 0:
        return [0]

    return list(range(0, raw_sequence_len - total_len + 1, window_stride))


def _count_records(tf, files: list[Path]) -> int:
    total = 0
    for file in files:
        for _ in tf.data.TFRecordDataset(str(file)):
            total += 1
    return total


def _decode_image_bytes(tf, image_bytes: bytes) -> np.ndarray:
    """Decode one BAIR image.

    The BAIR TFRecord feature is named ".../encoded", but in this release the
    stored payload is raw 64x64x3 RGB bytes rather than JPEG/PNG bytes.
    A fallback image decoder is kept for compatibility with repacked variants.
    """
    raw_rgb_size = 64 * 64 * 3

    if len(image_bytes) == raw_rgb_size:
        return np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        ).reshape(64, 64, 3)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image = image.numpy()

    if image.shape != (64, 64, 3):
        raise ValueError(f"Expected decoded BAIR image shape (64, 64, 3), " f"got {image.shape}.")

    return image.astype(np.uint8)


def _decode_record(tf, raw_record, feature_spec: dict, raw_sequence_len: int, camera: str):
    parsed = tf.io.parse_single_example(raw_record, feature_spec)

    images = []
    ee_pos = []
    actions = []

    for step in range(raw_sequence_len):
        image_bytes = parsed[f"{step}/{camera}/encoded"].numpy()
        image = _decode_image_bytes(
            tf=tf,
            image_bytes=image_bytes,
        )
        images.append(image)
        ee_pos.append(parsed[f"{step}/endeffector_pos"].numpy())
        actions.append(parsed[f"{step}/action"].numpy())

    images = np.asarray(images, dtype=np.uint8)
    ee_pos = np.asarray(ee_pos, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)

    return images, ee_pos, actions


def _make_num_sequence(
    ee_pos: np.ndarray,
    actions: np.ndarray,
    num_mode: str,
) -> np.ndarray:
    if num_mode == "ee_only":
        return ee_pos.astype(np.float32)

    if num_mode == "ee_action":
        return np.concatenate([ee_pos, actions], axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported num_mode: {num_mode}")


def _prepare_output_paths(output_dir: Path, split: str, force: bool) -> dict[str, Path]:
    paths = {
        "x_img": output_dir / f"{split}_x_img.npy",
        "y_img": output_dir / f"{split}_y_img.npy",
        "x_num": output_dir / f"{split}_x_num.npy",
        "y_num": output_dir / f"{split}_y_num.npy",
    }

    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Converted arrays already exist. Use --force to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    return paths


def _write_split(
    tf,
    files: list[Path],
    output_dir: Path,
    split: str,
    input_len: int,
    pred_len: int,
    raw_sequence_len: int,
    camera: str,
    num_mode: str,
    window_stride: int,
    max_samples: int,
    force: bool,
) -> int:
    feature_spec = _build_feature_spec(
        tf=tf,
        raw_sequence_len=raw_sequence_len,
        camera=camera,
    )
    starts = _window_starts(
        raw_sequence_len=raw_sequence_len,
        input_len=input_len,
        pred_len=pred_len,
        window_stride=window_stride,
    )

    trajectory_count = _count_records(tf, files)
    sample_count = trajectory_count * len(starts)
    if max_samples > 0:
        sample_count = min(sample_count, max_samples)

    num_dim = len(NUM_FEATURE_NAMES[num_mode])
    paths = _prepare_output_paths(
        output_dir=output_dir,
        split=split,
        force=force,
    )

    x_img = np.lib.format.open_memmap(
        paths["x_img"],
        mode="w+",
        dtype=np.uint8,
        shape=(sample_count, input_len, 64, 64, 3),
    )
    y_img = np.lib.format.open_memmap(
        paths["y_img"],
        mode="w+",
        dtype=np.uint8,
        shape=(sample_count, pred_len, 64, 64, 3),
    )
    x_num = np.lib.format.open_memmap(
        paths["x_num"],
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, input_len, num_dim),
    )
    y_num = np.lib.format.open_memmap(
        paths["y_num"],
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, pred_len, num_dim),
    )

    write_index = 0
    total_len = input_len + pred_len

    for file_index, file in enumerate(files, start=1):
        dataset = tf.data.TFRecordDataset(str(file))

        for raw_record in dataset:
            images, ee_pos, actions = _decode_record(
                tf=tf,
                raw_record=raw_record,
                feature_spec=feature_spec,
                raw_sequence_len=raw_sequence_len,
                camera=camera,
            )
            num_seq = _make_num_sequence(
                ee_pos=ee_pos,
                actions=actions,
                num_mode=num_mode,
            )

            for start in starts:
                if write_index >= sample_count:
                    break

                end = start + total_len
                x_slice = slice(start, start + input_len)
                y_slice = slice(start + input_len, end)

                x_img[write_index] = images[x_slice]
                y_img[write_index] = images[y_slice]
                x_num[write_index] = num_seq[x_slice]
                y_num[write_index] = num_seq[y_slice]

                write_index += 1

            if write_index >= sample_count:
                break

        if file_index % 10 == 0 or file_index == len(files):
            print(
                f"[{split}] files={file_index}/{len(files)} "
                f"samples={write_index}/{sample_count}"
            )

        if write_index >= sample_count:
            break

    if write_index != sample_count:
        raise RuntimeError(
            f"Expected to write {sample_count} {split} samples, " f"but wrote {write_index}."
        )

    del x_img, y_img, x_num, y_num
    return sample_count


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required only for BAIR preprocessing. "
            "Install it first, then rerun this script."
        ) from exc

    args = parse_args()

    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files = _list_tfrecords(raw_root, "train")
    test_files = _list_tfrecords(raw_root, "test")

    print(f"Raw root: {raw_root}")
    print(f"Output dir: {output_dir}")
    print(f"Train TFRecord shards: {len(train_files)}")
    print(f"Test TFRecord shards: {len(test_files)}")

    train_count = _write_split(
        tf=tf,
        files=train_files,
        output_dir=output_dir,
        split="train",
        input_len=args.input_len,
        pred_len=args.pred_len,
        raw_sequence_len=args.raw_sequence_len,
        camera=args.camera,
        num_mode=args.num_mode,
        window_stride=args.window_stride,
        max_samples=args.max_train_samples,
        force=args.force,
    )

    test_count = _write_split(
        tf=tf,
        files=test_files,
        output_dir=output_dir,
        split="test",
        input_len=args.input_len,
        pred_len=args.pred_len,
        raw_sequence_len=args.raw_sequence_len,
        camera=args.camera,
        num_mode=args.num_mode,
        window_stride=args.window_stride,
        max_samples=args.max_test_samples,
        force=args.force,
    )

    metadata = {
        "dataset": "BAIR_Robot_Pushing_Small",
        "raw_root": str(raw_root),
        "camera": args.camera,
        "num_mode": args.num_mode,
        "num_feature_names": NUM_FEATURE_NAMES[args.num_mode],
        "input_len": args.input_len,
        "pred_len": args.pred_len,
        "raw_sequence_len": args.raw_sequence_len,
        "window_stride": args.window_stride,
        "image_layout": "NTHWC",
        "image_encoding": "raw_rgb_64x64x3",
        "image_dtype": "uint8",
        "image_shape_per_step": [64, 64, 3],
        "train_samples": train_count,
        "test_samples": test_count,
        "split_protocol": (
            "train arrays are split inside data_provider into "
            "prior_train/gemini_train/val; official BAIR test arrays are "
            "used as test."
        ),
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("BAIR preprocessing finished.")
    print(f"Metadata saved to: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

