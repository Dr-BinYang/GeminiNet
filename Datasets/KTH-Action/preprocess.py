from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

import cv2
import numpy as np

ACTION_NAMES = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking",
]

NUM_FEATURE_NAMES = [
    "mean_intensity",
    "std_intensity",
    "motion_energy",
    "motion_area_ratio",
    "motion_centroid_x",
    "motion_centroid_y",
    "motion_spread_x",
    "motion_spread_y",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Preprocess KTH Action frame sequences for GeminiNet.")
    parser.add_argument(
        "--raw_root",
        type=str,
        default=str(SCRIPT_DIR / "raw"),
        help="Folder containing the extracted KTH action videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR / "processed"),
        help="Folder used to save GeminiNet-ready KTH arrays.",
    )
    parser.add_argument("--img_height", type=int, default=120, help="Output frame height.")
    parser.add_argument("--img_width", type=int, default=120, help="Output frame width.")
    parser.add_argument(
        "--sequence_len",
        type=int,
        default=90,
        help="Fixed sequence length after temporal resampling.",
    )
    parser.add_argument(
        "--input_len",
        type=int,
        default=15,
        help="Recommended historical sequence length stored in metadata.",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=15,
        help="Recommended forecasting sequence length stored in metadata.",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=0,
        help="Maximum sequences to process. Use 0 for all available sequences.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite output_dir if it already exists."
    )
    return parser.parse_args()


def _parse_sequence_dir(path: Path) -> dict:
    name = path.name
    match = re.search(r"person(\d+)_(\w+)_d(\d+)", name.lower())
    action = path.parent.name.lower()
    if match:
        subject_id = int(match.group(1))
        parsed_action = match.group(2)
        scenario_id = int(match.group(3))
        action = parsed_action if parsed_action in ACTION_NAMES else action
    else:
        subject_id = 999
        scenario_id = 999
    return {
        "sequence_name": name,
        "action": action,
        "action_id": ACTION_NAMES.index(action) if action in ACTION_NAMES else -1,
        "subject_id": subject_id,
        "scenario_id": scenario_id,
    }


def _discover_sequence_dirs(raw_root: Path) -> list[Path]:
    sequence_dirs: list[Path] = []
    for action in ACTION_NAMES:
        action_dir = raw_root / action
        if not action_dir.exists():
            continue
        for path in action_dir.iterdir():
            if not path.is_dir():
                continue
            if any(path.glob("*.jpg")):
                sequence_dirs.append(path)

    def sort_key(path: Path) -> tuple[int, str, int, str]:
        info = _parse_sequence_dir(path)
        return (
            int(info["subject_id"]),
            str(info["action"]),
            int(info["scenario_id"]),
            path.name,
        )

    return sorted(sequence_dirs, key=sort_key)


def _read_grayscale_frames(sequence_dir: Path) -> np.ndarray:
    frame_paths = sorted(
        sequence_dir.glob("*.jpg"),
        key=lambda path: (
            int(re.search(r"(\d+)", path.stem).group(1))
            if re.search(r"(\d+)", path.stem)
            else path.name
        ),
    )
    if len(frame_paths) < 2:
        raise ValueError(f"Need at least two frames in {sequence_dir}, got {len(frame_paths)}.")

    frames: list[np.ndarray] = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f"Cannot read image frame: {frame_path}")
        frames.append(frame.astype(np.float32))
    return np.stack(frames, axis=0)


def _temporal_resample(frames: np.ndarray, sequence_len: int) -> np.ndarray:
    if len(frames) < 2:
        raise ValueError(f"Need at least two frames for temporal resampling, got {len(frames)}.")
    indices = np.linspace(0, len(frames) - 1, sequence_len)
    indices = np.clip(np.round(indices).astype(np.int64), 0, len(frames) - 1)
    return frames[indices].astype(np.float32)


def _center_crop_resize(frames: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
    processed: list[np.ndarray] = []
    for frame in frames:
        height, width = frame.shape
        crop_size = min(height, width)
        top = max(0, (height - crop_size) // 2)
        left = max(0, (width - crop_size) // 2)
        crop = frame[top : top + crop_size, left : left + crop_size]
        if crop.shape != (img_height, img_width):
            crop = cv2.resize(
                crop,
                (img_width, img_height),
                interpolation=(
                    cv2.INTER_AREA if crop_size >= max(img_height, img_width) else cv2.INTER_LINEAR
                ),
            )
        processed.append(crop)
    return np.stack(processed, axis=0).astype(np.float32)


def _weighted_centroid_and_spread(weight_map: np.ndarray) -> tuple[float, float, float, float]:
    height, width = weight_map.shape
    total = float(weight_map.sum())
    if total <= 1e-8:
        return 0.5, 0.5, 0.0, 0.0

    y_coords, x_coords = np.mgrid[0:height, 0:width]
    x_norm = x_coords.astype(np.float32) / max(width - 1, 1)
    y_norm = y_coords.astype(np.float32) / max(height - 1, 1)
    weights = weight_map.astype(np.float32) / total

    cx = float((weights * x_norm).sum())
    cy = float((weights * y_norm).sum())
    sx = float(np.sqrt((weights * (x_norm - cx) ** 2).sum()))
    sy = float(np.sqrt((weights * (y_norm - cy) ** 2).sum()))
    return cx, cy, sx, sy


def _motion_statistics(frames_uint8: np.ndarray) -> np.ndarray:
    frames = frames_uint8.astype(np.float32) / 255.0
    features = np.zeros((frames.shape[0], len(NUM_FEATURE_NAMES)), dtype=np.float32)
    previous = frames[0]

    for time_index, frame in enumerate(frames):
        diff = np.abs(frame - previous) if time_index > 0 else np.zeros_like(frame)
        threshold = max(0.03, float(np.percentile(diff, 90)) * 0.5)
        motion_mask = diff > threshold
        cx, cy, sx, sy = _weighted_centroid_and_spread(diff)

        features[time_index] = np.array(
            [
                float(frame.mean()),
                float(frame.std()),
                float(diff.mean()),
                float(motion_mask.mean()),
                cx,
                cy,
                sx,
                sy,
            ],
            dtype=np.float32,
        )
        previous = frame

    return features


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_dirs = _discover_sequence_dirs(raw_root)
    if args.max_sequences > 0:
        sequence_dirs = sequence_dirs[: args.max_sequences]
    if not sequence_dirs:
        raise RuntimeError(f"No KTH frame sequence folders found under {raw_root}.")

    images: list[np.ndarray] = []
    numbers: list[np.ndarray] = []
    records: list[dict] = []

    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        try:
            raw_frames = _read_grayscale_frames(sequence_dir)
            sampled_frames = _temporal_resample(raw_frames, sequence_len=args.sequence_len)
            square_frames = _center_crop_resize(
                sampled_frames, img_height=args.img_height, img_width=args.img_width
            )
            image_sequence = np.clip(square_frames, 0.0, 255.0).astype(np.uint8)[..., None]
            numeric_sequence = _motion_statistics(image_sequence[..., 0])
        except Exception as error:
            print(f"[skip] {index}/{len(sequence_dirs)} {sequence_dir}: {error}")
            continue

        record = _parse_sequence_dir(sequence_dir)
        record.update(
            {
                "frame_dir": str(sequence_dir),
                "original_frames": int(len(raw_frames)),
                "processed_frames": int(args.sequence_len),
            }
        )
        images.append(image_sequence)
        numbers.append(numeric_sequence)
        records.append(record)

        if index % 50 == 0 or index == len(sequence_dirs):
            print(
                f"[process] {index}/{len(sequence_dirs)} success={len(images)} latest={sequence_dir.name}"
            )

    if not images:
        raise RuntimeError("No KTH sequences were successfully processed.")

    image_array = np.stack(images, axis=0).astype(np.uint8)
    number_array = np.stack(numbers, axis=0).astype(np.float32)
    if not np.isfinite(number_array).all():
        raise ValueError("numbers.npy contains non-finite values.")

    np.save(output_dir / "images.npy", image_array)
    np.save(output_dir / "numbers.npy", number_array)

    metadata = {
        "dataset": "KTH_Action_MotionStats",
        "image_modality": "KTH grayscale human-action frame sequences",
        "numerical_modality": "video-derived motion-state statistics",
        "raw_root": str(raw_root),
        "num_sequences": int(image_array.shape[0]),
        "sequence_len": int(image_array.shape[1]),
        "image_shape": list(image_array.shape[2:]),
        "array_shape_images": list(image_array.shape),
        "array_shape_numbers": list(number_array.shape),
        "image_dtype": str(image_array.dtype),
        "number_dtype": str(number_array.dtype),
        "actions": ACTION_NAMES,
        "num_feature_names": NUM_FEATURE_NAMES,
        "num_vars": len(NUM_FEATURE_NAMES),
        "spatial_processing": "center-crop the original 160x120 frame to a square region, then resize to 120x120 if needed",
        "task_recommendation": {
            "input_len": int(args.input_len),
            "pred_len": int(args.pred_len),
            "meaning": "past 15 grayscale action frames and motion statistics -> future 15 frames and motion statistics",
        },
        "notes": [
            "KTH Action is originally a single video modality dataset.",
            "The numerical modality is constructed from each frame sequence using interpretable motion statistics.",
            "Sequences are sorted by subject id before saving, so range-based train/validation/test splits reduce subject leakage.",
            "Action labels are not used as numerical variables because they are static class labels rather than forecast targets.",
        ],
        "records": records,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("KTH Action preprocessing finished.")
    print(f"images: {image_array.shape} {image_array.dtype}")
    print(f"numbers: {number_array.shape} {number_array.dtype}")
    print(f"metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
