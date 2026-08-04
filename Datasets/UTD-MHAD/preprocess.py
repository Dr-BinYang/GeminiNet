from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

DEFAULT_NUM_FEATURE_NAMES = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Preprocess UTD-MHAD RGB/depth + inertial sequences for GeminiNet."
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default=str(SCRIPT_DIR / "raw"),
        help="Folder containing downloaded/extracted UTD-MHAD .mat or .zip files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR / "processed"),
        help="Folder used to save GeminiNet-ready arrays.",
    )
    parser.add_argument(
        "--image_source",
        type=str,
        default="rgb",
        choices=["rgb", "depth"],
        help="Image modality source. rgb uses *_color.avi; depth uses *_depth.mat.",
    )
    parser.add_argument("--img_height", type=int, default=240, help="Output frame height.")
    parser.add_argument("--img_width", type=int, default=320, help="Output frame width.")
    parser.add_argument(
        "--sequence_len",
        type=int,
        default=60,
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
        "--numeric_source",
        type=str,
        default="auto",
        choices=["auto", "inertial", "skeleton"],
        help="Numerical modality source. auto prefers inertial and falls back to skeleton.",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=0,
        help="Maximum paired sequences to process. Use 0 for all available pairs.",
    )
    parser.add_argument(
        "--extract_zip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract .zip files found under raw_root before scanning .mat files.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite output_dir if it already exists."
    )
    return parser.parse_args()


def _extract_zip_files(raw_root: Path) -> None:
    for zip_path in raw_root.rglob("*.zip"):
        extract_dir = zip_path.with_suffix("")
        if extract_dir.exists():
            continue
        print(f"[extract] {zip_path} -> {extract_dir}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)


def _sequence_key(path: Path) -> str | None:
    match = re.search(r"(a\d+_s\d+_t\d+)", path.stem.lower())
    if match:
        return match.group(1)
    return None


def _group_sequence_files(raw_root: Path) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = {}
    for path in raw_root.rglob("*.mat"):
        key = _sequence_key(path)
        if key is None:
            continue
        stem = path.stem.lower()
        role = None
        if "depth" in stem:
            role = "depth"
        elif "inertial" in stem or "inertia" in stem or "imu" in stem:
            role = "inertial"
        elif "skeleton" in stem or "joint" in stem:
            role = "skeleton"
        if role is None:
            continue
        groups.setdefault(key, {})[role] = path
    for path in raw_root.rglob("*.avi"):
        key = _sequence_key(path)
        if key is None:
            continue
        stem = path.stem.lower()
        if "color" in stem or "rgb" in stem:
            groups.setdefault(key, {})["rgb"] = path
    return groups


def _numeric_arrays_from_mat(path: Path) -> list[np.ndarray]:
    data = loadmat(path)
    arrays: list[np.ndarray] = []
    for key, value in data.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and array.size > 0:
            arrays.append(array)
    arrays.sort(key=lambda array: array.size, reverse=True)
    return arrays


def _load_rgb_sequence(
    path: Path, img_height: int, img_width: int, sequence_len: int
) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "opencv-python is required for reading UTD-MHAD RGB .avi files. "
            "Install it with: pip install opencv-python"
        ) from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open RGB video: {path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.astype(np.float32))
    capture.release()

    if len(frames) < 2:
        raise ValueError(f"Need at least two RGB frames, got {len(frames)} from {path}.")

    array = np.stack(frames, axis=0).astype(np.float32)
    array = _fill_nonfinite(array)
    array = _temporal_resample_frames(array, sequence_len=sequence_len)

    if array.shape[1] != img_height or array.shape[2] != img_width:
        tensor = torch.from_numpy(array).float().permute(0, 3, 1, 2).contiguous()
        tensor = F.interpolate(
            tensor,
            size=(img_height, img_width),
            mode="bilinear",
            align_corners=False,
        )
        array = tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _load_depth_sequence(
    path: Path, img_height: int, img_width: int, sequence_len: int
) -> np.ndarray:
    arrays = [array for array in _numeric_arrays_from_mat(path) if array.ndim >= 3]
    if not arrays:
        raise ValueError(f"No depth-like 3D array found in {path}.")

    array = np.squeeze(arrays[0]).astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"Depth array must be 3D after squeeze, got {array.shape} from {path}.")

    shape = array.shape
    spatial_axes = _find_spatial_axes(shape, target_height=img_height, target_width=img_width)
    time_axis = [axis for axis in range(3) if axis not in spatial_axes][0]
    height_axis, width_axis = spatial_axes
    array = np.moveaxis(array, [time_axis, height_axis, width_axis], [0, 1, 2])
    array = _fill_nonfinite(array)

    array = _temporal_resample_frames(array, sequence_len=sequence_len)
    if array.shape[1] != img_height or array.shape[2] != img_width:
        tensor = torch.from_numpy(array[:, None, :, :]).float()
        tensor = F.interpolate(
            tensor,
            size=(img_height, img_width),
            mode="bilinear",
            align_corners=False,
        )
        array = tensor[:, 0].cpu().numpy().astype(np.float32)

    return array[..., None].astype(np.float32)


def _find_spatial_axes(
    shape: tuple[int, int, int], target_height: int, target_width: int
) -> tuple[int, int]:
    axes = range(3)
    best_pair = None
    best_score = float("inf")
    for height_axis in axes:
        for width_axis in axes:
            if height_axis == width_axis:
                continue
            score = abs(shape[height_axis] - target_height) + abs(shape[width_axis] - target_width)
            if score < best_score:
                best_score = score
                best_pair = (height_axis, width_axis)
    if best_pair is None:
        raise ValueError(f"Cannot infer spatial axes from shape={shape}.")
    return best_pair


def _fill_nonfinite(array: np.ndarray) -> np.ndarray:
    valid = np.isfinite(array)
    if valid.all():
        return array.astype(np.float32)
    if not valid.any():
        raise ValueError("Array contains no finite values.")
    fill_value = float(np.nanmean(array))
    return np.where(valid, array, fill_value).astype(np.float32)


def _temporal_resample_frames(frames: np.ndarray, sequence_len: int) -> np.ndarray:
    if len(frames) < 2:
        raise ValueError(f"Need at least two frames for temporal resampling, got {len(frames)}.")
    indices = np.linspace(0, len(frames) - 1, sequence_len)
    indices = np.clip(np.round(indices).astype(np.int64), 0, len(frames) - 1)
    return frames[indices].astype(np.float32)


def _temporal_resample_features(features: np.ndarray, sequence_len: int) -> np.ndarray:
    if features.shape[0] < 2:
        raise ValueError(
            f"Need at least two timesteps for temporal resampling, got {features.shape[0]}."
        )
    old_x = np.linspace(0.0, 1.0, features.shape[0])
    new_x = np.linspace(0.0, 1.0, sequence_len)
    resampled = np.zeros((sequence_len, features.shape[1]), dtype=np.float32)
    for feature_index in range(features.shape[1]):
        resampled[:, feature_index] = np.interp(new_x, old_x, features[:, feature_index])
    return resampled


def _load_inertial_sequence(path: Path, sequence_len: int) -> tuple[np.ndarray, list[str]]:
    arrays = [array for array in _numeric_arrays_from_mat(path) if array.ndim == 2]
    if not arrays:
        raise ValueError(f"No inertial-like 2D array found in {path}.")

    array = np.squeeze(arrays[0]).astype(np.float32)
    if array.shape[0] <= array.shape[1]:
        array = array.T
    array = _fill_nonfinite(array)
    if array.shape[1] > 12:
        array = array[:, :12]

    names = DEFAULT_NUM_FEATURE_NAMES[: array.shape[1]]
    if len(names) < array.shape[1]:
        names = names + [f"imu_{index}" for index in range(len(names), array.shape[1])]

    return _temporal_resample_features(array, sequence_len=sequence_len), names


def _load_skeleton_sequence(path: Path, sequence_len: int) -> tuple[np.ndarray, list[str]]:
    arrays = [array for array in _numeric_arrays_from_mat(path) if array.ndim >= 2]
    if not arrays:
        raise ValueError(f"No skeleton-like numeric array found in {path}.")

    array = np.squeeze(arrays[0]).astype(np.float32)
    if array.ndim == 2:
        if array.shape[0] <= array.shape[1]:
            array = array.T
        features = array
    else:
        time_axis = int(np.argmax(array.shape))
        array = np.moveaxis(array, time_axis, 0)
        features = array.reshape(array.shape[0], -1)

    features = _fill_nonfinite(features)
    features = features[:, : min(features.shape[1], 60)]
    names = [f"skeleton_{index}" for index in range(features.shape[1])]
    return _temporal_resample_features(features, sequence_len=sequence_len), names


def _load_numeric_sequence(
    files: dict[str, Path], source: str, sequence_len: int
) -> tuple[np.ndarray, list[str], str]:
    if source in ["auto", "inertial"] and "inertial" in files:
        values, names = _load_inertial_sequence(files["inertial"], sequence_len=sequence_len)
        return values, names, "inertial"
    if source in ["auto", "skeleton"] and "skeleton" in files:
        values, names = _load_skeleton_sequence(files["skeleton"], sequence_len=sequence_len)
        return values, names, "skeleton"
    raise FileNotFoundError(
        f"No usable numerical modality found for source={source}; files={files}."
    )


def _parse_sequence_metadata(key: str) -> dict:
    match = re.match(r"a(\d+)_s(\d+)_t(\d+)", key)
    if not match:
        return {"sequence_key": key}
    action, subject, trial = [int(value) for value in match.groups()]
    return {
        "sequence_key": key,
        "action_id": action,
        "subject_id": subject,
        "trial_id": trial,
    }


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    if args.extract_zip:
        _extract_zip_files(raw_root)

    groups = _group_sequence_files(raw_root)
    paired_keys = [
        key
        for key, files in sorted(groups.items())
        if args.image_source in files
        and (
            (args.numeric_source in ["auto", "inertial"] and "inertial" in files)
            or (args.numeric_source in ["auto", "skeleton"] and "skeleton" in files)
        )
    ]
    if args.max_sequences > 0:
        paired_keys = paired_keys[: args.max_sequences]
    if not paired_keys:
        raise RuntimeError(
            f"No paired UTD-MHAD {args.image_source} + numerical files found. "
            "Place extracted UTD-MHAD files under datasets/UTD_MHAD_Depth_Inertial/raw."
        )

    images: list[np.ndarray] = []
    numbers: list[np.ndarray] = []
    records: list[dict] = []
    feature_names: list[str] | None = None
    numeric_source_used = None

    for index, key in enumerate(paired_keys, start=1):
        files = groups[key]
        try:
            if args.image_source == "rgb":
                image_sequence = _load_rgb_sequence(
                    files["rgb"],
                    img_height=args.img_height,
                    img_width=args.img_width,
                    sequence_len=args.sequence_len,
                )
            else:
                image_sequence = _load_depth_sequence(
                    files["depth"],
                    img_height=args.img_height,
                    img_width=args.img_width,
                    sequence_len=args.sequence_len,
                )
            numeric, names, source_used = _load_numeric_sequence(
                files=files,
                source=args.numeric_source,
                sequence_len=args.sequence_len,
            )
        except Exception as error:
            print(f"[skip] {index}/{len(paired_keys)} {key}: {error}")
            continue

        if feature_names is None:
            feature_names = names
            numeric_source_used = source_used
        if numeric.shape[1] != len(feature_names):
            numeric = numeric[:, : len(feature_names)]

        images.append(
            image_sequence.astype(np.uint8)
            if args.image_source == "rgb"
            else image_sequence.astype(np.float32)
        )
        numbers.append(numeric.astype(np.float32))
        record = _parse_sequence_metadata(key)
        record.update(
            {
                "image_file": str(files[args.image_source]),
                "image_source": args.image_source,
                "numeric_file": str(files.get(source_used, "")),
                "numeric_source": source_used,
            }
        )
        records.append(record)

        if index % 20 == 0 or index == len(paired_keys):
            print(f"[process] {index}/{len(paired_keys)} success={len(images)} latest={key}")

    if not images:
        raise RuntimeError("No UTD-MHAD sequences were successfully processed.")
    if feature_names is None:
        raise RuntimeError("No numerical feature names were resolved.")

    image_array = np.stack(images, axis=0)
    if args.image_source == "rgb":
        image_array = image_array.astype(np.uint8)
    else:
        image_array = image_array.astype(np.float32)
    number_array = np.stack(numbers, axis=0).astype(np.float32)
    if not np.isfinite(image_array).all():
        raise ValueError("images.npy contains non-finite values.")
    if not np.isfinite(number_array).all():
        raise ValueError("numbers.npy contains non-finite values.")

    np.save(output_dir / "images.npy", image_array)
    np.save(output_dir / "numbers.npy", number_array)

    metadata = {
        "dataset": "UTD_MHAD_RGB_Inertial",
        "image_modality": (
            "UTD-MHAD RGB human action video sequences"
            if args.image_source == "rgb"
            else "UTD-MHAD depth-frame human action sequences"
        ),
        "numerical_modality": f"UTD-MHAD {numeric_source_used} motion-sensor sequences",
        "image_source": args.image_source,
        "image_shape": list(image_array.shape[2:]),
        "array_shape_images": list(image_array.shape),
        "array_shape_numbers": list(number_array.shape),
        "image_dtype": str(image_array.dtype),
        "number_dtype": str(number_array.dtype),
        "num_sequences": int(image_array.shape[0]),
        "sequence_len": int(image_array.shape[1]),
        "num_feature_names": feature_names,
        "num_vars": len(feature_names),
        "raw_root": str(raw_root),
        "records": records,
        "task_recommendation": {
            "input_len": int(args.input_len),
            "pred_len": int(args.pred_len),
            "meaning": (
                "past 15 RGB/sensor frames -> future 15 RGB/sensor frames"
                if args.image_source == "rgb"
                else "past 15 depth/sensor frames -> future 15 depth/sensor frames"
            ),
        },
        "notes": [
            "RGB videos are used as the default image modality; depth frames remain available through --image_source depth.",
            "The numerical modality prefers inertial signals and falls back to skeleton features when requested.",
            "Sequences are temporally resampled to a fixed length to support stable sliding-window forecasting.",
            "The DataLoader splits by sequence key, so windows from the same original trial do not cross train/val/test splits.",
        ],
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("UTD-MHAD preprocessing finished.")
    print(f"images: {image_array.shape} {image_array.dtype}")
    print(f"numbers: {number_array.shape} float32")
    print(f"metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
