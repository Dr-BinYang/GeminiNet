from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

NUM_FEATURES = [
    ("ff", "wind_speed"),
    ("precip", "station_precipitation"),
    ("hu", "relative_humidity"),
    ("td", "dew_point_temperature"),
    ("t", "temperature"),
    ("psl", "sea_level_pressure"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Preprocess MeteoNet NW data for GeminiNet.")
    parser.add_argument(
        "--raw_root",
        type=str,
        default="./datasets/MeteoNet_NW/raw",
        help="Folder containing extracted MeteoNet NW archives.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/MeteoNet_NW/processed",
        help="Folder used to save GeminiNet-ready arrays.",
    )
    parser.add_argument(
        "--img_size", type=int, default=64, help="Output radar image height and width."
    )
    parser.add_argument(
        "--max_hours",
        type=int,
        default=0,
        help="Maximum hours to process. Use 0 for all available data.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite output_dir if it already exists."
    )
    return parser.parse_args()


def _find_rainfall_files(raw_root: Path) -> list[Path]:
    files = sorted({path.resolve() for path in raw_root.rglob("rainfall-NW-*/*.npz")})
    if not files:
        raise FileNotFoundError(f"No extracted MeteoNet NW rainfall files found under {raw_root}.")
    return files


def _find_station_csvs(raw_root: Path) -> list[Path]:
    files = sorted(
        path.resolve()
        for path in raw_root.rglob("NW*.csv")
        if re.fullmatch(r"NW\d{4}\.csv", path.name)
    )
    if not files:
        raise FileNotFoundError(
            f"No extracted MeteoNet NW station CSV files found under {raw_root}."
        )
    return files


def _block_average_2d(frame: np.ndarray, img_size: int) -> np.ndarray:
    """Fast area downsampling by cropping to a divisible grid and block averaging."""
    height, width = frame.shape
    block_h = height // img_size
    block_w = width // img_size
    if block_h <= 0 or block_w <= 0:
        raise ValueError(f"img_size={img_size} is too large for radar frame shape {frame.shape}.")

    crop_h = block_h * img_size
    crop_w = block_w * img_size
    cropped = frame[:crop_h, :crop_w]
    return cropped.reshape(img_size, block_h, img_size, block_w).mean(axis=(1, 3))


def _load_hourly_radar(
    npz_files: list[Path],
    img_size: int,
    max_hours: int,
) -> tuple[np.ndarray, list[pd.Timestamp]]:
    images: list[np.ndarray] = []
    timestamps: list[pd.Timestamp] = []

    for file_index, npz_path in enumerate(npz_files, start=1):
        archive = np.load(npz_path, allow_pickle=True)
        data = archive["data"]
        dates = pd.to_datetime(list(archive["dates"]))
        hours = pd.Series(dates).dt.floor("h")

        for hour in sorted(hours.unique()):
            if max_hours > 0 and len(images) >= max_hours:
                break

            indices = np.where(hours.to_numpy() == hour)[0]
            frames = data[indices]
            frames = np.where(frames >= 0, frames, 0).astype(np.float32)
            hourly_frame = frames.mean(axis=0)
            small_frame = _block_average_2d(hourly_frame, img_size=img_size)
            images.append(small_frame[..., None].astype(np.float32))
            timestamps.append(pd.Timestamp(hour))

        print(
            f"[radar] files={file_index}/{len(npz_files)} "
            f"hours={len(images)} latest={timestamps[-1] if timestamps else 'NA'}"
        )

        if max_hours > 0 and len(images) >= max_hours:
            break

    if not images:
        raise RuntimeError("No hourly radar frames were produced.")

    return np.stack(images, axis=0), timestamps


def _load_hourly_station_features(
    station_csvs: list[Path],
    target_timestamps: list[pd.Timestamp],
) -> np.ndarray:
    usecols = ["date"] + [source_name for source_name, _ in NUM_FEATURES]
    frames = [pd.read_csv(path, usecols=usecols) for path in station_csvs]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d %H:%M")
    df["hour"] = df["date"].dt.floor("h")

    grouped = df.groupby("hour")[[source_name for source_name, _ in NUM_FEATURES]].mean()
    grouped = grouped.rename(
        columns={source_name: target_name for source_name, target_name in NUM_FEATURES}
    )

    target_index = pd.DatetimeIndex(target_timestamps)
    aligned = grouped.reindex(target_index)
    aligned = aligned.interpolate(method="time", limit_direction="both")
    aligned = aligned.ffill().bfill()

    if aligned.isna().any().any():
        missing = aligned.columns[aligned.isna().any()].tolist()
        raise RuntimeError(f"Station features still contain NaN values: {missing}")

    return aligned.to_numpy(dtype=np.float32)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite it.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    rainfall_files = _find_rainfall_files(raw_root)
    station_csvs = _find_station_csvs(raw_root)

    print(f"Raw root: {raw_root}")
    print(f"Rainfall files: {len(rainfall_files)}")
    print(f"Station files: {len(station_csvs)}")
    print(f"Output dir: {output_dir}")

    images, timestamps = _load_hourly_radar(
        npz_files=rainfall_files,
        img_size=args.img_size,
        max_hours=args.max_hours,
    )
    numbers = _load_hourly_station_features(
        station_csvs=station_csvs,
        target_timestamps=timestamps,
    )

    if len(images) != len(numbers):
        raise RuntimeError(
            f"Image and numerical sequence lengths differ: {len(images)} vs {len(numbers)}."
        )

    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "numbers.npy", numbers)

    metadata = {
        "dataset": "MeteoNet_NW",
        "time_frequency": "1h",
        "image_modality": "NW radar rainfall aggregated to hourly maps",
        "numerical_modality": "regional hourly means from NW ground stations",
        "image_shape": list(images.shape[1:]),
        "num_feature_names": [target_name for _, target_name in NUM_FEATURES],
        "num_vars": len(NUM_FEATURES),
        "num_timesteps": int(len(images)),
        "start_time": str(timestamps[0]),
        "end_time": str(timestamps[-1]),
        "raw_root": str(raw_root),
        "rainfall_files": [str(path) for path in rainfall_files],
        "station_csvs": [str(path) for path in station_csvs],
        "notes": [
            "Radar frames are aggregated from 5-minute frames to hourly mean rainfall maps.",
            "Negative radar values are treated as missing/background and set to 0 before aggregation.",
            "Station records are aggregated to hourly regional means and aligned to radar timestamps.",
            "Train/validation/test windows are created later by data_provider without crossing split boundaries.",
        ],
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("MeteoNet preprocessing finished.")
    print(f"images: {images.shape} float32")
    print(f"numbers: {numbers.shape} float32")
    print(f"metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()