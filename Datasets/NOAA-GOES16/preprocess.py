from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

import numpy as np
import requests
import torch
import torch.nn.functional as F

S3_BUCKET_URL = "https://noaa-goes16.s3.amazonaws.com"
PRODUCT = "ABI-L2-CMIPC"
CHANNEL = 13
NUM_FEATURE_NAMES = [
    "bt_mean_k",
    "bt_std_k",
    "bt_min_k",
    "bt_p10_k",
    "cold_fraction_235k",
    "cold_fraction_220k",
    "bt_gradient_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Preprocess NOAA GOES-16 ABI Channel-13 infrared data for GeminiNet."
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default=str(SCRIPT_DIR / "raw"),
        help="Folder for downloaded GOES-16 NetCDF files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR / "processed"),
        help="Folder used to save GeminiNet-ready arrays.",
    )
    parser.add_argument(
        "--start_time", type=str, required=True, help="UTC start time in ISO-8601 format."
    )
    parser.add_argument(
        "--end_time", type=str, required=True, help="UTC end time in ISO-8601 format."
    )
    parser.add_argument(
        "--sample_minutes",
        type=int,
        default=10,
        help="Temporal sampling interval in minutes. GOES CONUS files are commonly available every 5 minutes.",
    )
    parser.add_argument("--img_size", type=int, default=128, help="Output image height and width.")
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Maximum files after temporal sampling. Use 0 for the full selected period.",
    )
    parser.add_argument(
        "--download_retries",
        type=int,
        default=3,
        help="Number of download retries for each NetCDF file.",
    )
    parser.add_argument(
        "--keep_raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to keep downloaded raw NetCDF files.",
    )
    parser.add_argument(
        "--disable_ssl_verify",
        action="store_true",
        help="Disable SSL verification if local/network SSL issues interrupt public NOAA S3 access.",
    )
    parser.add_argument(
        "--skip_failed_files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files that fail download/read after retries instead of aborting the whole preprocessing run.",
    )
    parser.add_argument(
        "--max_failed_files",
        type=int,
        default=200,
        help="Abort preprocessing if skipped files exceed this number. Set to 0 to disable the limit.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite output_dir if it already exists."
    )
    return parser.parse_args()


def _parse_utc_time(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _day_of_year_path(timestamp: datetime) -> tuple[int, int, int]:
    return timestamp.year, int(timestamp.strftime("%j")), timestamp.hour


def _iter_hours(start_time: datetime, end_time: datetime):
    current = start_time.replace(minute=0, second=0, microsecond=0)
    while current <= end_time:
        yield current
        current += timedelta(hours=1)


def _http_get_with_retries(url: str, retries: int, verify_ssl: bool, **kwargs):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, verify=verify_ssl, **kwargs)
            response.raise_for_status()
            return response
        except Exception as error:
            last_error = error
            if attempt >= retries:
                raise
            time.sleep(2 * attempt)
    raise last_error


def _list_s3_keys(prefix: str, retries: int, verify_ssl: bool) -> list[str]:
    keys: list[str] = []
    continuation_token = None

    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
        }
        if continuation_token:
            params["continuation-token"] = continuation_token

        response = _http_get_with_retries(
            S3_BUCKET_URL,
            retries=retries,
            verify_ssl=verify_ssl,
            params=params,
            timeout=60,
        )

        root = ET.fromstring(response.text)
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

        for item in root.findall("s3:Contents", namespace):
            key = item.findtext("s3:Key", default="", namespaces=namespace)
            if key:
                keys.append(key)

        is_truncated = root.findtext("s3:IsTruncated", default="false", namespaces=namespace)
        if is_truncated.lower() != "true":
            break
        continuation_token = root.findtext(
            "s3:NextContinuationToken", default="", namespaces=namespace
        )
        if not continuation_token:
            break

    return keys


def _timestamp_from_goes_key(key: str) -> datetime:
    match = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
    if not match:
        raise ValueError(f"Cannot parse GOES timestamp from key: {key}")
    year, day_of_year, hour, minute, second = [int(part) for part in match.groups()]
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=day_of_year - 1, hours=hour, minutes=minute, seconds=second
    )


def _collect_candidate_files(
    start_time: datetime,
    end_time: datetime,
    retries: int,
    verify_ssl: bool,
) -> list[tuple[datetime, str]]:
    records: list[tuple[datetime, str]] = []
    for hour_time in _iter_hours(start_time, end_time):
        year, day_of_year, hour = _day_of_year_path(hour_time)
        prefix = f"{PRODUCT}/{year}/{day_of_year:03d}/{hour:02d}/"
        keys = _list_s3_keys(prefix, retries=retries, verify_ssl=verify_ssl)
        c13_keys = [key for key in keys if f"M6C{CHANNEL:02d}" in key and key.endswith(".nc")]
        for key in c13_keys:
            timestamp = _timestamp_from_goes_key(key)
            if start_time <= timestamp <= end_time:
                records.append((timestamp, key))
        print(
            f"[list] {prefix} C{CHANNEL:02d} files={len(c13_keys)} total_candidates={len(records)}"
        )
    records = sorted(set(records), key=lambda item: item[0])
    return records


def _sample_records(
    records: list[tuple[datetime, str]],
    start_time: datetime,
    sample_minutes: int,
    max_files: int,
) -> list[tuple[datetime, str]]:
    if sample_minutes <= 0:
        raise ValueError(f"sample_minutes must be positive, got {sample_minutes}.")

    sampled: list[tuple[datetime, str]] = []
    next_keep = start_time
    interval = timedelta(minutes=sample_minutes)

    for timestamp, key in records:
        if timestamp >= next_keep:
            sampled.append((timestamp, key))
            while next_keep <= timestamp:
                next_keep += interval
        if max_files > 0 and len(sampled) >= max_files:
            break

    return sampled


def _download_file(key: str, raw_root: Path, retries: int, verify_ssl: bool) -> Path:
    local_path = raw_root / key
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{S3_BUCKET_URL}/{key}"
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120, verify=verify_ssl) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            tmp_path.replace(local_path)
            return local_path
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt >= retries:
                raise
            time.sleep(2 * attempt)

    return local_path


def _read_cmi(path: Path) -> np.ndarray:
    try:
        import xarray as xr
    except ImportError as error:
        raise ImportError(
            "xarray is required for GOES preprocessing. Install requirements first."
        ) from error

    try:
        dataset = xr.open_dataset(path, engine="h5netcdf", mask_and_scale=True)
    except Exception:
        dataset = xr.open_dataset(path, mask_and_scale=True)

    try:
        if "CMI" not in dataset:
            raise KeyError(f"Variable CMI not found in {path}")
        cmi = dataset["CMI"].astype("float32").values
    finally:
        dataset.close()

    return np.asarray(cmi, dtype=np.float32)


def _fill_nonfinite(frame: np.ndarray) -> np.ndarray:
    valid = np.isfinite(frame)
    if valid.all():
        return frame.astype(np.float32)
    if not valid.any():
        raise ValueError("A GOES CMI frame contains no finite values.")
    fill_value = float(np.nanmean(frame))
    return np.where(valid, frame, fill_value).astype(np.float32)


def _resize_frame(frame: np.ndarray, img_size: int) -> np.ndarray:
    tensor = torch.from_numpy(frame[None, None, :, :]).float()
    resized = F.interpolate(
        tensor,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].cpu().numpy().astype(np.float32)


def _numeric_features_from_frame(frame: np.ndarray) -> np.ndarray:
    frame = _fill_nonfinite(frame)
    gy, gx = np.gradient(frame)
    gradient_mean = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    return np.asarray(
        [
            float(np.mean(frame)),
            float(np.std(frame)),
            float(np.min(frame)),
            float(np.percentile(frame, 10)),
            float(np.mean(frame < 235.0)),
            float(np.mean(frame < 220.0)),
            gradient_mean,
        ],
        dtype=np.float32,
    )


def _preprocess_records(
    records: list[tuple[datetime, str]],
    raw_root: Path,
    img_size: int,
    retries: int,
    keep_raw: bool,
    verify_ssl: bool,
    skip_failed_files: bool,
    max_failed_files: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[dict]]:
    images: list[np.ndarray] = []
    numbers: list[np.ndarray] = []
    timestamps: list[str] = []
    local_files: list[str] = []
    skipped_files: list[dict] = []

    for index, (timestamp, key) in enumerate(records, start=1):
        try:
            local_path = _download_file(
                key=key, raw_root=raw_root, retries=retries, verify_ssl=verify_ssl
            )
            frame = _read_cmi(local_path)
            frame = _fill_nonfinite(frame)
        except Exception as error:
            skipped_files.append(
                {
                    "index": index,
                    "timestamp": timestamp.isoformat(),
                    "key": key,
                    "error": repr(error),
                }
            )
            print(
                f"[skip] files={index}/{len(records)} timestamp={timestamp.isoformat()} key={key} error={error}"
            )
            if not skip_failed_files:
                raise
            if max_failed_files > 0 and len(skipped_files) > max_failed_files:
                raise RuntimeError(
                    f"Too many skipped GOES files: {len(skipped_files)} > {max_failed_files}. "
                    "Increase --max_failed_files or fix network/download issues."
                ) from error
            continue

        numbers.append(_numeric_features_from_frame(frame))
        resized = _resize_frame(frame, img_size=img_size)
        images.append(resized[..., None])
        timestamps.append(timestamp.isoformat())
        local_files.append(str(local_path))

        if not keep_raw and local_path.exists():
            local_path.unlink()

        if index % 20 == 0 or index == len(records):
            print(
                f"[process] files={index}/{len(records)} "
                f"success={len(images)} skipped={len(skipped_files)} "
                f"latest={timestamp.isoformat()} image={resized.shape}"
            )

    if not images:
        raise RuntimeError("No GOES files were successfully processed.")

    return (
        np.stack(images, axis=0).astype(np.float32),
        np.stack(numbers, axis=0).astype(np.float32),
        timestamps,
        local_files,
        skipped_files,
    )


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    start_time = _parse_utc_time(args.start_time)
    end_time = _parse_utc_time(args.end_time)

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    print(f"GOES source: {S3_BUCKET_URL}/{PRODUCT}")
    print(f"Channel: C{CHANNEL:02d}")
    print(f"Raw root: {raw_root}")
    print(f"Output dir: {output_dir}")
    print(f"Time range UTC: {start_time.isoformat()} -> {end_time.isoformat()}")
    print(f"Sample interval: {args.sample_minutes} minutes")

    verify_ssl = not args.disable_ssl_verify
    listing_end_time = end_time
    if args.max_files > 0:
        listing_end_time = min(
            end_time,
            start_time + timedelta(minutes=args.sample_minutes * args.max_files + 120),
        )
        print(f"Listing end UTC: {listing_end_time.isoformat()}")

    candidates = _collect_candidate_files(
        start_time=start_time,
        end_time=listing_end_time,
        retries=args.download_retries,
        verify_ssl=verify_ssl,
    )
    if not candidates:
        raise RuntimeError("No GOES-16 ABI C13 files found for the selected time range.")

    records = _sample_records(
        records=candidates,
        start_time=start_time,
        sample_minutes=args.sample_minutes,
        max_files=args.max_files,
    )
    if not records:
        raise RuntimeError("No GOES-16 ABI C13 files remain after temporal sampling.")

    print(f"Candidate files: {len(candidates)}")
    print(f"Selected files: {len(records)}")

    images, numbers, timestamps, local_files, skipped_files = _preprocess_records(
        records=records,
        raw_root=raw_root,
        img_size=args.img_size,
        retries=args.download_retries,
        keep_raw=bool(args.keep_raw),
        verify_ssl=verify_ssl,
        skip_failed_files=bool(args.skip_failed_files),
        max_failed_files=args.max_failed_files,
    )

    if len(images) != len(numbers):
        raise RuntimeError(
            f"Image and numerical sequence lengths differ: {len(images)} vs {len(numbers)}."
        )
    if not np.isfinite(images).all():
        raise ValueError("images.npy contains non-finite values.")
    if not np.isfinite(numbers).all():
        raise ValueError("numbers.npy contains non-finite values.")

    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "numbers.npy", numbers)

    metadata = {
        "dataset": "GOES16_ABI_C13_CONUS",
        "source": f"{S3_BUCKET_URL}/{PRODUCT}",
        "product": PRODUCT,
        "satellite": "GOES-16",
        "sector": "CONUS",
        "channel": "C13",
        "channel_description": "ABI clean infrared longwave window, approximately 10.3 micrometers",
        "image_modality": "GOES-16 ABI Channel-13 infrared brightness temperature fields",
        "numerical_modality": "regional infrared cloud-state statistics derived from the same C13 frames",
        "time_frequency": f"{args.sample_minutes}min",
        "start_time": timestamps[0],
        "end_time": timestamps[-1],
        "num_timesteps": int(len(timestamps)),
        "native_frame_shape": list(_read_cmi(Path(local_files[0])).shape) if local_files else None,
        "image_shape": list(images.shape[1:]),
        "num_feature_names": NUM_FEATURE_NAMES,
        "num_vars": len(NUM_FEATURE_NAMES),
        "brightness_temperature_unit": "K",
        "cold_cloud_thresholds_k": [235.0, 220.0],
        "sample_minutes": int(args.sample_minutes),
        "selected_files": int(len(records)),
        "processed_files": int(len(timestamps)),
        "skipped_files": int(len(skipped_files)),
        "task_recommendation": {
            "input_len": 12,
            "pred_len": 12,
            "meaning": f"past {12 * args.sample_minutes} minutes -> future {12 * args.sample_minutes} minutes",
        },
        "notes": [
            "Only ABI-L2-CMIPC Channel 13 files are used.",
            "The image modality is infrared brightness temperature, not precipitation and not near-surface air temperature.",
            "The numerical modality is derived from meteorologically meaningful infrared cloud-top statistics.",
            "Cold-cloud fractions are computed from brightness-temperature thresholds commonly used as deep-convection proxies.",
        ],
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    if skipped_files:
        with open(output_dir / "skipped_files.json", "w", encoding="utf-8") as file:
            json.dump(skipped_files, file, indent=2, ensure_ascii=False)

    print("GOES-16 ABI C13 preprocessing finished.")
    print(f"images: {images.shape} float32")
    print(f"numbers: {numbers.shape} float32")
    print(f"skipped files: {len(skipped_files)}")
    print(f"metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

