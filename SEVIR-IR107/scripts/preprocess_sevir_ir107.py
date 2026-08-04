from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F

SEVIR_BASE_URL = "https://sevir.s3.amazonaws.com"
SEVIR_CATALOG_URL = f"{SEVIR_BASE_URL}/CATALOG.csv"
IMAGE_TYPE = "ir107"
NUM_FEATURE_NAMES = [
    "ir107_mean",
    "ir107_std",
    "ir107_min_cold_cloud_proxy",
    "ir107_p10_cold_cloud_proxy",
    "ir107_p90_warm_background_proxy",
    "cold_fraction_global_q10",
    "cold_fraction_global_q25",
    "texture_gradient_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Preprocess SEVIR IR107 event sequences for GeminiNet.")
    parser.add_argument(
        "--raw_root",
        type=str,
        default="./datasets/SEVIR_IR107_192/raw",
        help="Folder for downloaded SEVIR catalog and HDF5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/SEVIR_IR107_192/processed",
        help="Folder used to save GeminiNet-ready arrays.",
    )
    parser.add_argument(
        "--catalog_url", type=str, default=SEVIR_CATALOG_URL, help="URL of SEVIR CATALOG.csv."
    )
    parser.add_argument(
        "--base_url", type=str, default=SEVIR_BASE_URL, help="Base URL for SEVIR HDF5 files."
    )
    parser.add_argument("--img_size", type=int, default=192, help="Output image height and width.")
    parser.add_argument(
        "--input_len",
        type=int,
        default=12,
        help="Recommended historical sequence length stored in metadata.",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=12,
        help="Recommended forecasting sequence length stored in metadata.",
    )
    parser.add_argument(
        "--max_raw_gb",
        type=float,
        default=0.0,
        help="Approximate maximum raw HDF5 download size. Use 0 to disable this cap.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Optional maximum number of IR107 HDF5 files. Use 0 to disable this cap.",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=0,
        help="Maximum number of IR107 events to process. Use 0 for all events from selected files.",
    )
    parser.add_argument(
        "--download_retries", type=int, default=3, help="Number of download retries for each file."
    )
    parser.add_argument(
        "--disable_ssl_verify",
        action="store_true",
        help="Disable SSL verification if local/network SSL issues interrupt public SEVIR access.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite output_dir if it already exists."
    )
    return parser.parse_args()


def _request_with_retries(url: str, retries: int, verify_ssl: bool, **kwargs) -> requests.Response:
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


def _download_file(url: str, path: Path, retries: int, verify_ssl: bool) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=180, verify=verify_ssl) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            tmp_path.replace(path)
            return path
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt >= retries:
                raise
            time.sleep(2 * attempt)

    return path


def _candidate_file_urls(base_url: str, file_name: str) -> list[str]:
    if file_name.startswith("http://") or file_name.startswith("https://"):
        return [file_name]

    cleaned = file_name.lstrip("/")
    urls = [f"{base_url.rstrip('/')}/{cleaned}"]
    if not cleaned.startswith("data/"):
        urls.append(f"{base_url.rstrip('/')}/data/{cleaned}")
    return urls


def _download_sevir_h5(
    file_name: str, raw_root: Path, base_url: str, retries: int, verify_ssl: bool
) -> Path:
    local_path = _local_h5_path(raw_root, file_name)
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    last_error = None
    for url in _candidate_file_urls(base_url=base_url, file_name=file_name):
        try:
            return _download_file(url=url, path=local_path, retries=retries, verify_ssl=verify_ssl)
        except Exception as error:
            last_error = error
            print(f"[download fallback] failed url={url} error={error}")
    raise last_error


def _download_catalog(raw_root: Path, catalog_url: str, retries: int, verify_ssl: bool) -> Path:
    catalog_path = raw_root / "CATALOG.csv"
    return _download_file(
        url=catalog_url,
        path=catalog_path,
        retries=retries,
        verify_ssl=verify_ssl,
    )


def _normalize_catalog_columns(catalog: pd.DataFrame) -> pd.DataFrame:
    catalog = catalog.copy()
    catalog.columns = [str(column).strip() for column in catalog.columns]
    lower_map = {column.lower(): column for column in catalog.columns}
    required = ["img_type", "file_name", "file_index"]
    missing = [name for name in required if name not in lower_map]
    if missing:
        raise ValueError(
            f"SEVIR catalog missing required columns: {missing}. Available columns: {list(catalog.columns)}"
        )
    catalog = catalog.rename(columns={lower_map[name]: name for name in required})
    return catalog


def _candidate_rows(catalog_path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path)
    catalog = _normalize_catalog_columns(catalog)
    rows = catalog[catalog["img_type"].astype(str).str.lower() == IMAGE_TYPE].copy()
    rows["file_index"] = rows["file_index"].astype(int)
    rows = rows.sort_values(["file_name", "file_index"]).reset_index(drop=True)
    if rows.empty:
        raise RuntimeError("No IR107 rows found in SEVIR catalog.")
    return rows


def _remote_file_size(url: str, retries: int, verify_ssl: bool) -> int | None:
    for attempt in range(1, retries + 1):
        try:
            response = requests.head(url, timeout=60, verify=verify_ssl, allow_redirects=True)
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
        except Exception:
            if attempt >= retries:
                return None
            time.sleep(2 * attempt)
    return None


def _remote_sevir_file_size(
    file_name: str, base_url: str, retries: int, verify_ssl: bool
) -> int | None:
    for url in _candidate_file_urls(base_url=base_url, file_name=file_name):
        size = _remote_file_size(url=url, retries=retries, verify_ssl=verify_ssl)
        if size is not None:
            return size
    return None


def _select_files(
    rows: pd.DataFrame,
    base_url: str,
    max_raw_gb: float,
    max_files: int,
    retries: int,
    verify_ssl: bool,
) -> tuple[list[str], dict[str, int | None]]:
    file_names = list(dict.fromkeys(rows["file_name"].astype(str).tolist()))
    selected: list[str] = []
    file_sizes: dict[str, int | None] = {}
    total_size = 0
    raw_cap = int(max_raw_gb * 1024**3) if max_raw_gb > 0 else 0

    for file_name in file_names:
        if max_files > 0 and len(selected) >= max_files:
            break
        size = _remote_sevir_file_size(
            file_name=file_name, base_url=base_url, retries=retries, verify_ssl=verify_ssl
        )
        if raw_cap > 0 and size is not None and selected and total_size + size > raw_cap:
            break
        selected.append(file_name)
        file_sizes[file_name] = size
        if size is not None:
            total_size += size

    if not selected:
        selected = [file_names[0]]
        file_sizes[selected[0]] = None
    return selected, file_sizes


def _local_h5_path(raw_root: Path, file_name: str) -> Path:
    return raw_root / file_name


def _download_selected_files(
    selected_files: list[str],
    raw_root: Path,
    base_url: str,
    retries: int,
    verify_ssl: bool,
) -> list[Path]:
    local_paths: list[Path] = []
    for index, file_name in enumerate(selected_files, start=1):
        print(f"[download] {index}/{len(selected_files)} {file_name}")
        local_paths.append(
            _download_sevir_h5(
                file_name=file_name,
                raw_root=raw_root,
                base_url=base_url,
                retries=retries,
                verify_ssl=verify_ssl,
            )
        )
    return local_paths


def _find_dataset(handle: h5py.File) -> h5py.Dataset:
    candidates: list[h5py.Dataset] = []

    def visitor(name: str, node):
        if isinstance(node, h5py.Dataset):
            lowered = name.lower()
            if IMAGE_TYPE in lowered:
                candidates.insert(0, node)
            else:
                candidates.append(node)

    handle.visititems(visitor)

    for dataset in candidates:
        if dataset.ndim >= 4:
            return dataset
    raise KeyError("No SEVIR IR107-like dataset with ndim >= 4 was found in HDF5 file.")


def _standardize_event_array(event: np.ndarray, img_size: int) -> np.ndarray:
    event = np.asarray(event)
    event = np.squeeze(event)

    if event.ndim != 3:
        raise ValueError(
            f"Expected one SEVIR event to be 3D after squeeze, got shape={event.shape}."
        )

    shape = event.shape
    time_axis = None
    for axis, size in enumerate(shape):
        if size == 49:
            time_axis = axis
            break
    if time_axis is None:
        time_axis = int(np.argmin(shape))

    event = np.moveaxis(event, time_axis, 0).astype(np.float32)
    event = _fill_nonfinite(event)

    if event.shape[1] != img_size or event.shape[2] != img_size:
        tensor = torch.from_numpy(event[:, None, :, :]).float()
        tensor = F.interpolate(
            tensor,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        event = tensor[:, 0].cpu().numpy().astype(np.float32)

    return event[..., None].astype(np.float32)


def _fill_nonfinite(array: np.ndarray) -> np.ndarray:
    valid = np.isfinite(array)
    if valid.all():
        return array.astype(np.float32)
    if not valid.any():
        raise ValueError("A SEVIR IR107 event contains no finite values.")
    fill_value = float(np.nanmean(array))
    return np.where(valid, array, fill_value).astype(np.float32)


def _read_events_from_file(path: Path, file_indices: list[int], img_size: int) -> list[np.ndarray]:
    events: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        dataset = _find_dataset(handle)
        for file_index in file_indices:
            event = np.asarray(dataset[file_index])
            events.append(_standardize_event_array(event, img_size=img_size))
    return events


def _numeric_features_from_frame(frame: np.ndarray, cold_q10: float, cold_q25: float) -> np.ndarray:
    frame = _fill_nonfinite(np.asarray(frame, dtype=np.float32))
    gy, gx = np.gradient(frame)
    gradient_mean = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    return np.asarray(
        [
            float(np.mean(frame)),
            float(np.std(frame)),
            float(np.min(frame)),
            float(np.percentile(frame, 10)),
            float(np.percentile(frame, 90)),
            float(np.mean(frame <= cold_q10)),
            float(np.mean(frame <= cold_q25)),
            gradient_mean,
        ],
        dtype=np.float32,
    )


def _build_numbers(images: np.ndarray) -> tuple[np.ndarray, dict]:
    values = images[..., 0]
    cold_q10 = float(np.percentile(values, 10))
    cold_q25 = float(np.percentile(values, 25))
    numbers = np.zeros((images.shape[0], images.shape[1], len(NUM_FEATURE_NAMES)), dtype=np.float32)

    for event_index in range(images.shape[0]):
        for time_index in range(images.shape[1]):
            numbers[event_index, time_index] = _numeric_features_from_frame(
                frame=values[event_index, time_index],
                cold_q10=cold_q10,
                cold_q25=cold_q25,
            )

    thresholds = {
        "global_ir107_q10": cold_q10,
        "global_ir107_q25": cold_q25,
    }
    return numbers, thresholds


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    verify_ssl = not args.disable_ssl_verify

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    print(f"SEVIR catalog URL: {args.catalog_url}")
    print(f"SEVIR base URL: {args.base_url}")
    print(f"Raw root: {raw_root}")
    print(f"Output dir: {output_dir}")
    print(f"Image type: {IMAGE_TYPE}")
    print(f"Output image size: {args.img_size} x {args.img_size}")
    print(f"Raw download cap: {args.max_raw_gb} GB")
    print(f"Max events: {args.max_events}")

    catalog_path = _download_catalog(
        raw_root=raw_root,
        catalog_url=args.catalog_url,
        retries=args.download_retries,
        verify_ssl=verify_ssl,
    )
    rows = _candidate_rows(catalog_path)
    selected_files, file_sizes = _select_files(
        rows=rows,
        base_url=args.base_url,
        max_raw_gb=args.max_raw_gb,
        max_files=args.max_files,
        retries=args.download_retries,
        verify_ssl=verify_ssl,
    )
    selected_rows = rows[rows["file_name"].isin(selected_files)].copy()
    if args.max_events > 0:
        selected_rows = selected_rows.head(args.max_events).copy()
    selected_files = list(dict.fromkeys(selected_rows["file_name"].astype(str).tolist()))

    _download_selected_files(
        selected_files=selected_files,
        raw_root=raw_root,
        base_url=args.base_url,
        retries=args.download_retries,
        verify_ssl=verify_ssl,
    )

    events: list[np.ndarray] = []
    event_records: list[dict] = []
    for file_counter, file_name in enumerate(selected_files, start=1):
        local_path = _local_h5_path(raw_root, file_name)
        file_rows = selected_rows[selected_rows["file_name"] == file_name]
        file_indices = file_rows["file_index"].astype(int).tolist()
        print(
            f"[process] file={file_counter}/{len(selected_files)} events={len(file_indices)} {file_name}"
        )
        file_events = _read_events_from_file(
            local_path, file_indices=file_indices, img_size=args.img_size
        )
        for row, event in zip(file_rows.to_dict(orient="records"), file_events):
            events.append(event)
            event_records.append(
                {
                    "file_name": str(file_name),
                    "file_index": int(row["file_index"]),
                    "id": str(row.get("id", "")),
                    "event_id": str(row.get("event_id", "")),
                    "time_utc": str(row.get("time_utc", "")),
                }
            )

    if not events:
        raise RuntimeError("No SEVIR IR107 events were successfully processed.")

    images = np.stack(events, axis=0).astype(np.float32)
    numbers, thresholds = _build_numbers(images)

    if len(images) != len(numbers):
        raise RuntimeError(
            f"Image and numerical event counts differ: {len(images)} vs {len(numbers)}."
        )
    if not np.isfinite(images).all():
        raise ValueError("images.npy contains non-finite values.")
    if not np.isfinite(numbers).all():
        raise ValueError("numbers.npy contains non-finite values.")

    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "numbers.npy", numbers)

    metadata = {
        "dataset": "SEVIR_IR107_192",
        "source": args.base_url,
        "catalog_url": args.catalog_url,
        "image_modality": "SEVIR IR107 infrared satellite image event sequences",
        "numerical_modality": "physically meaningful IR107 cloud-top and texture statistics derived from each image frame",
        "image_type": IMAGE_TYPE,
        "image_shape": list(images.shape[2:]),
        "array_shape_images": list(images.shape),
        "array_shape_numbers": list(numbers.shape),
        "num_events": int(images.shape[0]),
        "sequence_len": int(images.shape[1]),
        "num_feature_names": NUM_FEATURE_NAMES,
        "num_vars": len(NUM_FEATURE_NAMES),
        "thresholds": thresholds,
        "selected_files": selected_files,
        "file_sizes_bytes": file_sizes,
        "event_records": event_records,
        "task_recommendation": {
            "input_len": int(args.input_len),
            "pred_len": int(args.pred_len),
            "meaning": "past 12 SEVIR IR107 frames -> future 12 SEVIR IR107 frames",
        },
        "notes": [
            "Only SEVIR IR107 image sequences are used as the image modality.",
            "The numerical modality is derived from image-frame statistics when external station variables are not used.",
            "Cold-cloud fractions use dataset-level low-IR107 thresholds saved in metadata.",
            "Events are split by event index in the DataLoader to reduce train/validation/test leakage across dynamic windows.",
        ],
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("SEVIR IR107 preprocessing finished.")
    print(f"images: {images.shape} float32")
    print(f"numbers: {numbers.shape} float32")
    print(f"metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()