from __future__ import annotations

import argparse
import calendar
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CDS_DATASET = "reanalysis-era5-land"
ZIP_NETCDF_EXTRACT_DIR = "extracted_zip_netcdf"

REGION = {
    "name": "Chengdu-Chongqing urban agglomeration / Sichuan Basin",
    "north": 35.0,
    "south": 25.0,
    "west": 100.0,
    "east": 110.0,
}

CDS_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "volumetric_soil_water_layer_1",
    "soil_temperature_level_1",
    "skin_temperature",
]

VARIABLE_ALIASES = {
    "2m_temperature": ["t2m", "2m_temperature"],
    "2m_dewpoint_temperature": ["d2m", "2m_dewpoint_temperature"],
    "10m_u_component_of_wind": ["u10", "10m_u_component_of_wind"],
    "10m_v_component_of_wind": ["v10", "10m_v_component_of_wind"],
    "surface_pressure": ["sp", "surface_pressure"],
    "volumetric_soil_water_layer_1": ["swvl1", "volumetric_soil_water_layer_1"],
    "soil_temperature_level_1": ["stl1", "soil_temperature_level_1"],
    "skin_temperature": ["skt", "skin_temperature"],
}

NUM_FEATURE_NAMES = [
    "t2m_mean_c",
    "d2m_mean_c",
    "u10_mean_ms",
    "v10_mean_ms",
    "wind10_mean_ms",
    "sp_mean_hpa",
    "swvl1_mean_m3m3",
    "stl1_mean_c",
    "skt_mean_c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Download/preprocess ERA5-Land Chengdu-Chongqing regional data for GeminiNet."
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default="./datasets/ERA5_Land_ChengYu/raw",
        help="Folder for downloaded monthly ERA5-Land NetCDF files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/ERA5_Land_ChengYu/processed",
        help="Folder used to save GeminiNet-ready arrays.",
    )
    parser.add_argument(
        "--start_year",
        type=int,
        default=None,
        help="First year to download. Required with --download.",
    )
    parser.add_argument(
        "--end_year",
        type=int,
        default=None,
        help="Last year to download. Required with --download.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download monthly ERA5-Land NetCDF files through CDS API before preprocessing.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Redownload monthly files even if they already exist.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite output_dir if it already exists."
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Maximum number of raw files to process. Use 0 for all files.",
    )
    args = parser.parse_args()
    if args.download and (args.start_year is None or args.end_year is None):
        parser.error("--start_year and --end_year are required with --download")
    return args


def _month_days(year: int, month: int) -> list[str]:
    days = calendar.monthrange(year, month)[1]
    return [f"{day:02d}" for day in range(1, days + 1)]


def _hours() -> list[str]:
    return [f"{hour:02d}:00" for hour in range(24)]


def _download_monthly_files(
    raw_root: Path, start_year: int, end_year: int, force_download: bool
) -> None:
    try:
        import cdsapi
    except ImportError as error:
        raise ImportError(
            "cdsapi is required only for downloading ERA5-Land data. "
            "Install it with `pip install cdsapi` and configure your CDS API key first."
        ) from error

    raw_root.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            target = raw_root / f"era5_land_chengyu_{year}_{month:02d}.nc"
            if target.exists() and not force_download:
                print(f"[download skip] {target}")
                continue

            request = {
                "variable": CDS_VARIABLES,
                "year": f"{year}",
                "month": f"{month:02d}",
                "day": _month_days(year, month),
                "time": _hours(),
                "area": [
                    REGION["north"],
                    REGION["west"],
                    REGION["south"],
                    REGION["east"],
                ],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }

            print(f"[download] {year}-{month:02d} -> {target}")
            try:
                client.retrieve(CDS_DATASET, request, str(target))
            except Exception:
                legacy_request = dict(request)
                legacy_request.pop("data_format", None)
                legacy_request.pop("download_format", None)
                legacy_request["format"] = "netcdf"
                client.retrieve(CDS_DATASET, legacy_request, str(target))


def _extract_zip_files(raw_root: Path) -> None:
    for zip_path in raw_root.rglob("*.zip"):
        extract_dir = raw_root / "extracted" / zip_path.stem
        if extract_dir.exists():
            continue
        print(f"[extract] {zip_path} -> {extract_dir}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)


def _raw_files(raw_root: Path, max_files: int = 0) -> list[Path]:
    _extract_zip_files(raw_root)
    candidates: list[Path] = []
    for pattern in ["*.nc", "*.nc4", "*.netcdf"]:
        candidates.extend(
            path for path in raw_root.rglob(pattern) if ZIP_NETCDF_EXTRACT_DIR not in path.parts
        )
    files = sorted(set(candidates))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(
            f"No ERA5-Land NetCDF files found under {raw_root}. "
            "Run with --download or place downloaded .nc files under raw_root."
        )
    return files


def _materialize_zip_wrapped_netcdf(path: Path) -> Path:
    """Extract a CDS ZIP response that was saved with a .nc suffix."""
    extract_dir = path.parent / ZIP_NETCDF_EXTRACT_DIR / path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path, "r") as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        netcdf_members = [
            info
            for info in members
            if Path(info.filename).suffix.lower() in [".nc", ".nc4", ".netcdf"]
        ]
        if not netcdf_members:
            raise RuntimeError(f"ZIP-wrapped CDS response contains no NetCDF file: {path}")

        member = netcdf_members[0]
        target = extract_dir / Path(member.filename).name
        if not target.exists() or target.stat().st_size != member.file_size:
            print(f"[extract wrapped nc] {path.name} -> {target}")
            with archive.open(member, "r") as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)

    return target


def _open_dataset(path: Path) -> xr.Dataset:
    if zipfile.is_zipfile(path):
        path = _materialize_zip_wrapped_netcdf(path)

    errors: list[str] = []
    for engine in [None, "h5netcdf", "scipy"]:
        try:
            kwargs = {} if engine is None else {"engine": engine}
            return xr.open_dataset(path, **kwargs)
        except Exception as error:
            errors.append(f"{engine or 'default'}: {error}")
    raise RuntimeError(f"Cannot open {path}. Tried engines: {errors}")


def _coord_name(ds: xr.Dataset, candidates: list[str]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"None of coordinate names {candidates} found in dataset.")


def _data_var(ds: xr.Dataset, canonical_name: str) -> xr.DataArray:
    for name in VARIABLE_ALIASES[canonical_name]:
        if name in ds.data_vars:
            return ds[name]
    raise KeyError(
        f"Variable {canonical_name} not found. Available variables: {list(ds.data_vars)}"
    )


def _var_values(
    ds: xr.Dataset, canonical_name: str, time_name: str, lat_name: str, lon_name: str
) -> np.ndarray:
    data_array = _data_var(ds, canonical_name).squeeze()
    data_array = data_array.transpose(time_name, lat_name, lon_name)
    return data_array.load().values.astype(np.float32)


def _to_celsius(array: np.ndarray) -> np.ndarray:
    return array.astype(np.float32) - 273.15


def _process_one_file(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with _open_dataset(path) as ds:
        time_name = _coord_name(ds, ["valid_time", "time"])
        lat_name = _coord_name(ds, ["latitude", "lat"])
        lon_name = _coord_name(ds, ["longitude", "lon"])

        ds = ds.sortby(time_name)
        if float(ds[lat_name][0]) < float(ds[lat_name][-1]):
            ds = ds.sortby(lat_name, ascending=False)
        ds = ds.sortby(lon_name)

        t2m = _var_values(ds, "2m_temperature", time_name, lat_name, lon_name)
        d2m = _var_values(ds, "2m_dewpoint_temperature", time_name, lat_name, lon_name)
        u10 = _var_values(ds, "10m_u_component_of_wind", time_name, lat_name, lon_name)
        v10 = _var_values(ds, "10m_v_component_of_wind", time_name, lat_name, lon_name)
        sp = _var_values(ds, "surface_pressure", time_name, lat_name, lon_name)
        swvl1 = _var_values(ds, "volumetric_soil_water_layer_1", time_name, lat_name, lon_name)
        stl1 = _var_values(ds, "soil_temperature_level_1", time_name, lat_name, lon_name)
        skt = _var_values(ds, "skin_temperature", time_name, lat_name, lon_name)

        t2m_c = _to_celsius(t2m)
        d2m_c = _to_celsius(d2m)
        stl1_c = _to_celsius(stl1)
        skt_c = _to_celsius(skt)
        wind10 = np.sqrt(u10**2 + v10**2).astype(np.float32)
        sp_hpa = sp / 100.0

        image = t2m_c[..., None].astype(np.float32)
        numbers = np.stack(
            [
                np.nanmean(t2m_c, axis=(1, 2)),
                np.nanmean(d2m_c, axis=(1, 2)),
                np.nanmean(u10, axis=(1, 2)),
                np.nanmean(v10, axis=(1, 2)),
                np.nanmean(wind10, axis=(1, 2)),
                np.nanmean(sp_hpa, axis=(1, 2)),
                np.nanmean(swvl1, axis=(1, 2)),
                np.nanmean(stl1_c, axis=(1, 2)),
                np.nanmean(skt_c, axis=(1, 2)),
            ],
            axis=-1,
        ).astype(np.float32)

        times = pd.to_datetime(ds[time_name].values).astype(str).to_numpy()
        latitudes = ds[lat_name].values.astype(np.float32)
        longitudes = ds[lon_name].values.astype(np.float32)

    return image, numbers, times, latitudes, longitudes


def _sort_by_timestamp(
    images: np.ndarray,
    numbers: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(timestamps)
    images = images[order]
    numbers = numbers[order]
    timestamps = timestamps[order]
    _, unique_indices = np.unique(timestamps, return_index=True)
    unique_indices = np.sort(unique_indices)
    return images[unique_indices], numbers[unique_indices], timestamps[unique_indices]


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)

    raw_root.mkdir(parents=True, exist_ok=True)
    if args.download:
        _download_monthly_files(
            raw_root=raw_root,
            start_year=args.start_year,
            end_year=args.end_year,
            force_download=args.force_download,
        )

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _raw_files(raw_root, max_files=args.max_files)
    print(f"Raw root: {raw_root}")
    print(f"Output dir: {output_dir}")
    print(f"NetCDF files: {len(files)}")

    image_parts: list[np.ndarray] = []
    number_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    latitudes = None
    longitudes = None

    for index, path in enumerate(files, start=1):
        image, numbers, timestamps, lat, lon = _process_one_file(path)
        image_parts.append(image)
        number_parts.append(numbers)
        timestamp_parts.append(timestamps)
        if latitudes is None:
            latitudes = lat
            longitudes = lon
        print(
            f"[process] {index}/{len(files)} {path.name} hours={len(timestamps)} image={image.shape[1:]} numbers={numbers.shape[-1]}"
        )

    images = np.concatenate(image_parts, axis=0).astype(np.float32)
    numbers = np.concatenate(number_parts, axis=0).astype(np.float32)
    timestamps = np.concatenate(timestamp_parts, axis=0)
    images, numbers, timestamps = _sort_by_timestamp(images, numbers, timestamps)

    if not np.isfinite(images).all():
        raise ValueError("images.npy contains non-finite values.")
    if not np.isfinite(numbers).all():
        raise ValueError("numbers.npy contains non-finite values.")
    if latitudes is None or longitudes is None:
        raise RuntimeError("Latitude/longitude coordinates were not resolved.")

    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "numbers.npy", numbers)
    np.save(output_dir / "timestamps.npy", timestamps.astype("datetime64[ns]"))
    np.save(output_dir / "latitudes.npy", latitudes.astype(np.float32))
    np.save(output_dir / "longitudes.npy", longitudes.astype(np.float32))

    metadata = {
        "dataset": "ERA5_Land_ChengYu",
        "source_dataset": CDS_DATASET,
        "image_modality": "Regional hourly ERA5-Land 2m temperature spatial field in Celsius",
        "numerical_modality": "Regional-mean multivariate ERA5-Land meteorological and land-surface state",
        "region": REGION,
        "time_range": {
            "start": str(timestamps[0]),
            "end": str(timestamps[-1]),
            "hours": int(len(timestamps)),
        },
        "spatial_grid": {
            "latitude_count": int(len(latitudes)),
            "longitude_count": int(len(longitudes)),
            "image_shape": list(images.shape[1:]),
            "latitude_min": float(np.min(latitudes)),
            "latitude_max": float(np.max(latitudes)),
            "longitude_min": float(np.min(longitudes)),
            "longitude_max": float(np.max(longitudes)),
            "native_resolution_degree": 0.1,
        },
        "array_shape_images": list(images.shape),
        "array_shape_numbers": list(numbers.shape),
        "image_dtype": str(images.dtype),
        "number_dtype": str(numbers.dtype),
        "image_variable": "t2m_c",
        "num_feature_names": NUM_FEATURE_NAMES,
        "num_vars": len(NUM_FEATURE_NAMES),
        "cds_variables": CDS_VARIABLES,
        "raw_root": str(raw_root),
        "raw_files": [str(path) for path in files],
        "task_recommendation": {
            "input_len": 12,
            "pred_len": 12,
            "meaning": "past 12 hourly temperature fields and regional-mean states -> future 12 hourly fields and states",
        },
        "notes": [
            "The image modality keeps the native ERA5-Land 0.1 degree grid without interpolation.",
            "The numerical modality is computed by spatially averaging variables over the same region and timestamp as the image field.",
            "Temperatures are converted from Kelvin to Celsius; surface pressure is converted from Pa to hPa.",
            "The selected region is much smaller than East Asia, so regional averaging is meteorologically more meaningful.",
        ],
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("ERA5-Land ChengYu preprocessing finished.")
    print(f"images: {images.shape} {images.dtype}")
    print(f"numbers: {numbers.shape} {numbers.dtype}")
    print(f"timestamps: {timestamps[0]} -> {timestamps[-1]}")
    print(f"metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()