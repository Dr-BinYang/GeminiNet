from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESS_SCRIPT = SCRIPT_DIR / "preprocess.py"
CDS_DATASET = "reanalysis-era5-land"
CDS_HOME = "https://cds.climate.copernicus.eu/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download ERA5-Land data through the CDS API.")
    parser.add_argument("--raw_root", type=Path, default=SCRIPT_DIR / "raw")
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_preprocessor():
    spec = importlib.util.spec_from_file_location("era5_dataset_preprocess", PREPROCESS_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {PREPROCESS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start_year must not exceed end_year")
    module = load_preprocessor()
    print(f"CDS endpoint: {CDS_HOME}")
    print(f"CDS dataset: {CDS_DATASET}")
    module._download_monthly_files(
        raw_root=args.raw_root,
        start_year=args.start_year,
        end_year=args.end_year,
        force_download=args.force,
    )


if __name__ == "__main__":
    main()