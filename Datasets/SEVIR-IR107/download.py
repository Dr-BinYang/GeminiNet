from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESS_SCRIPT = SCRIPT_DIR / "preprocess.py"
SEVIR_BASE_URL = "https://sevir.s3.amazonaws.com"
SEVIR_CATALOG_URL = f"{SEVIR_BASE_URL}/CATALOG.csv"
IMAGE_TYPE = "ir107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download the SEVIR IR107 catalog and HDF5 files.")
    parser.add_argument("--raw_root", type=Path, default=SCRIPT_DIR / "raw")
    parser.add_argument("--max_raw_gb", type=float, default=0.0)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--download_retries", type=int, default=3)
    parser.add_argument("--disable_ssl_verify", action="store_true")
    return parser.parse_args()


def load_preprocessor():
    spec = importlib.util.spec_from_file_location("sevir_dataset_preprocess", PREPROCESS_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {PREPROCESS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    module = load_preprocessor()
    verify_ssl = not args.disable_ssl_verify
    print(f"S3 endpoint: {SEVIR_BASE_URL}")
    print(f"Catalog: {SEVIR_CATALOG_URL}")
    print(f"Image type: {IMAGE_TYPE}")
    catalog = module._download_catalog(
        raw_root=args.raw_root,
        catalog_url=SEVIR_CATALOG_URL,
        retries=args.download_retries,
        verify_ssl=verify_ssl,
    )
    rows = module._candidate_rows(catalog)
    selected_files, _ = module._select_files(
        rows=rows,
        base_url=SEVIR_BASE_URL,
        max_raw_gb=args.max_raw_gb,
        max_files=args.max_files,
        retries=args.download_retries,
        verify_ssl=verify_ssl,
    )
    module._download_selected_files(
        selected_files=selected_files,
        raw_root=args.raw_root,
        base_url=SEVIR_BASE_URL,
        retries=args.download_retries,
        verify_ssl=verify_ssl,
    )


if __name__ == "__main__":
    main()