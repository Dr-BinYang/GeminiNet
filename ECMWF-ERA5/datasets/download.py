from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_era5_land_chengyu.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download ERA5-Land data from the Copernicus Data Store.")
    parser.add_argument("--raw_root", type=Path, default=Path("./datasets/ERA5_Land_ChengYu/raw"))
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_preprocessor():
    spec = importlib.util.spec_from_file_location("era5_preprocessor", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start_year must not exceed end_year")
    module = load_preprocessor()
    module._download_monthly_files(
        raw_root=args.raw_root,
        start_year=args.start_year,
        end_year=args.end_year,
        force_download=args.force,
    )


if __name__ == "__main__":
    main()