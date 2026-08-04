from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_goes16_abi_c13.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download GOES-16 ABI Channel-13 CONUS files.")
    parser.add_argument(
        "--raw_root", type=Path, default=Path("./datasets/GOES16_ABI_C13_CONUS/raw")
    )
    parser.add_argument("--start_time", required=True, help="UTC start time in ISO-8601 format.")
    parser.add_argument("--end_time", required=True, help="UTC end time in ISO-8601 format.")
    parser.add_argument("--sample_minutes", type=int, default=10)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--download_retries", type=int, default=3)
    parser.add_argument("--disable_ssl_verify", action="store_true")
    return parser.parse_args()


def load_preprocessor():
    spec = importlib.util.spec_from_file_location("goes_preprocessor", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    module = load_preprocessor()
    start_time = module._parse_utc_time(args.start_time)
    end_time = module._parse_utc_time(args.end_time)
    if start_time > end_time:
        raise ValueError("start_time must not exceed end_time")
    records = module._collect_candidate_files(
        start_time=start_time,
        end_time=end_time,
        retries=args.download_retries,
        verify_ssl=not args.disable_ssl_verify,
    )
    records = module._sample_records(
        records=records,
        start_time=start_time,
        sample_minutes=args.sample_minutes,
        max_files=args.max_files,
    )
    for index, (_, key) in enumerate(records, start=1):
        print(f"[download] {index}/{len(records)} {key}")
        module._download_file(
            key=key,
            raw_root=args.raw_root,
            retries=args.download_retries,
            verify_ssl=not args.disable_ssl_verify,
        )


if __name__ == "__main__":
    main()