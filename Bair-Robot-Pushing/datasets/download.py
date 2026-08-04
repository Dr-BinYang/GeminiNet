from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import requests

DATASET_URL = "http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download the BAIR Robot Pushing dataset.")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./datasets/BAIR_Robot_Pushing_Small/raw")
    )
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--no_extract", action="store_true")
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, stream=True, timeout=120, headers=headers) as response:
        response.raise_for_status()
        mode = "ab" if offset and response.status_code == 206 else "wb"
        with open(destination, mode) as file:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    file.write(chunk)


def extract(archive: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (output_dir / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
        handle.extractall(output_dir)


def main() -> None:
    args = parse_args()
    archive = args.output_dir / "bair_robot_pushing_dataset_v0.tar"
    download(args.url, archive)
    if not args.no_extract:
        extract(archive, args.output_dir)
    print(f"BAIR data is available under {args.output_dir}")


if __name__ == "__main__":
    main()