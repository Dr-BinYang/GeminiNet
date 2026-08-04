from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

import requests

BASE_URL = "https://www.csc.kth.se/cvap/actions"
ACTIONS = ("walking", "jogging", "running", "boxing", "handwaving", "handclapping")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download the KTH Action dataset.")
    parser.add_argument("--output_dir", type=Path, default=SCRIPT_DIR / "raw")
    parser.add_argument("--no_extract", action="store_true")
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(temporary, "wb") as file:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    file.write(chunk)
    temporary.replace(destination)


def extract(archive: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (output_dir / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        handle.extractall(output_dir)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for action in ACTIONS:
        archive = args.output_dir / f"{action}.zip"
        download(f"{BASE_URL}/{action}.zip", archive)
        if not args.no_extract:
            extract(archive, args.output_dir)
    print(f"KTH Action data is available under {args.output_dir}")


if __name__ == "__main__":
    main()
