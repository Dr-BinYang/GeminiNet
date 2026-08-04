from __future__ import annotations

import argparse
import re
import tarfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
from urllib.parse import urljoin

import requests

BASE_URL = "https://meteonet.umr-cnrm.fr/dataset/data/NW"
ARCHIVE_INDEXES = ("ground_stations/", "radar/rainfall/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download the MeteoNet north-west dataset.")
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
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    file.write(chunk)
    temporary.replace(destination)


def archive_urls() -> list[str]:
    urls: list[str] = []
    for relative_index in ARCHIVE_INDEXES:
        index_url = f"{BASE_URL}/{relative_index}"
        response = requests.get(index_url, timeout=60)
        response.raise_for_status()
        names = re.findall(r'href=["\']([^"\']+\.tar\.gz)["\']', response.text)
        urls.extend(urljoin(index_url, name) for name in names)
    if not urls:
        raise RuntimeError("No MeteoNet NW archives were found in the official index")
    return sorted(set(urls))


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for url in archive_urls():
        archive = args.output_dir / Path(url).name
        download(url, archive)
        if not args.no_extract:
            extract(archive, args.output_dir)
    print(f"MeteoNet data is available under {args.output_dir}")


if __name__ == "__main__":
    main()
