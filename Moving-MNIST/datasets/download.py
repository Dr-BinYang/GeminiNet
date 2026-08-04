from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download the MNIST source images.")
    parser.add_argument("--output_dir", type=Path, default=Path("./datasets/MNIST/raw"))
    return parser.parse_args()


def main() -> None:
    try:
        from torchvision.datasets import MNIST
    except ImportError as error:
        raise ImportError("torchvision is required to download MNIST") from error

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    MNIST(root=str(args.output_dir), train=True, download=True)
    MNIST(root=str(args.output_dir), train=False, download=True)
    print(f"MNIST is available under {args.output_dir}")


if __name__ == "__main__":
    main()