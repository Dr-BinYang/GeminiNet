from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests

DEFAULT_PAGE_URLS = [
    "https://personal.utdallas.edu/~kehtar/UTD-MHAD.html",
    "http://www.utdallas.edu/~kehtar/UTD-MHAD.html",
]

FALLBACK_URLS = [
    "https://personal.utdallas.edu/~kehtar/UTD-MHAD.zip",
    "http://www.utdallas.edu/~kehtar/UTD-MHAD.zip",
    "https://personal.utdallas.edu/~kehtar/UTD-MHAD/Depth.zip",
    "https://personal.utdallas.edu/~kehtar/UTD-MHAD/Inertial.zip",
    "https://personal.utdallas.edu/~kehtar/UTD-MHAD/Skeleton.zip",
    "http://www.utdallas.edu/~kehtar/UTD-MHAD/Depth.zip",
    "http://www.utdallas.edu/~kehtar/UTD-MHAD/Inertial.zip",
    "http://www.utdallas.edu/~kehtar/UTD-MHAD/Skeleton.zip",
]

ARCHIVE_EXTENSIONS = [".zip", ".rar", ".7z", ".tar", ".tar.gz", ".tgz"]
MODALITY_KEYWORDS = [
    "depth",
    "inertial",
    "inertia",
    "imu",
    "skeleton",
    "joint",
    "utd-mhad",
    "utd_mhad",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download UTD-MHAD files needed by GeminiNet.")
    parser.add_argument(
        "--raw_root",
        type=str,
        default="./datasets/UTD_MHAD_Depth_Inertial/raw",
        help="Folder used to save downloaded UTD-MHAD archives.",
    )
    parser.add_argument(
        "--page_url",
        action="append",
        default=[],
        help="Dataset webpage URL to scrape for download links. Can be used multiple times.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Direct archive URL to download. Can be used multiple times.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="depth,inertial,skeleton",
        help="Comma-separated modality keywords to keep from scraped links.",
    )
    parser.add_argument(
        "--max_total_gb",
        type=float,
        default=0.0,
        help="Maximum total downloaded size. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--download_retries", type=int, default=3, help="Number of download retries per file."
    )
    parser.add_argument(
        "--disable_ssl_verify",
        action="store_true",
        help="Disable SSL verification if local/network SSL issues interrupt access.",
    )
    parser.add_argument(
        "--no_proxy",
        action="store_true",
        help="Ignore system/environment proxy settings when downloading.",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default="",
        help="Explicit proxy URL, for example http://127.0.0.1:7890 or socks5h://127.0.0.1:7890.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download files even if local files already exist."
    )
    return parser.parse_args()


def _build_session(no_proxy: bool, proxy: str = "") -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            )
        }
    )
    if proxy:
        session.proxies.update(
            {
                "http": proxy,
                "https": proxy,
            }
        )
    if no_proxy:
        session.trust_env = False
    return session


def _request_with_retries(
    session: requests.Session, url: str, retries: int, verify_ssl: bool, **kwargs
) -> requests.Response:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, verify=verify_ssl, **kwargs)
            response.raise_for_status()
            return response
        except Exception as error:
            last_error = error
            if attempt >= retries:
                raise
            time.sleep(2 * attempt)
    raise last_error


def _head_size(session: requests.Session, url: str, retries: int, verify_ssl: bool) -> int | None:
    for attempt in range(1, retries + 1):
        try:
            response = session.head(url, timeout=60, verify=verify_ssl, allow_redirects=True)
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
        except Exception:
            if attempt >= retries:
                return None
            time.sleep(2 * attempt)
    return None


def _extract_links(page_url: str, html: str) -> list[str]:
    links: list[str] = []
    patterns = [
        r'href=["\']([^"\']+)["\']',
        r'["\'](https?://[^"\']+)["\']',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, html, flags=re.IGNORECASE):
            href = match.group(1).strip()
            if not href or href.startswith("#"):
                continue
            links.append(urljoin(page_url, href))
    return sorted(set(links))


def _looks_like_archive(url: str) -> bool:
    lowered = unquote(urlparse(url).path).lower()
    return any(lowered.endswith(extension) for extension in ARCHIVE_EXTENSIONS)


def _matches_modalities(url: str, modalities: list[str]) -> bool:
    lowered = unquote(url).lower()
    if not modalities:
        return True
    return any(keyword in lowered for keyword in modalities)


def _scrape_candidate_urls(
    session: requests.Session,
    page_urls: list[str],
    retries: int,
    verify_ssl: bool,
    modalities: list[str],
) -> list[str]:
    candidates: list[str] = []
    for page_url in page_urls:
        try:
            print(f"[page] {page_url}")
            response = _request_with_retries(
                session, page_url, retries=retries, verify_ssl=verify_ssl, timeout=60
            )
        except Exception as error:
            print(f"[page skip] {page_url}: {error}")
            continue
        links = _extract_links(page_url=page_url, html=response.text)
        for link in links:
            if _looks_like_archive(link) and _matches_modalities(link, modalities):
                candidates.append(link)
    return sorted(set(candidates))


def _filename_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name
    if not name:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", url)
        name = f"{safe}.download"
    return name


def _download_file(
    session: requests.Session,
    url: str,
    output_path: Path,
    retries: int,
    verify_ssl: bool,
    force: bool,
) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        print(f"[exists] {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    for attempt in range(1, retries + 1):
        try:
            with session.get(
                url, stream=True, timeout=180, verify=verify_ssl, allow_redirects=True
            ) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            tmp_path.replace(output_path)
            return output_path
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt >= retries:
                raise
            time.sleep(2 * attempt)

    return output_path


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    verify_ssl = not args.disable_ssl_verify
    session = _build_session(no_proxy=args.no_proxy, proxy=args.proxy)
    modalities = [item.strip().lower() for item in args.modalities.split(",") if item.strip()]

    page_urls = args.page_url if args.page_url else DEFAULT_PAGE_URLS
    candidate_urls = list(args.url)
    candidate_urls.extend(
        _scrape_candidate_urls(
            session=session,
            page_urls=page_urls,
            retries=args.download_retries,
            verify_ssl=verify_ssl,
            modalities=modalities,
        )
    )

    if not candidate_urls:
        print("[warn] No archive links found from pages. Trying built-in fallback URL guesses.")
        candidate_urls.extend(
            [url for url in FALLBACK_URLS if _matches_modalities(url, modalities)]
        )

    candidate_urls = sorted(set(candidate_urls))
    if not candidate_urls:
        raise RuntimeError(
            "No UTD-MHAD download URL was found. " "Use --url to provide a direct archive link."
        )

    max_total_bytes = int(args.max_total_gb * 1024**3) if args.max_total_gb > 0 else 0
    downloaded_bytes = 0
    downloaded_files: list[Path] = []
    failed_urls: list[tuple[str, str]] = []

    print("=" * 80)
    print("UTD-MHAD downloader")
    print(f"Raw root: {raw_root}")
    print(f"Candidate URLs: {len(candidate_urls)}")
    print(f"Modalities: {modalities}")
    print(f"Max total GB: {args.max_total_gb}")
    print(f"No proxy: {args.no_proxy}")
    print(f"Explicit proxy: {args.proxy if args.proxy else '(not set)'}")
    print("=" * 80)

    for index, url in enumerate(candidate_urls, start=1):
        try:
            size = _head_size(
                session=session, url=url, retries=args.download_retries, verify_ssl=verify_ssl
            )
            if (
                max_total_bytes > 0
                and size is not None
                and downloaded_bytes + size > max_total_bytes
                and downloaded_files
            ):
                print(f"[skip cap] {url} size={size} would exceed max_total_gb={args.max_total_gb}")
                continue

            filename = _filename_from_url(url)
            output_path = raw_root / filename
            print(f"[download] {index}/{len(candidate_urls)} {url}")
            path = _download_file(
                session=session,
                url=url,
                output_path=output_path,
                retries=args.download_retries,
                verify_ssl=verify_ssl,
                force=args.force,
            )
            downloaded_files.append(path)
            if size is not None:
                downloaded_bytes += size
            elif path.exists():
                downloaded_bytes += path.stat().st_size
        except Exception as error:
            failed_urls.append((url, repr(error)))
            print(f"[failed] {url}: {error}")

    print("=" * 80)
    print("Download finished.")
    print(f"Downloaded files: {len(downloaded_files)}")
    for path in downloaded_files:
        print(f"  - {path} ({path.stat().st_size / 1024**2:.2f} MB)")
    if failed_urls:
        print("Failed URLs:")
        for url, error in failed_urls:
            print(f"  - {url}: {error}")
    print("=" * 80)

    if not downloaded_files:
        raise RuntimeError(
            "No UTD-MHAD archive was downloaded. "
            "The official page may require manual agreement or may have changed. "
            "If you have a direct URL, rerun with: "
            "python scripts/download_utd_mhad.py --url YOUR_DIRECT_ARCHIVE_URL"
        )

    print("Next step:")
    print("python scripts/preprocess_utd_mhad.py --force")


if __name__ == "__main__":
    main()