from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


def download_file(url: str, output_path: Path, timeout: int = 120) -> Path:
    """Download a file from a URL to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def maybe_download_sources(
    raw_dir: Path,
    listings_url: str | None = None,
    calendar_url: str | None = None,
) -> tuple[Path, Path]:
    """Download listings/calendar data when URLs are supplied."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    listings_path = raw_dir / "listings.csv.gz"
    calendar_path = raw_dir / "calendar.csv.gz"

    if listings_url:
        download_file(listings_url, listings_path)
    if calendar_url:
        download_file(calendar_url, calendar_path)

    # Fallback to plain csv if already present
    if not listings_path.exists():
        listings_path = raw_dir / "listings.csv"
    if not calendar_path.exists():
        calendar_path = raw_dir / "calendar.csv"

    if not listings_path.exists() or not calendar_path.exists():
        raise FileNotFoundError(
            "Missing source data. Provide --listings-url and --calendar-url or place "
            "listings.csv(.gz) and calendar.csv(.gz) into data/raw/."
        )

    return listings_path, calendar_path


def load_raw_data(listings_path: Path, calendar_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw listings and calendar files from CSV or CSV.GZ."""
    listings = pd.read_csv(listings_path, low_memory=False)
    calendar = pd.read_csv(calendar_path, low_memory=False)
    return listings, calendar
