"""MovieLens download (checksum-verified) and rating parsing.

Data is subject to the GroupLens usage license and is never committed to git.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DOWNLOAD_TIMEOUT_SECONDS = 120
_HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class MovieLensVariant:
    url: str
    sha256: str  # computed from a download matching the official MD5
    ratings_member: str
    separator: bytes


MOVIELENS_VARIANTS: dict[str, MovieLensVariant] = {
    "ml-1m": MovieLensVariant(
        url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        sha256="a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
        ratings_member="ml-1m/ratings.dat",
        separator=b"::",
    ),
}


def _get_variant(variant: str) -> MovieLensVariant:
    if variant not in MOVIELENS_VARIANTS:
        raise ValueError(f"Unknown MovieLens variant {variant!r}; available: {sorted(MOVIELENS_VARIANTS)}")
    return MOVIELENS_VARIANTS[variant]


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_movielens(variant: str, raw_dir: Path, url: str | None = None) -> Path:
    """Download the variant's zip into `raw_dir` (skipped if a verified copy exists).

    `url` overrides the official URL (e.g. a mirror); the checksum is still enforced.
    """
    spec = _get_variant(variant)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{variant}.zip"
    if zip_path.exists() and sha256_of_file(zip_path) == spec.sha256:
        return zip_path

    partial_path = zip_path.with_suffix(".zip.part")
    with urllib.request.urlopen(url or spec.url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        with open(partial_path, "wb") as f:
            shutil.copyfileobj(response, f)
    actual = sha256_of_file(partial_path)
    if actual != spec.sha256:
        partial_path.unlink()
        raise ValueError(f"Checksum mismatch for {variant}: expected {spec.sha256}, got {actual}")
    partial_path.replace(zip_path)
    return zip_path


def read_movielens_ratings(zip_path: Path, variant: str) -> pd.DataFrame:
    """Return columns user_id, item_id, rating, timestamp in file order."""
    spec = _get_variant(variant)
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read(spec.ratings_member)
    # Multi-char separators force pandas' slow python engine; use a tab instead.
    return pd.read_csv(
        io.BytesIO(raw.replace(spec.separator, b"\t")),
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": "int64", "item_id": "int64", "rating": "float64", "timestamp": "int64"},
    )
