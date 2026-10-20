"""Exclusive, collision-safe publication of completed recording files."""

from __future__ import annotations

import errno
import os
from pathlib import Path

_MAX_FILENAME_BYTES = 240


def _utf8_prefix(value: str, limit: int) -> str:
    """Truncate text without splitting any UTF-8 character."""
    result: list[str] = []
    remaining = limit

    for char in value:
        size = len(char.encode("utf-8"))
        if size > remaining:
            break
        result.append(char)
        remaining -= size

    return "".join(result)


def publish_unique(
    *,
    temp_path: Path,
    directory: Path,
    stem: str,
    extension: str,
) -> Path:
    """Publish a completed same-directory temporary file without overwriting."""
    if not stem or not stem.strip():
        raise ValueError("stem must not be empty")
    if stem in {".", ".."} or any(char in stem for char in "/\\\x00"):
        raise ValueError("unsafe output stem")
    if extension not in {".mp3", ".aac", ".m4a"}:
        raise ValueError("unsupported output extension")
    if temp_path.parent.resolve() != directory.resolve():
        raise ValueError("temporary file must be in the destination directory")

    index = 1

    while True:
        suffix = "" if index == 1 else f"-{index}"
        available = _MAX_FILENAME_BYTES - len((suffix + extension).encode("utf-8"))
        if available <= 0:
            raise ValueError("collision suffix exceeds filename budget")

        bounded_stem = _utf8_prefix(stem, available).rstrip(" .")
        if not bounded_stem:
            bounded_stem = "track"

        target = directory / f"{bounded_stem}{suffix}{extension}"

        try:
            os.link(temp_path, target)
        except FileExistsError:
            index += 1
            continue
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                index += 1
                continue
            raise

        temp_path.unlink()
        return target
