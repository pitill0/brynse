"""Conservative metadata ordering for radio recording filenames.

This module changes filename labels only. Original ICY metadata remains
unchanged throughout the ripping pipeline.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

MetadataOrder = Literal["auto", "preserve", "artist-title", "title-artist"]
ResolvedMetadataOrder = Literal["preserve", "title-artist"]

_JAZZ_HOST = "jazz-wr01.ice.infomaniak.ch"
_JAZZ_PATH = "/jazz-wr01-128.mp3"


def resolve_metadata_order(
    url: str,
    order: MetadataOrder = "auto",
) -> ResolvedMetadataOrder:
    """Resolve an explicit order or a narrowly verified stream profile."""
    if order not in {"auto", "preserve", "artist-title", "title-artist"}:
        raise ValueError(f"unsupported metadata order: {order!r}")

    if order in {"preserve", "artist-title"}:
        return "preserve"
    if order == "title-artist":
        return "title-artist"

    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            return "preserve"
        if parsed.username is not None or parsed.password is not None:
            return "preserve"
        if parsed.hostname != _JAZZ_HOST:
            return "preserve"
        if parsed.path != _JAZZ_PATH or parsed.query or parsed.fragment:
            return "preserve"

        default_port = 80 if parsed.scheme == "http" else 443
        if parsed.port not in {None, default_port}:
            return "preserve"
    except ValueError:
        return "preserve"

    return "title-artist"


def filename_title(
    title: str,
    order: ResolvedMetadataOrder,
) -> str:
    """Return a filename label, leaving ambiguous metadata untouched."""
    if order == "preserve":
        return title
    if order != "title-artist":
        raise ValueError(f"unsupported resolved metadata order: {order!r}")

    # Do not guess which of three or more fields is the artist.
    parts = title.split(" - ")
    if len(parts) != 2:
        return title

    song, artist = (part.strip() for part in parts)
    if not song or not artist:
        return title

    return f"{artist} - {song}"
