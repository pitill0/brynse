"""Conservative filename formatting for radio metadata."""

from __future__ import annotations

METADATA_ORDERS = ("preserve", "artist-title", "title-artist")


def validate_metadata_order(metadata_order: str) -> None:
    """Reject unsupported metadata order declarations."""
    if metadata_order not in METADATA_ORDERS:
        raise ValueError("metadata_order must be 'preserve', 'artist-title' or 'title-artist'")


def format_radio_track_title(
    title: str,
    *,
    metadata_order: str = "preserve",
) -> str:
    """Format a filename title without changing the source metadata.

    Reordering requires an explicitly declared title-artist source order,
    exactly one spaced separator, and two nonempty fields. Other input is
    returned unchanged. Filesystem sanitization belongs to the output writer.
    """
    validate_metadata_order(metadata_order)
    if metadata_order != "title-artist" or title.count(" - ") != 1:
        return title

    track_title, artist = (part.strip() for part in title.split(" - "))
    if not track_title or not artist:
        return title

    return f"{artist} - {track_title}"
