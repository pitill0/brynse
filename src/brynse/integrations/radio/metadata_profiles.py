"""Verified source conventions for radio output filenames."""

from __future__ import annotations

from urllib.parse import urlsplit

from brynse.integrations.radio.naming import validate_metadata_order

# Match individual stream endpoints, never an entire hosting domain.
# Verified against live ICY metadata and finalized output on 2026-10-19.
_STREAM_METADATA_ORDERS = {
    ("jazz-wr01.ice.infomaniak.ch", "/jazz-wr01-128.mp3"): "title-artist",
}


def resolve_radio_metadata_order(
    url: str,
    *,
    metadata_order: str = "auto",
) -> str:
    """Resolve a verified stream convention, preserving unknown input.

    Explicit settings override profiles. Automatic matching accepts only
    HTTP/HTTPS endpoints on their default ports, without query parameters
    or embedded credentials. It does not perform network lookups.
    """
    validate_metadata_order(metadata_order)
    if metadata_order != "auto":
        return metadata_order

    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            return "preserve"
        default_port = 80 if parsed.scheme == "http" else 443
        if (
            parsed.port not in {None, default_port}
            or parsed.query
            or parsed.username is not None
            or parsed.password is not None
        ):
            return "preserve"
        key = (parsed.hostname or "", parsed.path)
    except ValueError:
        return "preserve"

    return _STREAM_METADATA_ORDERS.get(key, "preserve")
