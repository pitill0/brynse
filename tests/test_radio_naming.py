from __future__ import annotations

import pytest

from brynse.integrations.radio.naming import format_radio_track_title


@pytest.mark.parametrize(
    "raw",
    [
        "Artist - Song",
        "My one and only love - JOHN COLTRANE",
        "Station identification",
        "",
    ],
)
def test_default_preserves_original_metadata(raw: str) -> None:
    assert format_radio_track_title(raw) == raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "My one and only love - JOHN COLTRANE",
            "JOHN COLTRANE - My one and only love",
        ),
        ("Thunderstruck - AC-DC", "AC-DC - Thunderstruck"),
        ("Part-time lover - Stevie Wonder", "Stevie Wonder - Part-time lover"),
        ("Déjà vu - Beyoncé", "Beyoncé - Déjà vu"),
        ("Song / One? - Artist: Name", "Artist: Name - Song / One?"),
    ],
)
def test_declared_title_artist_order(raw: str, expected: str) -> None:
    assert format_radio_track_title(raw, metadata_order="title-artist") == expected


def test_declared_artist_title_order_preserves_title() -> None:
    raw = "JOHN COLTRANE - My one and only love"
    assert format_radio_track_title(raw, metadata_order="artist-title") == raw


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "Station identification",
        "Song-Artist",
        "Song - Artist - Live",
        " - Artist",
        "Song - ",
        "   - Artist",
        "Song -   ",
    ],
)
def test_ambiguous_or_missing_fields_preserve_original(raw: str) -> None:
    assert format_radio_track_title(raw, metadata_order="title-artist") == raw


def test_unknown_metadata_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="metadata_order"):
        format_radio_track_title("Song - Artist", metadata_order="unknown")
