"""Regression contracts for radio metadata filename ordering."""

from __future__ import annotations

import pytest

from brynse.integrations.radio.metadata_order import (
    filename_title,
    resolve_metadata_order,
)

JAZZ_URL = "jazz-wr01.ice.infomaniak.ch/jazz-wr01-128.mp3"


@pytest.mark.parametrize(
    "url",
    [
        f"http://{JAZZ_URL}",
        f"https://{JAZZ_URL}",
        "http://jazz-wr01.ice.infomaniak.ch:80/jazz-wr01-128.mp3",
        "https://jazz-wr01.ice.infomaniak.ch:443/jazz-wr01-128.mp3",
    ],
)
def test_auto_recognizes_exact_jazz_radio_profile(url: str) -> None:
    assert resolve_metadata_order(url, "auto") == "title-artist"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/synthetic",
        "https://jazz-wr01.ice.infomaniak.ch/another-station.mp3",
        f"https://{JAZZ_URL}?token=value",
        f"https://user:pass@{JAZZ_URL}",
        "https://jazz-wr01.ice.infomaniak.ch:8443/jazz-wr01-128.mp3",
        "https://jazz-wr01.ice.infomaniak.ch.evil.invalid/jazz-wr01-128.mp3",
    ],
)
def test_auto_preserves_unknown_or_nonmatching_urls(url: str) -> None:
    assert resolve_metadata_order(url, "auto") == "preserve"


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (
            "Synthetic A - Synthetic Artist",
            "Synthetic Artist - Synthetic A",
        ),
        (
            "Despechá - Rosalía",
            "Rosalía - Despechá",
        ),
        (
            "Осень - ДДТ",
            "ДДТ - Осень",
        ),
        (
            "Song - Artist - Remix",
            "Song - Artist - Remix",
        ),
        (
            "Unknown",
            "Unknown",
        ),
        (
            " - Artist",
            " - Artist",
        ),
    ],
)
def test_title_artist_filename_order(title: str, expected: str) -> None:
    assert filename_title(title, "title-artist") == expected


def test_preserve_order_keeps_original_title() -> None:
    original = "Synthetic A - Synthetic Artist"
    assert filename_title(original, "preserve") == original


def test_ripping_config_preserves_legacy_positional_arguments(tmp_path):
    from pathlib import Path

    from brynse.integrations.radio.runner import RippingRunConfig

    config = RippingRunConfig(
        "https://example.invalid/synthetic",
        Path(tmp_path),
        "mp3",
        "ffmpeg",
        12.0,
    )

    assert config.metadata_threshold_seconds == 12.0
    assert config.metadata_order == "auto"
