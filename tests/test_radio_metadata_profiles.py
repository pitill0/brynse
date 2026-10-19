from __future__ import annotations

from pathlib import Path

import pytest

from brynse.integrations.radio.metadata_profiles import resolve_radio_metadata_order
from brynse.integrations.radio.runner import RippingRunConfig

JAZZ_HTTP = "http://jazz-wr01.ice.infomaniak.ch/jazz-wr01-128.mp3"
JAZZ_HTTPS = "https://jazz-wr01.ice.infomaniak.ch/jazz-wr01-128.mp3"


@pytest.mark.parametrize("url", [JAZZ_HTTP, JAZZ_HTTPS])
def test_known_stream_automatically_resolves_title_artist(url: str) -> None:
    assert resolve_radio_metadata_order(url) == "title-artist"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/radio.mp3",
        "https://jazz-wr01.ice.infomaniak.ch/another-stream.mp3",
        "https://another.ice.infomaniak.ch/jazz-wr01-128.mp3",
        "https://jazz-wr01.ice.infomaniak.ch.example.invalid/jazz-wr01-128.mp3",
        JAZZ_HTTPS + "?stream=another",
        "https://jazz-wr01.ice.infomaniak.ch:8443/jazz-wr01-128.mp3",
        "ftp://jazz-wr01.ice.infomaniak.ch/jazz-wr01-128.mp3",
        "",
    ],
)
def test_unrecognized_stream_preserves_received_order(url: str) -> None:
    assert resolve_radio_metadata_order(url) == "preserve"


@pytest.mark.parametrize("order", ["preserve", "artist-title", "title-artist"])
@pytest.mark.parametrize(
    "url",
    [JAZZ_HTTPS, "https://example.invalid/radio.mp3"],
)
def test_explicit_order_overrides_automatic_profiles(url: str, order: str) -> None:
    assert resolve_radio_metadata_order(url, metadata_order=order) == order


def test_unknown_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="metadata_order"):
        resolve_radio_metadata_order(JAZZ_HTTPS, metadata_order="unknown")


def test_runner_uses_automatic_order_by_default(tmp_path: Path) -> None:
    config = RippingRunConfig(url=JAZZ_HTTPS, output_directory=tmp_path)
    assert config.metadata_order == "auto"


@pytest.mark.parametrize(
    ("url", "explicit_order", "expected"),
    [
        (JAZZ_HTTP, None, "title-artist"),
        (JAZZ_HTTPS, None, "title-artist"),
        ("https://example.invalid/radio.mp3", None, "preserve"),
        (JAZZ_HTTPS, "preserve", "preserve"),
        (JAZZ_HTTPS, "artist-title", "artist-title"),
    ],
)
def test_runner_resolves_order_before_constructing_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    url: str,
    explicit_order: str | None,
    expected: str,
) -> None:
    import io

    from brynse.integrations.radio import runner as runner_module
    from brynse.source import BinaryIOStreamSource

    orders = []
    original_writer = runner_module.SessionOutputWriter

    def capture_writer(**kwargs):
        orders.append(kwargs["metadata_order"])
        return original_writer(**kwargs)

    monkeypatch.setattr(runner_module, "SessionOutputWriter", capture_writer)
    config = RippingRunConfig(url=url, output_directory=tmp_path)
    if explicit_order is not None:
        from dataclasses import replace

        config = replace(config, metadata_order=explicit_order)
    stream = io.BytesIO()
    runner = runner_module.RippingRunner(
        config,
        stream_opener=lambda source_url: BinaryIOStreamSource(
            stream,
            metadata={"Content-Type": "audio/mpeg", "icy-metaint": "417"},
        ),
    )

    runner.run()

    assert orders == [expected]
    assert stream.closed


@pytest.mark.parametrize(
    "url",
    [
        "http://jazz-wr01.ice.infomaniak.ch:80/jazz-wr01-128.mp3",
        "https://jazz-wr01.ice.infomaniak.ch:443/jazz-wr01-128.mp3",
        "HTTPS://JAZZ-WR01.ICE.INFOMANIAK.CH/jazz-wr01-128.mp3",
    ],
)
def test_profile_url_normalization(url: str) -> None:
    assert resolve_radio_metadata_order(url) == "title-artist"
    assert resolve_radio_metadata_order(url, metadata_order="auto") == "title-artist"


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@jazz-wr01.ice.infomaniak.ch/jazz-wr01-128.mp3",
        "https://jazz-wr01.ice.infomaniak.ch:invalid/jazz-wr01-128.mp3",
        "https://jazz-wr01.ice.infomaniak.ch:99999/jazz-wr01-128.mp3",
        "https://[invalid",
    ],
)
def test_automatic_profiles_preserve_unsupported_urls(url: str) -> None:
    assert resolve_radio_metadata_order(url) == "preserve"
