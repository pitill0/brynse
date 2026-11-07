"""Regression contracts for ICY metadata text encoding."""

from brynse.integrations.radio.icy import IcyStreamParser


def _icy_packet(title: bytes) -> bytes:
    metadata = b"StreamTitle='" + title + b"';"
    padded = metadata.ljust(
        ((len(metadata) + 15) // 16) * 16,
        b"\x00",
    )
    assert len(padded) // 16 <= 255
    return b"ABCD" + bytes([len(padded) // 16]) + padded


def test_icy_preserves_valid_utf8_title():
    parser = IcyStreamParser(metaint=4)
    result = parser.feed(_icy_packet("Beyoncé".encode()))

    assert len(result.events) == 1
    assert result.events[0].title == "Beyoncé"
    assert result.events[0].audio_offset == 4


def test_icy_recovers_windows_1252_title():
    parser = IcyStreamParser(metaint=4)
    result = parser.feed(_icy_packet("Beyoncé".encode("cp1252")))

    assert len(result.events) == 1
    assert result.events[0].title == "Beyoncé"
    assert result.events[0].audio_offset == 4


def test_icy_recovers_fragmented_windows_1252_title():
    parser = IcyStreamParser(metaint=4)
    packet = _icy_packet("Beyoncé – Música".encode("cp1252"))

    results = [
        parser.feed(packet[:5]),
        parser.feed(packet[5:9]),
        parser.feed(packet[9:]),
    ]

    events = [event for result in results for event in result.events]

    assert len(events) == 1
    assert events[0].title == "Beyoncé – Música"
    assert events[0].audio_offset == 4
    assert b"".join(result.audio for result in results) == b"ABCD"


def test_icy_preserves_multibyte_utf8_title():
    parser = IcyStreamParser(metaint=4)
    result = parser.feed(_icy_packet("Música 東京 🎶".encode()))

    assert len(result.events) == 1
    assert result.events[0].title == "Música 東京 🎶"
    assert result.events[0].audio_offset == 4
