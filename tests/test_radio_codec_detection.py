"""Deterministic MP3/AAC detection without network or external decoders."""

from __future__ import annotations

import pytest

from brynse.integrations.radio.codec_detection import detect_audio_codec


def _mp3_frame() -> bytes:
    # MPEG-1 Layer III, 128 kbps, 44.1 kHz; 417-byte frame.
    return b"\xff\xfb\x90\x00" + bytes(413)


def _aac_frame(length: int = 64) -> bytes:
    # AAC-LC ADTS, 44.1 kHz, stereo; complete frame.
    header = bytes(
        (
            0xFF,
            0xF1,
            0x50,
            0x80 | ((length >> 11) & 0x03),
            (length >> 3) & 0xFF,
            ((length & 0x07) << 5) | 0x1F,
            0xFC,
        )
    )
    return header + bytes(length - 7)


@pytest.mark.parametrize(
    ("audio", "expected"),
    [
        (b"", None),
        (b"not an audio stream", None),
        (bytes(2048), None),
        (_mp3_frame(), None),
        (_mp3_frame() * 2, None),
        (_mp3_frame() * 3, "mp3"),
        (b"leading junk" + _mp3_frame() * 3, "mp3"),
        (_aac_frame(), None),
        (_aac_frame() * 2, None),
        (_aac_frame() * 3, "aac"),
        (b"leading junk" + _aac_frame() * 3, "aac"),
        (_mp3_frame() * 3 + _aac_frame() * 3, None),
    ],
)
def test_detect_audio_codec(audio: bytes, expected: str | None) -> None:
    assert detect_audio_codec(audio) == expected


def test_detector_requires_contiguous_complete_frames() -> None:
    frame = _mp3_frame()

    assert detect_audio_codec(frame + b"junk" + frame + b"junk" + frame) is None
    assert detect_audio_codec(frame * 2 + frame[:20]) is None


def test_detector_rejects_conflicting_aac_and_mp3_evidence() -> None:
    audio = _aac_frame() * 3 + b"separator" + _mp3_frame() * 3

    assert detect_audio_codec(audio) is None
