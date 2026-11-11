"""Bounded codec probing that preserves the original ICY byte stream."""

from __future__ import annotations

from threading import Event
from typing import Protocol

from brynse.integrations.radio.codec_detection import detect_audio_codec
from brynse.integrations.radio.icy import IcyStreamParser

MAX_PROBE_BYTES = 128 * 1024
PROBE_READ_BYTES = 4096


class ReadableStream(Protocol):
    def read(self, max_bytes: int) -> bytes: ...


def probe_stream_codec(
    source: ReadableStream,
    *,
    metaint: int,
    stop_event: Event,
) -> tuple[str | None, tuple[bytes, ...]]:
    """Identify MP3/AAC from clean audio, retaining every raw ICY byte.

    None means no conclusive codec, including an interrupted probe.
    The caller must replay the returned raw chunks before reading again.
    """
    parser = IcyStreamParser(metaint)
    audio = bytearray()
    raw_chunks: list[bytes] = []
    total = 0

    while total < MAX_PROBE_BYTES and not stop_event.is_set():
        try:
            chunk = source.read(min(PROBE_READ_BYTES, MAX_PROBE_BYTES - total))
        except Exception:
            if stop_event.is_set():
                return None, tuple(raw_chunks)
            raise

        if not chunk:
            break

        if len(chunk) > MAX_PROBE_BYTES - total:
            raise ValueError("stream exceeded the codec probe read limit")

        raw_chunks.append(chunk)
        total += len(chunk)

        parsed = parser.feed(chunk)
        audio.extend(parsed.audio)

        codec = detect_audio_codec(bytes(audio))
        if stop_event.is_set():
            return None, tuple(raw_chunks)
        if codec is not None:
            return codec, tuple(raw_chunks)

    return None, tuple(raw_chunks)
