from __future__ import annotations

from brynse.integrations.radio.models import (
    IcyParseResult,
    MetadataEvent,
)


def _extract_stream_title(metadata: str) -> str | None:
    """Extract the ICY ``StreamTitle`` value from one metadata block."""

    marker = "StreamTitle='"
    start = metadata.find(marker)
    if start < 0:
        return None

    start += len(marker)
    end = metadata.find("';", start)
    if end < 0:
        return None

    title = metadata[start:end].strip()
    return title or None


class IcyStreamParser:
    """Incremental ICY metadata parser for streams using ``icy-metaint``.

    The parser accepts arbitrary chunk boundaries, strips ICY metadata blocks,
    and reports metadata events anchored to absolute offsets in the clean audio
    byte stream.
    """

    def __init__(self, metaint: int) -> None:
        if metaint <= 0:
            raise ValueError("metaint must be greater than zero")

        self._metaint = metaint
        self._audio_until_metadata = metaint
        self._audio_offset = 0
        self._state = "audio"
        self._metadata_remaining = 0
        self._metadata_buffer = bytearray()

    @property
    def metaint(self) -> int:
        return self._metaint

    @property
    def audio_offset(self) -> int:
        """Absolute count of clean audio bytes emitted so far."""

        return self._audio_offset

    def feed(self, chunk: bytes) -> IcyParseResult:
        """Consume one raw ICY chunk and return clean audio plus new events."""

        if not chunk:
            return IcyParseResult(audio=b"", events=())

        audio_out = bytearray()
        events: list[MetadataEvent] = []
        pos = 0

        while pos < len(chunk):
            if self._state == "audio":
                take = min(self._audio_until_metadata, len(chunk) - pos)
                if take:
                    audio_out.extend(chunk[pos : pos + take])
                    pos += take
                    self._audio_offset += take
                    self._audio_until_metadata -= take

                if self._audio_until_metadata == 0:
                    self._state = "metadata_length"
                continue

            if self._state == "metadata_length":
                length_byte = chunk[pos]
                pos += 1
                self._metadata_remaining = length_byte * 16
                self._metadata_buffer.clear()

                if self._metadata_remaining == 0:
                    self._audio_until_metadata = self._metaint
                    self._state = "audio"
                else:
                    self._state = "metadata_body"
                continue

            if self._state == "metadata_body":
                take = min(self._metadata_remaining, len(chunk) - pos)
                if take:
                    self._metadata_buffer.extend(chunk[pos : pos + take])
                    pos += take
                    self._metadata_remaining -= take

                if self._metadata_remaining == 0:
                    event = self._decode_metadata_event()
                    if event is not None:
                        events.append(event)

                    self._metadata_buffer.clear()
                    self._audio_until_metadata = self._metaint
                    self._state = "audio"
                continue

            raise RuntimeError(f"invalid ICY parser state: {self._state}")

        return IcyParseResult(audio=bytes(audio_out), events=tuple(events))

    def _decode_metadata_event(self) -> MetadataEvent | None:
        raw = bytes(self._metadata_buffer).rstrip(b"\x00")
        if not raw:
            return None

        text = raw.decode("utf-8", errors="replace")
        title = _extract_stream_title(text)
        if title is None:
            return None

        return MetadataEvent(title=title, audio_offset=self._audio_offset)
