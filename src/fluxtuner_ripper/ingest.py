"""Source-agnostic encoded stream ingestion."""

from __future__ import annotations

from fluxtuner_ripper.buffer import EncodedAudioRingBuffer
from fluxtuner_ripper.frames import IncrementalFrameTimeline


class EncodedStreamIngestor:
    """Retain encoded stream bytes while building an exact frame timeline."""

    def __init__(self, *, codec: str, ring_max_bytes: int) -> None:
        self._ring = EncodedAudioRingBuffer(ring_max_bytes)
        self._timeline = IncrementalFrameTimeline(codec)

    @property
    def ring_buffer(self) -> EncodedAudioRingBuffer:
        return self._ring

    @property
    def timeline(self) -> IncrementalFrameTimeline:
        return self._timeline

    def feed(self, chunk: bytes) -> None:
        """Consume encoded stream bytes without assuming any metadata protocol."""

        if not chunk:
            return

        self._ring.append(chunk)
        self._timeline.feed(chunk)
        self._timeline.discard_before(self._ring.start_offset)
