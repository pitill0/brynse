from __future__ import annotations

from fluxtuner_ripper.buffer import EncodedAudioRingBuffer
from fluxtuner_ripper.frames import IncrementalFrameTimeline
from fluxtuner_ripper.integrations.radio.icy import IcyStreamParser
from fluxtuner_ripper.integrations.radio.models import (
    MetadataEvent,
    RippingIngestResult,
    TimedMetadataEvent,
)


class RippingStreamIngestor:
    """Join ICY parsing, encoded buffering, and exact frame-timeline resolution."""

    def __init__(self, *, metaint: int, codec: str, ring_max_bytes: int) -> None:
        self._icy = IcyStreamParser(metaint)
        self._ring = EncodedAudioRingBuffer(ring_max_bytes)
        self._timeline = IncrementalFrameTimeline(codec)
        self._pending_metadata: list[MetadataEvent] = []

    @property
    def ring_buffer(self) -> EncodedAudioRingBuffer:
        return self._ring

    @property
    def timeline(self) -> IncrementalFrameTimeline:
        return self._timeline

    def feed(self, chunk: bytes) -> RippingIngestResult:
        """Consume one raw ICY chunk and resolve new metadata onto audio time."""

        parsed = self._icy.feed(chunk)

        if parsed.audio:
            self._ring.append(parsed.audio)
            self._timeline.feed(parsed.audio)

        self._pending_metadata.extend(parsed.events)

        timed: list[TimedMetadataEvent] = []
        still_pending: list[MetadataEvent] = []
        for event in self._pending_metadata:
            frame = self._timeline.frame_for_audio_offset(event.audio_offset)
            if frame is None:
                still_pending.append(event)
                continue

            timed.append(
                TimedMetadataEvent(
                    title=event.title,
                    audio_offset=event.audio_offset,
                    audio_time_seconds=frame.time_seconds,
                )
            )

        self._pending_metadata = still_pending
        self._timeline.discard_before(self._ring.start_offset)

        return RippingIngestResult(
            audio=parsed.audio,
            metadata_events=parsed.events,
            timed_metadata_events=tuple(timed),
        )
