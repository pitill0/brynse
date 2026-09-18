"""Fan-out encoded live input to analysis retention and disk spool."""

from __future__ import annotations

from pathlib import Path

from fluxtuner_ripper.buffer import EncodedAudioRingBuffer
from fluxtuner_ripper.frames import IncrementalFrameTimeline
from fluxtuner_ripper.ingest import EncodedStreamIngestor
from fluxtuner_ripper.streaming_spool import (
    StreamingSpool,
    create_safe_streaming_spool,
)


def create_safe_spooling_ingestor(
    *,
    ingestor: EncodedStreamIngestor,
    spool_directory: Path,
) -> SpoolingEncodedStreamIngestor:
    """Create a live fan-out ingestor using the runtime spool safety policy."""
    spool = create_safe_streaming_spool(
        directory=spool_directory,
    )
    return SpoolingEncodedStreamIngestor(
        ingestor=ingestor,
        spool=spool,
    )


class SpoolingEncodedStreamIngestor:
    """Feed identical encoded bytes to the normal ingestor and disk spool."""

    def __init__(
        self,
        *,
        ingestor: EncodedStreamIngestor,
        spool: StreamingSpool,
    ) -> None:
        self._ingestor = ingestor
        self._spool = spool
        self._first_frame_offset: int | None = None

    @property
    def timeline(self) -> IncrementalFrameTimeline:
        return self._ingestor.timeline

    @property
    def ring_buffer(self) -> EncodedAudioRingBuffer:
        return self._ingestor.ring_buffer

    @property
    def spool(self) -> StreamingSpool:
        return self._spool

    @property
    def first_frame_offset(self) -> int | None:
        return self._first_frame_offset

    def feed(self, data: bytes) -> None:
        """Persist bytes and feed the normal encoded-stream parser."""

        if not data:
            self._ingestor.feed(data)
            return

        expected_start = self._spool.end_offset
        appended_start, appended_end = self._spool.append(data)

        if appended_start != expected_start:
            raise RuntimeError("streaming spool absolute offset discontinuity")

        if appended_end - appended_start != len(data):
            raise RuntimeError("streaming spool appended unexpected byte count")

        self._ingestor.feed(data)

        if self._first_frame_offset is None:
            frames = self._ingestor.timeline.frames
            if frames:
                self._first_frame_offset = frames[0].offset
