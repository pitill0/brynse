"""Fan-out encoded live input to analysis retention and disk spool."""

from __future__ import annotations

from fluxtuner_ripper.buffer import EncodedAudioRingBuffer
from fluxtuner_ripper.ingest import EncodedStreamIngestor
from fluxtuner_ripper.ripping import IncrementalFrameTimeline
from fluxtuner_ripper.streaming_spool import StreamingSpool


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

    @property
    def timeline(self) -> IncrementalFrameTimeline:
        return self._ingestor.timeline

    @property
    def ring_buffer(self) -> EncodedAudioRingBuffer:
        return self._ingestor.ring_buffer

    @property
    def spool(self) -> StreamingSpool:
        return self._spool

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
