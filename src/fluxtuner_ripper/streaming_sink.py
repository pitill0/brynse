"""Incremental materialization of resolved live stream segments."""

from __future__ import annotations

from pathlib import Path

from fluxtuner_ripper.generic_output import MaterializedSegment
from fluxtuner_ripper.ingest import EncodedStreamIngestor
from fluxtuner_ripper.models import TrackByteRange
from fluxtuner_ripper.orchestrator import CandidateResolution
from fluxtuner_ripper.output import (
    EncodedByteSource,
    TrackOutputService,
    TrackRangePlanner,
)
from fluxtuner_ripper.streaming_spool import StreamingSpool


class StreamingSegmentSink:
    """Write completed segments while leaving the current tail open."""

    def __init__(
        self,
        *,
        ingestor: EncodedStreamIngestor,
        directory: Path,
        codec: str,
        spool: StreamingSpool | None = None,
        range_planner: TrackRangePlanner | None = None,
        output_service: TrackOutputService | None = None,
    ) -> None:
        if codec not in {"mp3", "aac"}:
            raise ValueError("codec must be 'mp3' or 'aac'")

        self._ingestor = ingestor
        self._spool = spool
        self._source: EncodedByteSource = spool if spool is not None else ingestor.ring_buffer
        self._directory = directory
        self._codec = codec
        self._range_planner = range_planner or TrackRangePlanner()
        self._output_service = output_service or TrackOutputService()

        self._current_start_offset: int | None = None
        self._next_index = 1
        self._finalized = False

    @property
    def current_start_offset(self) -> int | None:
        return self._current_start_offset

    def accept(
        self,
        resolution: CandidateResolution,
    ) -> MaterializedSegment:
        """Materialize the segment closed by one resolved boundary."""

        if self._finalized:
            raise RuntimeError("cannot accept resolution after finalization")

        frames = self._ingestor.timeline.frames
        if not frames:
            raise RuntimeError("cannot write segment without timeline frames")

        if self._current_start_offset is None:
            self._current_start_offset = frames[0].offset

        stream_end_offset = self._source.end_offset

        plan = self._range_planner.plan(
            previous_start_offset=self._current_start_offset,
            next_end_offset=stream_end_offset,
            decision=resolution.split,
        )

        segment = self._write_range(plan.outgoing)

        self._current_start_offset = plan.incoming.start_offset

        if self._spool is not None:
            self._spool.discard_before(self._current_start_offset)

        return segment

    def finalize(self) -> MaterializedSegment | None:
        """Materialize the still-open tail at end-of-stream."""

        if self._finalized:
            return None

        self._finalized = True

        frames = self._ingestor.timeline.frames
        if not frames:
            return None

        if self._current_start_offset is None:
            self._current_start_offset = frames[0].offset

        last_frame = frames[-1]
        stream_end_offset = last_frame.offset + last_frame.length

        if self._current_start_offset >= stream_end_offset:
            return None

        return self._write_range(
            TrackByteRange(
                start_offset=self._current_start_offset,
                end_offset=stream_end_offset,
            )
        )

    def _write_range(
        self,
        byte_range: TrackByteRange,
    ) -> MaterializedSegment:
        if not self._source.contains(
            byte_range.start_offset,
            byte_range.end_offset,
        ):
            raise RuntimeError("segment bytes are no longer retained in the encoded byte source")

        index = self._next_index

        path = self._output_service.write_track(
            source=self._source,
            byte_range=byte_range,
            directory=self._directory,
            stem=f"segment-{index:04d}",
            codec=self._codec,
        )

        self._next_index += 1

        return MaterializedSegment(
            index=index,
            start_offset=byte_range.start_offset,
            end_offset=byte_range.end_offset,
            path=path,
        )
