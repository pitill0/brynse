"""Incremental materialization of resolved live stream segments."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from brynse.generic_output import MaterializedSegment
from brynse.ingest import EncodedStreamIngestor
from brynse.models import SegmentByteRange
from brynse.orchestrator import CandidateResolution
from brynse.output import (
    EncodedByteSource,
    SegmentOutputService,
    SegmentRangePlanner,
    create_safe_segment_output_service,
)
from brynse.streaming_spool import StreamingSpool


class InitialStartOffsetSource(Protocol):
    """Source that can expose the first valid encoded-frame offset lazily."""

    @property
    def first_frame_offset(self) -> int | None: ...


class StreamingSegmentSink:
    """Write completed segments while leaving the current tail open."""

    def __init__(
        self,
        *,
        ingestor: EncodedStreamIngestor,
        directory: Path,
        codec: str,
        spool: StreamingSpool | None = None,
        initial_start_offset: int | None = None,
        initial_start_source: InitialStartOffsetSource | None = None,
        range_planner: SegmentRangePlanner | None = None,
        output_service: SegmentOutputService | None = None,
    ) -> None:
        if codec not in {"mp3", "aac"}:
            raise ValueError("codec must be 'mp3' or 'aac'")

        self._ingestor = ingestor
        self._spool = spool
        self._source: EncodedByteSource = spool if spool is not None else ingestor.ring_buffer
        self._directory = directory
        self._codec = codec
        self._initial_start_offset = initial_start_offset
        self._initial_start_source = initial_start_source
        self._range_planner = range_planner or SegmentRangePlanner()
        self._output_service = output_service or create_safe_segment_output_service()

        self._current_start_offset: int | None = None
        self._next_index = 1
        self._finalized = False

    @property
    def current_start_offset(self) -> int | None:
        return self._current_start_offset

    def _ensure_current_start_offset(self) -> None:
        if self._current_start_offset is not None:
            return

        start_offset = self._initial_start_offset

        if start_offset is None and self._initial_start_source is not None:
            start_offset = self._initial_start_source.first_frame_offset

        if start_offset is None:
            frames = self._ingestor.timeline.frames
            if not frames:
                raise RuntimeError("cannot determine segment start without timeline frames")
            start_offset = frames[0].offset

        self._current_start_offset = start_offset

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

        self._ensure_current_start_offset()

        current_start_offset = self._current_start_offset
        if current_start_offset is None:
            raise RuntimeError("segment start offset was not initialized")

        stream_end_offset = self._source.end_offset

        plan = self._range_planner.plan(
            previous_start_offset=current_start_offset,
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

        self._ensure_current_start_offset()

        current_start_offset = self._current_start_offset
        if current_start_offset is None:
            raise RuntimeError("segment start offset was not initialized")

        last_frame = frames[-1]
        stream_end_offset = last_frame.offset + last_frame.length

        if current_start_offset >= stream_end_offset:
            return None

        return self._write_range(
            SegmentByteRange(
                start_offset=current_start_offset,
                end_offset=stream_end_offset,
            )
        )

    def _write_range(
        self,
        byte_range: SegmentByteRange,
    ) -> MaterializedSegment:
        if not self._source.contains(
            byte_range.start_offset,
            byte_range.end_offset,
        ):
            raise RuntimeError("segment bytes are no longer retained in the encoded byte source")

        index = self._next_index

        path = self._output_service.write_segment(
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
