"""Materialize source-agnostic resolved segments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fluxtuner_ripper.ingest import EncodedStreamIngestor
from fluxtuner_ripper.models import SegmentByteRange
from fluxtuner_ripper.orchestrator import CandidateResolution
from fluxtuner_ripper.output import (
    SegmentRangePlanner,
    TrackOutputService,
    create_safe_track_output_service,
)


@dataclass(frozen=True)
class MaterializedSegment:
    """One finalized segment persisted from the retained encoded stream."""

    index: int
    start_offset: int
    end_offset: int
    path: Path


class GenericSegmentWriter:
    """Materialize resolved generic boundaries as finalized segment files."""

    def __init__(
        self,
        *,
        ingestor: EncodedStreamIngestor,
        directory: Path,
        codec: str,
        minimum_tail_seconds: float = 1.0,
        range_planner: SegmentRangePlanner | None = None,
        output_service: TrackOutputService | None = None,
    ) -> None:
        if codec not in {"mp3", "aac"}:
            raise ValueError("codec must be 'mp3' or 'aac'")
        if minimum_tail_seconds <= 0:
            raise ValueError("minimum_tail_seconds must be greater than zero")

        self._ingestor = ingestor
        self._directory = directory
        self._codec = codec
        self._minimum_tail_seconds = minimum_tail_seconds
        self._range_planner = range_planner or SegmentRangePlanner()
        self._output_service = output_service or create_safe_track_output_service()

    def write(
        self,
        resolutions: tuple[CandidateResolution, ...],
    ) -> tuple[MaterializedSegment, ...]:
        """Write all resolved segments, including the final stream tail."""

        frames = self._ingestor.timeline.frames
        if not frames:
            raise RuntimeError("cannot write segments without timeline frames")

        first_frame = frames[0]
        last_frame = frames[-1]

        current_start_offset = first_frame.offset
        stream_end_offset = last_frame.offset + last_frame.length
        stream_end_time_seconds = (
            last_frame.time_seconds + last_frame.samples / last_frame.sample_rate
        )

        written: list[MaterializedSegment] = []

        for index, resolution in enumerate(resolutions, start=1):
            decision = resolution.split
            is_last_resolution = index == len(resolutions)

            if is_last_resolution:
                tail_seconds = max(
                    0.0,
                    stream_end_time_seconds - resolution.temporal.incoming_start_seconds,
                )

                if tail_seconds < self._minimum_tail_seconds:
                    final_range = SegmentByteRange(
                        start_offset=current_start_offset,
                        end_offset=stream_end_offset,
                    )

                    path = self._output_service.write_track(
                        source=self._ingestor.ring_buffer,
                        byte_range=final_range,
                        directory=self._directory,
                        stem=f"segment-{index:04d}",
                        codec=self._codec,
                    )

                    written.append(
                        MaterializedSegment(
                            index=index,
                            start_offset=final_range.start_offset,
                            end_offset=final_range.end_offset,
                            path=path,
                        )
                    )

                    current_start_offset = stream_end_offset
                    break

            plan = self._range_planner.plan(
                previous_start_offset=current_start_offset,
                next_end_offset=stream_end_offset,
                decision=decision,
            )

            path = self._output_service.write_track(
                source=self._ingestor.ring_buffer,
                byte_range=plan.outgoing,
                directory=self._directory,
                stem=f"segment-{index:04d}",
                codec=self._codec,
            )

            written.append(
                MaterializedSegment(
                    index=index,
                    start_offset=plan.outgoing.start_offset,
                    end_offset=plan.outgoing.end_offset,
                    path=path,
                )
            )

            current_start_offset = plan.incoming.start_offset

        if current_start_offset < stream_end_offset:
            final_index = len(written) + 1
            final_range = SegmentByteRange(
                start_offset=current_start_offset,
                end_offset=stream_end_offset,
            )

            path = self._output_service.write_track(
                source=self._ingestor.ring_buffer,
                byte_range=final_range,
                directory=self._directory,
                stem=f"segment-{final_index:04d}",
                codec=self._codec,
            )

            written.append(
                MaterializedSegment(
                    index=final_index,
                    start_offset=final_range.start_offset,
                    end_offset=final_range.end_offset,
                    path=path,
                )
            )

        return tuple(written)
