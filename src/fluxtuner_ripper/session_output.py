"""Write resolved session transitions to finalized track files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fluxtuner_ripper.output import (
    TrackOutputService,
    TrackRangePlanner,
    create_safe_track_output_service,
)
from fluxtuner_ripper.ripping import RippingStreamIngestor
from fluxtuner_ripper.session import SegmentTransition, TrackTransition

_UNSAFE_STEM_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")
_MULTI_SPACE = re.compile(r"\s+")


def safe_track_stem(title: str) -> str:
    """Return a filesystem-friendly stem while preserving readable titles."""
    stem = _UNSAFE_STEM_CHARS.sub("_", title)
    stem = _MULTI_SPACE.sub(" ", stem).strip(" ._-")
    return stem or "track"


@dataclass(frozen=True)
class WrittenSegment:
    """A finalized outgoing segment written from one resolved transition."""

    transition: SegmentTransition
    path: Path


@dataclass(frozen=True)
class WrittenTrack:
    """A finalized outgoing track written from one resolved transition."""

    transition: TrackTransition
    path: Path


class SessionOutputWriter:
    """Convert resolved transitions into finalized outgoing track files."""

    def __init__(
        self,
        *,
        ingestor: RippingStreamIngestor,
        directory: Path,
        codec: str,
        range_planner: TrackRangePlanner | None = None,
        output_service: TrackOutputService | None = None,
    ) -> None:
        if codec not in {"mp3", "aac"}:
            raise ValueError("codec must be 'mp3' or 'aac'")

        self._ingestor = ingestor
        self._directory = directory
        self._codec = codec
        self._range_planner = range_planner or TrackRangePlanner()
        self._output_service = output_service or create_safe_track_output_service()
        self._current_start_offset: int | None = None

    def write_segment_transition(
        self,
        transition: SegmentTransition,
    ) -> WrittenSegment:
        frames = self._ingestor.timeline.frames
        if not frames:
            raise RuntimeError("cannot write segment without timeline frames")

        last_frame = frames[-1]
        next_end_offset = last_frame.offset + last_frame.length
        previous_start_offset = (
            self._current_start_offset
            if self._current_start_offset is not None
            else transition.outgoing.start_offset
        )

        plan = self._range_planner.plan(
            previous_start_offset=previous_start_offset,
            next_end_offset=next_end_offset,
            decision=transition.boundary.split,
        )

        path = self._output_service.write_track(
            source=self._ingestor.ring_buffer,
            byte_range=plan.outgoing,
            directory=self._directory,
            stem=safe_track_stem(transition.outgoing.label or "segment"),
            codec=self._codec,
        )

        self._current_start_offset = plan.incoming.start_offset

        return WrittenSegment(
            transition=transition,
            path=path,
        )

    def write_transition(self, transition: TrackTransition) -> WrittenTrack:
        written = self.write_segment_transition(
            transition.as_segment_transition(),
        )

        return WrittenTrack(
            transition=transition,
            path=written.path,
        )
