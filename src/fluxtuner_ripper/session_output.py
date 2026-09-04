"""Write resolved session transitions to finalized track files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fluxtuner_ripper.output import TrackOutputService, TrackRangePlanner
from fluxtuner_ripper.ripping import RippingStreamIngestor
from fluxtuner_ripper.session import TrackTransition

_UNSAFE_STEM_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")
_MULTI_SPACE = re.compile(r"\s+")


def safe_track_stem(title: str) -> str:
    """Return a filesystem-friendly stem while preserving readable titles."""
    stem = _UNSAFE_STEM_CHARS.sub("_", title)
    stem = _MULTI_SPACE.sub(" ", stem).strip(" ._-")
    return stem or "track"


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
        self._output_service = output_service or TrackOutputService()

    def write_transition(self, transition: TrackTransition) -> WrittenTrack:
        frames = self._ingestor.timeline.frames
        if not frames:
            raise RuntimeError("cannot write track without timeline frames")

        last_frame = frames[-1]
        next_end_offset = last_frame.offset + last_frame.length

        plan = self._range_planner.plan(
            previous_start_offset=transition.outgoing.start_offset,
            next_end_offset=next_end_offset,
            decision=transition.boundary.split,
        )

        path = self._output_service.write_track(
            source=self._ingestor.ring_buffer,
            byte_range=plan.outgoing,
            directory=self._directory,
            stem=safe_track_stem(transition.outgoing.title),
            codec=self._codec,
        )

        return WrittenTrack(
            transition=transition,
            path=path,
        )
