"""Incremental generic boundary resolution over a live encoded stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fluxtuner_ripper.buffer import EncodedAudioRingBuffer
from fluxtuner_ripper.models import BoundaryCandidate
from fluxtuner_ripper.orchestrator import CandidateResolution, CandidateResolver
from fluxtuner_ripper.ripping import IncrementalFrameTimeline


class StreamingIngestor(Protocol):
    """Minimal encoded-stream contract required by the streaming runner."""

    @property
    def timeline(self) -> IncrementalFrameTimeline: ...

    @property
    def ring_buffer(self) -> EncodedAudioRingBuffer: ...

    def feed(self, data: bytes) -> object: ...


@dataclass(frozen=True)
class StreamingCandidateResult:
    """Completed resolution attempt for one submitted boundary candidate."""

    candidate: BoundaryCandidate
    resolution: CandidateResolution | None


class StreamingGenericRunner:
    """Resolve submitted candidates once enough post-boundary audio exists."""

    def __init__(
        self,
        *,
        ingestor: StreamingIngestor,
        resolver: CandidateResolver,
        settle_seconds: float,
    ) -> None:
        if settle_seconds <= 0:
            raise ValueError("settle_seconds must be greater than zero")

        self._ingestor = ingestor
        self._resolver = resolver
        self._settle_seconds = settle_seconds
        self._pending: list[BoundaryCandidate] = []

    @property
    def pending(self) -> tuple[BoundaryCandidate, ...]:
        return tuple(self._pending)

    def submit_candidate(
        self,
        candidate: BoundaryCandidate,
    ) -> tuple[StreamingCandidateResult, ...]:
        """Submit one candidate and resolve it immediately if already mature."""

        self._pending.append(candidate)
        self._pending.sort(key=lambda item: item.time_seconds)

        return self._resolve_ready()

    def feed(
        self,
        chunk: bytes,
    ) -> tuple[StreamingCandidateResult, ...]:
        """Ingest encoded bytes and resolve newly mature candidates."""

        self._ingestor.feed(chunk)
        return self._resolve_ready()

    def _stream_end_time_seconds(self) -> float | None:
        frames = self._ingestor.timeline.frames
        if not frames:
            return None

        last_frame = frames[-1]
        return last_frame.time_seconds + last_frame.samples / last_frame.sample_rate

    def _resolve_ready(self) -> tuple[StreamingCandidateResult, ...]:
        stream_end = self._stream_end_time_seconds()
        if stream_end is None:
            return ()

        ready: list[BoundaryCandidate] = []
        waiting: list[BoundaryCandidate] = []

        for candidate in self._pending:
            if stream_end >= candidate.time_seconds + self._settle_seconds:
                ready.append(candidate)
            else:
                waiting.append(candidate)

        self._pending = waiting

        completed: list[StreamingCandidateResult] = []

        for candidate in ready:
            resolution = self._resolver.resolve_candidate(
                candidate=candidate,
                timeline=self._ingestor.timeline,
                ring_buffer=self._ingestor.ring_buffer,
            )
            completed.append(
                StreamingCandidateResult(
                    candidate=candidate,
                    resolution=resolution,
                )
            )

        return tuple(completed)
