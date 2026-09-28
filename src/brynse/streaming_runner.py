"""Incremental generic boundary resolution over a live encoded stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from brynse.buffer import EncodedAudioRingBuffer
from brynse.frames import IncrementalFrameTimeline
from brynse.models import BoundaryCandidate
from brynse.orchestrator import CandidateResolution, CandidateResolver
from brynse.providers import BoundaryProvider


class StreamingIngestor(Protocol):
    """Minimal encoded-stream contract required by the streaming runner."""

    @property
    def timeline(self) -> IncrementalFrameTimeline: ...

    @property
    def ring_buffer(self) -> EncodedAudioRingBuffer: ...

    def feed(self, data: bytes) -> object: ...


class StreamingProviderCursor:
    """Request boundary candidates only for newly observed stream time."""

    def __init__(self, *, provider: BoundaryProvider) -> None:
        self._provider = provider
        self._end_time_seconds = 0.0

    @property
    def end_time_seconds(self) -> float:
        return self._end_time_seconds

    def propose_until(
        self,
        end_time_seconds: float,
    ) -> tuple[BoundaryCandidate, ...]:
        if end_time_seconds < self._end_time_seconds:
            raise ValueError("end_time_seconds must not move backwards")

        if end_time_seconds == self._end_time_seconds:
            return ()

        candidates = self._provider.propose(
            start_time_seconds=self._end_time_seconds,
            end_time_seconds=end_time_seconds,
        )

        self._end_time_seconds = end_time_seconds
        return tuple(candidates)


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
        provider: BoundaryProvider | None = None,
    ) -> None:
        if settle_seconds <= 0:
            raise ValueError("settle_seconds must be greater than zero")

        self._ingestor = ingestor
        self._resolver = resolver
        self._settle_seconds = settle_seconds
        self._provider_cursor = (
            StreamingProviderCursor(provider=provider) if provider is not None else None
        )
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

        provider_cursor = self._provider_cursor
        if provider_cursor is not None:
            stream_end = self._stream_end_time_seconds()
            if stream_end is not None:
                candidates = provider_cursor.propose_until(stream_end)
                self._pending.extend(candidates)
                self._pending.sort(key=lambda item: item.time_seconds)

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

        return self._resolve_candidates(ready)

    def finalize(self) -> tuple[StreamingCandidateResult, ...]:
        """Resolve every candidate still pending at end of stream."""

        pending = self._pending
        self._pending = []
        return self._resolve_candidates(pending)

    def _resolve_candidates(
        self,
        candidates: list[BoundaryCandidate],
    ) -> tuple[StreamingCandidateResult, ...]:
        completed: list[StreamingCandidateResult] = []

        for candidate in candidates:
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
