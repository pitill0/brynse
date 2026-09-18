"""Radio-oriented orchestration adapters built on the generic resolver core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fluxtuner_ripper.buffer import EncodedAudioRingBuffer
from fluxtuner_ripper.frames import IncrementalFrameTimeline
from fluxtuner_ripper.models import (
    BoundaryMatch,
    BoundaryRelationResult,
    SplitDecision,
    TemporalSplitDecision,
    TrackCandidate,
)
from fluxtuner_ripper.orchestrator import (
    AcousticCandidateResolution,
    CandidateResolver,
    DefaultCandidateResolver,
)


@dataclass(frozen=True)
class BoundaryResolution:
    """All intermediate evidence used to resolve a radio track boundary."""

    track: TrackCandidate
    match: BoundaryMatch | None
    relation: BoundaryRelationResult | None
    temporal: TemporalSplitDecision
    split: SplitDecision


class BoundaryResolver(Protocol):
    """Resolve radio track candidates into aligned boundary decisions."""

    def resolve_boundary(
        self,
        *,
        track: TrackCandidate,
        timeline: IncrementalFrameTimeline,
        ring_buffer: EncodedAudioRingBuffer,
    ) -> BoundaryResolution | None: ...


class RippingOrchestrator(DefaultCandidateResolver):
    """Radio adapter over the source-agnostic candidate resolver."""

    def resolve_boundary(
        self,
        *,
        track: TrackCandidate,
        timeline: IncrementalFrameTimeline,
        ring_buffer: EncodedAudioRingBuffer,
    ) -> BoundaryResolution | None:
        resolved = self.resolve_candidate(
            candidate=track.as_boundary_candidate(),
            timeline=timeline,
            ring_buffer=ring_buffer,
        )
        if resolved is None:
            return None

        match = BoundaryMatch(
            track=track,
            acoustic=resolved.acoustic,
            delta_seconds=abs(resolved.acoustic.time_seconds - track.start_time_seconds),
        )

        return BoundaryResolution(
            track=track,
            match=match,
            relation=resolved.relation,
            temporal=resolved.temporal,
            split=resolved.split,
        )


class CandidateBoundaryResolver:
    """Adapt a generic candidate resolver to the radio track boundary contract."""

    def __init__(self, resolver: CandidateResolver) -> None:
        self._resolver = resolver

    def resolve_boundary(
        self,
        *,
        track: TrackCandidate,
        timeline: IncrementalFrameTimeline,
        ring_buffer: EncodedAudioRingBuffer,
    ) -> BoundaryResolution | None:
        resolved = self._resolver.resolve_candidate(
            candidate=track.as_boundary_candidate(),
            timeline=timeline,
            ring_buffer=ring_buffer,
        )
        if resolved is None:
            return None

        match = None
        relation = None

        if isinstance(resolved, AcousticCandidateResolution):
            match = BoundaryMatch(
                track=track,
                acoustic=resolved.acoustic,
                delta_seconds=abs(resolved.acoustic.time_seconds - track.start_time_seconds),
            )
            relation = resolved.relation

        return BoundaryResolution(
            track=track,
            match=match,
            relation=relation,
            temporal=resolved.temporal,
            split=resolved.split,
        )
