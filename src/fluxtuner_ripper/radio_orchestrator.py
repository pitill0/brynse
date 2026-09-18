"""Radio-oriented orchestration adapters built on the generic resolver core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fluxtuner_ripper.acoustic import (
    AcousticWindowExtractor,
    FfmpegAcousticDecoder,
    RmsAcousticAnalyzer,
)
from fluxtuner_ripper.buffer import EncodedAudioRingBuffer
from fluxtuner_ripper.frames import IncrementalFrameTimeline
from fluxtuner_ripper.hybrid import HybridAcousticSplitResolver
from fluxtuner_ripper.matching import TemporalSplitAligner
from fluxtuner_ripper.models import (
    BoundaryMatch,
    BoundaryRelationResult,
    SplitDecision,
    TemporalSplitDecision,
    TrackCandidate,
)
from fluxtuner_ripper.orchestrator import DefaultCandidateResolver


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


class HybridRippingOrchestrator:
    """Resolve AAC radio transitions using the hybrid acoustic strategy."""

    def __init__(
        self,
        *,
        window_extractor: AcousticWindowExtractor | None = None,
        decoder: FfmpegAcousticDecoder | None = None,
        analyzer: RmsAcousticAnalyzer | None = None,
        resolver: HybridAcousticSplitResolver | None = None,
        split_aligner: TemporalSplitAligner | None = None,
    ) -> None:
        self._window_extractor = window_extractor or AcousticWindowExtractor()
        self._decoder = decoder or FfmpegAcousticDecoder()
        self._analyzer = analyzer or RmsAcousticAnalyzer(window_seconds=0.05)
        self._resolver = resolver or HybridAcousticSplitResolver()
        self._split_aligner = split_aligner or TemporalSplitAligner()

    def resolve_boundary(
        self,
        *,
        track: TrackCandidate,
        timeline: IncrementalFrameTimeline,
        ring_buffer: EncodedAudioRingBuffer,
    ) -> BoundaryResolution | None:
        window = self._window_extractor.extract(
            candidate_time_seconds=track.start_time_seconds,
            timeline=timeline,
            ring_buffer=ring_buffer,
        )
        if window is None:
            return None

        pcm = self._decoder.decode(window)
        profile = self._analyzer.analyze(pcm)

        temporal = self._resolver.resolve(
            profile=profile,
            window=window,
            semantic_time_seconds=track.start_time_seconds,
        )
        if temporal is None:
            return None

        split = self._split_aligner.align(
            decision=temporal,
            timeline=timeline,
        )

        return BoundaryResolution(
            track=track,
            match=None,
            relation=None,
            temporal=temporal,
            split=split,
        )
