"""Composition layer for resolving semantic track candidates into split decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fluxtuner_ripper.acoustic import (
    AcousticCandidateFinder,
    AcousticWindowExtractor,
    FfmpegAcousticDecoder,
    RmsAcousticAnalyzer,
)
from fluxtuner_ripper.buffer import EncodedAudioRingBuffer
from fluxtuner_ripper.frames import IncrementalFrameTimeline
from fluxtuner_ripper.hybrid import HybridAcousticSplitResolver
from fluxtuner_ripper.matching import (
    BoundaryRelationClassifier,
    NearestBoundaryMatcher,
    TemporalSplitAligner,
    TemporalSplitPolicy,
)
from fluxtuner_ripper.models import (
    AcousticBoundaryCandidate,
    BoundaryCandidate,
    BoundaryMatch,
    BoundaryRelationResult,
    SplitDecision,
    TemporalSplitDecision,
    TemporalSplitKind,
    TrackCandidate,
)
from fluxtuner_ripper.mp3_refinement import Mp3BoundaryRefiner


@dataclass(frozen=True)
class CandidateResolution:
    """Resolved source-agnostic boundary candidate and its split decision."""

    candidate: BoundaryCandidate
    acoustic: AcousticBoundaryCandidate
    relation: BoundaryRelationResult
    temporal: TemporalSplitDecision
    split: SplitDecision


@dataclass(frozen=True)
class BoundaryResolution:
    """All intermediate evidence used to resolve a track boundary."""

    track: TrackCandidate
    match: BoundaryMatch | None
    relation: BoundaryRelationResult | None
    temporal: TemporalSplitDecision
    split: SplitDecision


class CandidateResolver(Protocol):
    """Resolve source-agnostic boundary candidates into aligned split decisions."""

    def resolve_candidate(
        self,
        *,
        candidate: BoundaryCandidate,
        timeline: IncrementalFrameTimeline,
        ring_buffer: EncodedAudioRingBuffer,
    ) -> CandidateResolution | None:
        """Resolve one generic boundary candidate."""


class BoundaryResolver(Protocol):
    """Common interface for boundary-resolution strategies."""

    def resolve_boundary(
        self,
        *,
        track: TrackCandidate,
        timeline: IncrementalFrameTimeline,
        ring_buffer: EncodedAudioRingBuffer,
    ) -> BoundaryResolution | None: ...


class DefaultCandidateResolver:
    """Resolve source-agnostic boundary candidates using bounded acoustic evidence."""

    def __init__(
        self,
        *,
        window_extractor: AcousticWindowExtractor | None = None,
        decoder: FfmpegAcousticDecoder | None = None,
        analyzer: RmsAcousticAnalyzer | None = None,
        candidate_finder: AcousticCandidateFinder | None = None,
        matcher: NearestBoundaryMatcher | None = None,
        relation_classifier: BoundaryRelationClassifier | None = None,
        split_policy: TemporalSplitPolicy | None = None,
        split_aligner: TemporalSplitAligner | None = None,
        mp3_refiner: Mp3BoundaryRefiner | None = None,
        refinement_analyzer: RmsAcousticAnalyzer | None = None,
    ) -> None:
        self._window_extractor = window_extractor or AcousticWindowExtractor()
        self._decoder = decoder or FfmpegAcousticDecoder()
        self._analyzer = analyzer or RmsAcousticAnalyzer()
        self._candidate_finder = candidate_finder or AcousticCandidateFinder()
        self._matcher = matcher or NearestBoundaryMatcher()
        self._relation_classifier = relation_classifier or BoundaryRelationClassifier()
        self._split_policy = split_policy or TemporalSplitPolicy()
        self._split_aligner = split_aligner or TemporalSplitAligner()
        self._mp3_refiner = mp3_refiner
        self._refinement_analyzer = refinement_analyzer or RmsAcousticAnalyzer(window_seconds=0.05)

    def resolve_candidate(
        self,
        *,
        candidate: BoundaryCandidate,
        timeline: IncrementalFrameTimeline,
        ring_buffer: EncodedAudioRingBuffer,
    ) -> CandidateResolution | None:
        """Resolve one generic boundary candidate into a frame-aligned split."""

        window = self._window_extractor.extract(
            candidate_time_seconds=candidate.time_seconds,
            timeline=timeline,
            ring_buffer=ring_buffer,
        )
        if window is None:
            return None

        pcm = self._decoder.decode(window)
        profile = self._analyzer.analyze(pcm)
        candidates = self._candidate_finder.find(
            profile=profile,
            window=window,
            center_time_seconds=candidate.time_seconds,
        )
        selected = self._matcher.match_candidate(
            candidate=candidate,
            acoustic_candidates=candidates,
        )
        if selected is None:
            return None

        relation = self._relation_classifier.classify(
            semantic_time_seconds=candidate.time_seconds,
            acoustic_time_seconds=selected.time_seconds,
        )
        temporal = self._split_policy.decide(relation)

        if self._mp3_refiner is not None and temporal.kind is TemporalSplitKind.HARD_CUT:
            refinement_profile = self._refinement_analyzer.analyze(pcm)
            temporal = self._mp3_refiner.refine(
                decision=temporal,
                profile=refinement_profile,
                window=window,
            )

        split = self._split_aligner.align(
            decision=temporal,
            timeline=timeline,
        )

        return CandidateResolution(
            candidate=candidate,
            acoustic=selected,
            relation=relation,
            temporal=temporal,
            split=split,
        )

    def resolve_boundary(
        self,
        *,
        track: TrackCandidate,
        timeline: IncrementalFrameTimeline,
        ring_buffer: EncodedAudioRingBuffer,
    ) -> BoundaryResolution | None:
        generic_candidate = track.as_boundary_candidate()

        resolved = self.resolve_candidate(
            candidate=generic_candidate,
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


class RippingOrchestrator(DefaultCandidateResolver):
    """Backward-compatible radio-oriented candidate resolver name."""


class HybridRippingOrchestrator:
    """Resolve AAC transitions using fine RMS basins and persistent energy edges."""

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
