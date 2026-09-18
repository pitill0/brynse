"""Composition layer for resolving boundary candidates into split decisions."""

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
from fluxtuner_ripper.matching import (
    BoundaryRelationClassifier,
    NearestBoundaryMatcher,
    TemporalSplitAligner,
    TemporalSplitPolicy,
)
from fluxtuner_ripper.models import (
    AcousticBoundaryCandidate,
    BoundaryCandidate,
    BoundaryRelationResult,
    SplitDecision,
    TemporalSplitDecision,
    TemporalSplitKind,
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
