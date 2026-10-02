"""Composition layer for resolving boundary candidates into split decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from brynse.acoustic import (
    AcousticCandidateFinder,
    AcousticWindowExtractor,
    FfmpegAcousticDecoder,
    RmsAcousticAnalyzer,
)
from brynse.buffer import EncodedAudioRingBuffer
from brynse.fade_specialist import (
    FadeSpecialist,
    OutgoingFadeSelection,
)
from brynse.frames import IncrementalFrameTimeline
from brynse.hybrid import HybridAcousticSplitResolver
from brynse.matching import (
    BoundaryRelationClassifier,
    NearestBoundaryMatcher,
    TemporalSplitAligner,
    TemporalSplitPolicy,
)
from brynse.models import (
    AcousticBoundaryCandidate,
    BoundaryCandidate,
    BoundaryRelationResult,
    SplitDecision,
    TemporalSplitDecision,
    TemporalSplitKind,
)
from brynse.mp3_refinement import Mp3BoundaryRefiner
from brynse.multisignal import (
    AcousticMultiSignalCandidateBuilder,
    MultiSignalCandidateResolver,
    MultiSignalTemporalResolver,
)
from brynse.transition_evidence import TransitionEvidenceExtractor
from brynse.transition_router import (
    DefaultTransitionRouter,
    TransitionKind,
    TransitionRouter,
    TransitionRoutingContext,
)


@dataclass(frozen=True)
class CandidateResolution:
    """Resolved source-agnostic boundary candidate and its split decision."""

    candidate: BoundaryCandidate
    temporal: TemporalSplitDecision
    split: SplitDecision


@dataclass(frozen=True)
class TransitionRoutingObservation:
    """Shadow comparison between router ownership and the final temporal decision."""

    context: TransitionRoutingContext
    kind: TransitionKind
    temporal: TemporalSplitDecision
    outgoing_fade_selection: OutgoingFadeSelection | None = None


@dataclass(frozen=True)
class AcousticCandidateResolution(CandidateResolution):
    """Candidate resolution enriched with acoustic matching evidence."""

    acoustic: AcousticBoundaryCandidate
    relation: BoundaryRelationResult


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
    ) -> AcousticCandidateResolution | None:
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

        return AcousticCandidateResolution(
            candidate=candidate,
            temporal=temporal,
            split=split,
            acoustic=selected,
            relation=relation,
        )


class MultiSignalAcousticCandidateResolver:
    """Resolve candidates through the frozen multisignal boundary policy."""

    def __init__(
        self,
        *,
        window_extractor: AcousticWindowExtractor | None = None,
        decoder8: FfmpegAcousticDecoder | None = None,
        decoder16: FfmpegAcousticDecoder | None = None,
        analyzer: RmsAcousticAnalyzer | None = None,
        candidate_builder: AcousticMultiSignalCandidateBuilder | None = None,
        candidate_resolver: MultiSignalCandidateResolver | None = None,
        temporal_resolver: MultiSignalTemporalResolver | None = None,
        split_aligner: TemporalSplitAligner | None = None,
        incoming_refiner=None,
        transition_router: TransitionRouter | None = None,
        on_routing_context: Callable[[TransitionRoutingContext], None] | None = None,
        on_routing_observation: Callable[[TransitionRoutingObservation], None] | None = None,
    ) -> None:
        self._window_extractor = (
            window_extractor
            if window_extractor is not None
            else AcousticWindowExtractor(
                search_radius_seconds=8.0,
            )
        )

        self._decoder8 = (
            decoder8
            if decoder8 is not None
            else FfmpegAcousticDecoder(
                output_sample_rate=8000,
            )
        )

        self._decoder16 = (
            decoder16
            if decoder16 is not None
            else FfmpegAcousticDecoder(
                output_sample_rate=16000,
            )
        )

        self._analyzer = (
            analyzer
            if analyzer is not None
            else RmsAcousticAnalyzer(
                window_seconds=0.05,
            )
        )

        self._candidate_builder = (
            candidate_builder
            if candidate_builder is not None
            else AcousticMultiSignalCandidateBuilder()
        )

        self._candidate_resolver = (
            candidate_resolver
            if candidate_resolver is not None
            else MultiSignalCandidateResolver(
                candidate_builder=self._candidate_builder,
            )
        )

        self._temporal_resolver = (
            temporal_resolver if temporal_resolver is not None else MultiSignalTemporalResolver()
        )

        self._split_aligner = split_aligner if split_aligner is not None else TemporalSplitAligner()
        self._incoming_refiner = incoming_refiner
        self._transition_router = (
            transition_router if transition_router is not None else DefaultTransitionRouter()
        )
        self._on_routing_context = on_routing_context
        self._on_routing_observation = on_routing_observation

    def resolve_candidate(
        self,
        *,
        candidate: BoundaryCandidate,
        timeline: IncrementalFrameTimeline,
        ring_buffer: EncodedAudioRingBuffer,
    ) -> CandidateResolution | None:
        window = self._window_extractor.extract(
            candidate_time_seconds=candidate.time_seconds,
            timeline=timeline,
            ring_buffer=ring_buffer,
        )

        if window is None:
            return None

        pcm8 = self._decoder8.decode(window)
        pcm16 = self._decoder16.decode(window)

        profile = self._analyzer.analyze(pcm8)

        selected = self._candidate_resolver.resolve_acoustic_boundary(
            semantic_time_seconds=candidate.time_seconds,
            window=window,
            pcm8=pcm8,
            pcm16=pcm16,
        )

        if selected is None:
            return None

        routing_hypotheses = None
        routing_context = None
        routing_kind = None

        if (
            self._on_routing_context is not None
            or self._on_routing_observation is not None
        ):
            fade = self._temporal_resolver.observe_fade_routing_evidence(
                profile=profile,
                window=window,
                semantic_time_seconds=candidate.time_seconds,
            )
            routing_hypotheses = (
                self._incoming_refiner.observe_hypotheses(
                    window=window,
                    pcm8=pcm8,
                )
                if self._incoming_refiner is not None
                else ()
            )
            routing_context = TransitionRoutingContext(
                reference_time_seconds=candidate.time_seconds,
                evidence=TransitionEvidenceExtractor().extract(
                    spectral=None,
                    local_discontinuity=selected.local_discontinuity,
                ),
                hypotheses=routing_hypotheses,
                fade=fade,
            )

            if self._on_routing_context is not None:
                self._on_routing_context(routing_context)

            if self._on_routing_observation is not None:
                routing_kind = self._transition_router.classify(
                    context=routing_context,
                )

        temporal = self._temporal_resolver.resolve(
            selected=selected,
            profile=profile,
            window=window,
            semantic_time_seconds=candidate.time_seconds,
        )

        if temporal is None:
            return None

        if self._incoming_refiner is not None:
            if routing_hypotheses is None:
                temporal = self._incoming_refiner.refine(
                    decision=temporal,
                    window=window,
                    pcm8=pcm8,
                    pcm16=pcm16,
                )
            else:
                temporal = self._incoming_refiner.refine(
                    decision=temporal,
                    window=window,
                    pcm8=pcm8,
                    pcm16=pcm16,
                    hypotheses=routing_hypotheses,
                )

        if self._on_routing_observation is not None:
            if routing_context is None or routing_kind is None:
                raise RuntimeError("routing observation requires classified context")

            outgoing_fade_selection = FadeSpecialist.select_outgoing(
                outgoing_fades=routing_context.fade.outgoing_fades,
                hypotheses=routing_context.hypotheses,
            )

            self._on_routing_observation(
                TransitionRoutingObservation(
                    context=routing_context,
                    kind=routing_kind,
                    temporal=temporal,
                    outgoing_fade_selection=outgoing_fade_selection,
                )
            )

        split = self._split_aligner.align(
            decision=temporal,
            timeline=timeline,
        )

        return CandidateResolution(
            candidate=candidate,
            temporal=temporal,
            split=split,
        )


class HybridCandidateResolver:
    """Resolve generic boundary candidates using the hybrid acoustic strategy."""

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

    def resolve_candidate(
        self,
        *,
        candidate: BoundaryCandidate,
        timeline: IncrementalFrameTimeline,
        ring_buffer: EncodedAudioRingBuffer,
    ) -> CandidateResolution | None:
        window = self._window_extractor.extract(
            candidate_time_seconds=candidate.time_seconds,
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
            semantic_time_seconds=candidate.time_seconds,
        )
        if temporal is None:
            return None

        split = self._split_aligner.align(
            decision=temporal,
            timeline=timeline,
        )

        return CandidateResolution(
            candidate=candidate,
            temporal=temporal,
            split=split,
        )
