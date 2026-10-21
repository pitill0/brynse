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
from brynse.acoustic_island import AcousticIslandDetector
from brynse.bounded_insert import BoundedInsertRegionDiscovery
from brynse.buffer import EncodedAudioRingBuffer
from brynse.fade_specialist import (
    FadeSpecialist,
    FadeSpecialistDiagnostics,
    IncomingFadeSelection,
    OutgoingFadeSelection,
)
from brynse.fade_to_floor import FadeToFloorDetector
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
from brynse.pcm_boundary import (
    PcmBoundaryObservation,
    PcmBoundaryVerifier,
)
from brynse.pcm_shared_edge import PcmSharedEdgeVerifier
from brynse.pcm_temporal_edge import PcmTemporalEdgeVerifier
from brynse.transition_evidence import (
    FadeRoutingEvidence,
    TransitionEvidenceExtractor,
)
from brynse.transition_router import (
    BoundedInsertRoutingAssessment,
    BoundedInsertRoutingProvenance,
    DefaultTransitionRouter,
    FadeRoutingAssessment,
    HardCutRoutingAssessment,
    InterstitialRoutingAssessment,
    TransitionKind,
    TransitionRoute,
    TransitionRouter,
    TransitionRoutingContext,
)
from brynse.transition_specialists import (
    BoundedInsertSpecialist,
    HardCutSpecialist,
    InterstitialSpecialist,
    LegacyAmbiguousSpecialist,
    RoutedTransitionResolver,
    TransitionSpecialistInput,
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
    route: TransitionRoute | None = None
    bounded_insert: BoundedInsertRoutingAssessment | None = None
    bounded_insert_routing: BoundedInsertRoutingProvenance | None = None
    fade_routing: FadeRoutingAssessment | None = None
    interstitial_routing: InterstitialRoutingAssessment | None = None
    hard_cut_routing: HardCutRoutingAssessment | None = None
    outgoing_fade_selection: OutgoingFadeSelection | None = None
    incoming_fade_selection: IncomingFadeSelection | None = None
    pcm_boundary_scan: tuple[PcmBoundaryObservation, ...] = ()


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
        incoming_refiner: object | None = None,
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
        self._routed_transition_resolver = RoutedTransitionResolver(
            router=self._transition_router,
            hard_cut=HardCutSpecialist(
                temporal_edge_verifier=PcmTemporalEdgeVerifier(),
            ),
            fade=FadeSpecialist(
                temporal_edge_verifier=PcmTemporalEdgeVerifier(),
            ),
            interstitial=InterstitialSpecialist(
                incoming_pipeline=self._incoming_refiner,
                temporal_edge_verifier=PcmTemporalEdgeVerifier(),
            ),
            bounded_insert=BoundedInsertSpecialist(
                temporal_edge_verifier=PcmTemporalEdgeVerifier(),
            ),
            ambiguous=LegacyAmbiguousSpecialist(
                shared_edge_verifier=PcmSharedEdgeVerifier(),
            ),
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

        observe_fade = getattr(
            self._temporal_resolver,
            "observe_fade_routing_evidence",
            None,
        )
        fade = (
            observe_fade(
                profile=profile,
                window=window,
                semantic_time_seconds=candidate.time_seconds,
            )
            if callable(observe_fade)
            else FadeRoutingEvidence()
        )

        observe_hypotheses = (
            getattr(
                self._incoming_refiner,
                "observe_hypotheses",
                None,
            )
            if self._incoming_refiner is not None
            else None
        )
        routing_hypotheses = (
            observe_hypotheses(
                window=window,
                pcm8=pcm8,
            )
            if callable(observe_hypotheses)
            else ()
        )

        fade_to_floor = FadeToFloorDetector().observe(
            reference_time_seconds=candidate.time_seconds,
            fade=fade,
            hypotheses=routing_hypotheses,
        )

        bounded_insert_discovery = BoundedInsertRegionDiscovery()

        bounded_insert_regions = bounded_insert_discovery.discover(
            reference_time_seconds=candidate.time_seconds,
            hypotheses=routing_hypotheses,
        )
        bounded_insert_asymmetric_regions = bounded_insert_discovery.discover_asymmetric(
            reference_time_seconds=candidate.time_seconds,
            hypotheses=routing_hypotheses,
        )

        acoustic_islands = AcousticIslandDetector().observe(
            reference_time_seconds=candidate.time_seconds,
            fade=fade,
            hypotheses=routing_hypotheses,
            bounded_insert_regions=bounded_insert_regions,
        )

        routing_context = TransitionRoutingContext(
            reference_time_seconds=candidate.time_seconds,
            evidence=TransitionEvidenceExtractor().extract(
                spectral=None,
                local_discontinuity=selected.local_discontinuity,
            ),
            hypotheses=routing_hypotheses,
            fade=fade,
            bounded_insert_regions=bounded_insert_regions,
            acoustic_islands=acoustic_islands,
            fade_to_floor=fade_to_floor,
            bounded_insert_asymmetric_regions=(bounded_insert_asymmetric_regions),
        )

        if self._on_routing_context is not None:
            self._on_routing_context(routing_context)

        specialist_input = TransitionSpecialistInput(
            routing=routing_context,
            selected=selected,
            profile=profile,
            window=window,
            pcm8=pcm8,
            pcm16=pcm16,
        )

        routing_kind, specialist_result, transition_route = (
            self._routed_transition_resolver.resolve_with_route(
                observation=specialist_input,
            )
        )
        temporal = specialist_result.temporal

        if temporal is None:
            return None

        pcm_boundary_scan: tuple[PcmBoundaryObservation, ...] = ()

        if self._on_routing_observation is not None:
            anchor_times = [
                temporal.outgoing_end_seconds,
                temporal.incoming_start_seconds,
            ]

            semantic_roles = None if transition_route is None else transition_route.semantic_roles

            # Router/specialist landmarks only delimit PCM observation.
            # None of them receives temporal authority here.
            anchor_times.extend(
                hypothesis.time_seconds for hypothesis in routing_context.hypotheses
            )

            anchor_times.extend(
                fade.floor_time_seconds for fade in routing_context.fade.outgoing_fades
            )

            anchor_times.extend(
                fade.start_time_seconds for fade in routing_context.fade.incoming_fades
            )

            if semantic_roles is not None:
                outgoing_handoff = semantic_roles.outgoing_handoff
                incoming_handoff = semantic_roles.incoming_handoff
                inserted_region = semantic_roles.inserted_region
                outgoing_edge_fade = semantic_roles.outgoing_edge_fade
                incoming_edge_fade = semantic_roles.incoming_edge_fade

                if outgoing_handoff is not None:
                    anchor_times.append(outgoing_handoff.time_seconds)

                if incoming_handoff is not None:
                    anchor_times.append(incoming_handoff.time_seconds)

                if inserted_region is not None:
                    region_left = getattr(
                        inserted_region,
                        "left",
                        None,
                    )
                    region_right = getattr(
                        inserted_region,
                        "right",
                        None,
                    )

                    if region_left is not None:
                        anchor_times.append(region_left.time_seconds)

                    if region_right is not None:
                        anchor_times.append(region_right.time_seconds)

                if outgoing_edge_fade is not None:
                    anchor_times.append(outgoing_edge_fade.floor_time_seconds)

                if incoming_edge_fade is not None:
                    anchor_times.append(incoming_edge_fade.start_time_seconds)

            unique_anchor_times = tuple(dict.fromkeys(anchor_times))

            pcm_boundary_scan = PcmBoundaryVerifier().observe_many(
                pcm=pcm16,
                absolute_start_time_seconds=window.start_time_seconds,
                anchor_times_seconds=unique_anchor_times,
            )

        outgoing_fade_selection = None
        incoming_fade_selection = None

        if routing_kind is TransitionKind.FADE:
            diagnostics = specialist_result.diagnostics
            if not isinstance(
                diagnostics,
                FadeSpecialistDiagnostics,
            ):
                raise RuntimeError("fade specialist returned invalid diagnostics")

            outgoing_fade_selection = diagnostics.outgoing_selection
            incoming_fade_selection = diagnostics.incoming_selection

        if self._on_routing_observation is not None:
            bounded_insert = None
            bounded_insert_routing = None
            fade_routing = None
            interstitial_routing = None
            hard_cut_routing = None

            if isinstance(
                self._transition_router,
                DefaultTransitionRouter,
            ):
                bounded_insert = self._transition_router.assess_bounded_insert(
                    context=routing_context,
                )
                bounded_insert_routing = self._transition_router.assess_bounded_insert_routing(
                    context=routing_context,
                )
                fade_routing = self._transition_router.assess_fade(
                    context=routing_context,
                )
                interstitial_routing = self._transition_router.assess_interstitial(
                    context=routing_context,
                )
                hard_cut_routing = self._transition_router.assess_hard_cut(
                    context=routing_context,
                )

            self._on_routing_observation(
                TransitionRoutingObservation(
                    context=routing_context,
                    kind=routing_kind,
                    temporal=temporal,
                    route=transition_route,
                    bounded_insert=bounded_insert,
                    bounded_insert_routing=bounded_insert_routing,
                    fade_routing=fade_routing,
                    interstitial_routing=interstitial_routing,
                    hard_cut_routing=hard_cut_routing,
                    outgoing_fade_selection=outgoing_fade_selection,
                    incoming_fade_selection=incoming_fade_selection,
                    pcm_boundary_scan=pcm_boundary_scan,
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
