from types import SimpleNamespace

from brynse.boundaries import BoundaryConfidence
from brynse.models import TemporalSplitDecision, TemporalSplitKind
from brynse.transition_evidence import TransitionEvidenceState
from brynse.transition_router import TransitionRoutingContext
from brynse.transition_specialists import (
    HardCutSpecialist,
    InterstitialSpecialist,
    LegacyAmbiguousSpecialist,
    TransitionSpecialistInput,
    TransitionSpecialistResult,
)


def _observation(*, boundary: float = 100.0):
    return TransitionSpecialistInput(
        routing=TransitionRoutingContext(
            reference_time_seconds=102.0,
            evidence=TransitionEvidenceState(),
            hypotheses=(),
        ),
        selected=SimpleNamespace(time_seconds=boundary),
        profile=object(),
        window=object(),
        pcm8=object(),
        pcm16=object(),
    )


def test_hard_cut_specialist_owns_both_edges() -> None:
    result = HardCutSpecialist().resolve(
        observation=_observation(boundary=98.5),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=98.5,
        incoming_start_seconds=98.5,
    )


def test_interstitial_specialist_reuses_incoming_pipeline() -> None:
    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=103.0,
    )

    class IncomingPipeline:
        def __init__(self):
            self.calls = []

        def refine(self, **kwargs):
            self.calls.append(kwargs)
            return refined

    pipeline = IncomingPipeline()
    observation = _observation(boundary=100.0)

    result = InterstitialSpecialist(
        incoming_pipeline=pipeline,
    ).resolve(
        observation=observation,
    )

    assert result.temporal is refined
    assert len(pipeline.calls) == 1

    call = pipeline.calls[0]

    assert call["decision"] == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=100.0,
    )
    assert call["window"] is observation.window
    assert call["pcm8"] is observation.pcm8
    assert call["pcm16"] is observation.pcm16
    assert call["hypotheses"] is observation.routing.hypotheses


def test_interstitial_specialist_refines_outgoing_from_supported_analysis() -> None:
    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=108.0,
    )

    high = SimpleNamespace(
        hypothesis=SimpleNamespace(time_seconds=104.0),
        assessment=SimpleNamespace(
            candidate_confidence=BoundaryConfidence.HIGH,
        ),
    )
    farther_high = SimpleNamespace(
        hypothesis=SimpleNamespace(time_seconds=94.0),
        assessment=SimpleNamespace(
            candidate_confidence=BoundaryConfidence.HIGH,
        ),
    )
    unresolved = SimpleNamespace(
        hypothesis=SimpleNamespace(time_seconds=101.0),
        assessment=SimpleNamespace(
            candidate_confidence=BoundaryConfidence.UNRESOLVED,
        ),
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

        def analyze_hypotheses(self, **kwargs):
            return (farther_high, unresolved, high)

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=_observation(boundary=100.0),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=104.0,
        incoming_start_seconds=108.0,
    )


def test_interstitial_specialist_keeps_outgoing_without_supported_analysis() -> None:
    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=108.0,
    )

    unresolved = SimpleNamespace(
        hypothesis=SimpleNamespace(time_seconds=104.0),
        assessment=SimpleNamespace(
            candidate_confidence=BoundaryConfidence.UNRESOLVED,
        ),
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

        def analyze_hypotheses(self, **kwargs):
            return (unresolved,)

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=_observation(boundary=100.0),
    )

    assert result.temporal is refined


def test_interstitial_specialist_does_not_move_outgoing_past_incoming() -> None:
    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=108.0,
    )

    too_late = SimpleNamespace(
        hypothesis=SimpleNamespace(time_seconds=109.0),
        assessment=SimpleNamespace(
            candidate_confidence=BoundaryConfidence.HIGH,
        ),
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

        def analyze_hypotheses(self, **kwargs):
            return (too_late,)

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=_observation(boundary=100.0),
    )

    assert result.temporal is refined


def test_ambiguous_fallback_preserves_existing_temporal_path() -> None:
    original = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=100.0,
    )

    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=102.5,
    )

    class TemporalResolver:
        def __init__(self):
            self.calls = []

        def resolve(self, **kwargs):
            self.calls.append(kwargs)
            return original

    class IncomingPipeline:
        def __init__(self):
            self.calls = []

        def refine(self, **kwargs):
            self.calls.append(kwargs)
            return refined

    temporal = TemporalResolver()
    incoming = IncomingPipeline()
    observation = _observation()

    result = LegacyAmbiguousSpecialist(
        temporal_resolver=temporal,
        incoming_pipeline=incoming,
    ).resolve(
        observation=observation,
    )

    assert result.temporal is refined

    assert temporal.calls == [
        {
            "selected": observation.selected,
            "profile": observation.profile,
            "window": observation.window,
            "semantic_time_seconds": 102.0,
        }
    ]

    assert incoming.calls[0]["decision"] is original
    assert incoming.calls[0]["hypotheses"] is (
        observation.routing.hypotheses
    )


def test_ambiguous_fallback_preserves_none() -> None:
    class TemporalResolver:
        def resolve(self, **kwargs):
            return None

    result = LegacyAmbiguousSpecialist(
        temporal_resolver=TemporalResolver(),
    ).resolve(
        observation=_observation(),
    )

    assert result.temporal is None


def test_fade_specialist_resolves_crossfade_from_observed_rise() -> None:
    from brynse.fade_specialist import FadeSpecialist
    from brynse.transition_evidence import (
        CrossfadeRiseEvidence,
        FadeRoutingEvidence,
    )

    observation = _observation(boundary=100.0)

    observation = TransitionSpecialistInput(
        routing=TransitionRoutingContext(
            reference_time_seconds=101.0,
            evidence=TransitionEvidenceState(),
            fade=FadeRoutingEvidence(
                rise=CrossfadeRiseEvidence(
                    time_seconds=101.5,
                    jump_ratio=3.0,
                    persistence_ratio=4.0,
                    after_global_ratio=3.0,
                ),
            ),
        ),
        selected=observation.selected,
        profile=observation.profile,
        window=observation.window,
        pcm8=observation.pcm8,
        pcm16=observation.pcm16,
    )

    result = FadeSpecialist.resolve(observation=observation)

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.CROSSFADE,
        incoming_start_seconds=100.0,
        outgoing_end_seconds=101.5,
    )


def test_fade_specialist_returns_its_selections_with_resolution() -> None:
    from brynse.fade_specialist import FadeSpecialist

    observation = _observation(boundary=100.0)

    result = FadeSpecialist.resolve(
        observation=observation,
    )

    assert result.diagnostics is not None
    assert result.diagnostics.outgoing_selection is None
    assert result.diagnostics.incoming_selection is None


def test_fade_specialist_uses_coherent_two_sided_boundary_support() -> None:
    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.fade_specialist import FadeSpecialist
    from brynse.transition_evidence import (
        CrossfadeRiseEvidence,
        FadeRoutingEvidence,
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )

    incoming_boundary = BoundaryHypothesis(
        time_seconds=101.1,
        proposals=(
            BoundaryProposal(
                time_seconds=101.1,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
    )
    outgoing_boundary = BoundaryHypothesis(
        time_seconds=101.3,
        proposals=(
            BoundaryProposal(
                time_seconds=101.3,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
    )

    original = _observation(boundary=100.0)

    observation = TransitionSpecialistInput(
        routing=TransitionRoutingContext(
            reference_time_seconds=101.0,
            evidence=TransitionEvidenceState(),
            hypotheses=(
                incoming_boundary,
                outgoing_boundary,
            ),
            fade=FadeRoutingEvidence(
                rise=CrossfadeRiseEvidence(
                    time_seconds=101.5,
                    jump_ratio=3.0,
                    persistence_ratio=4.0,
                    after_global_ratio=3.0,
                ),
                outgoing_fades=(
                    OutgoingFadeEvidence(
                        start_time_seconds=99.0,
                        floor_time_seconds=101.25,
                        rise_time_seconds=101.5,
                        start_rms=1000.0,
                        floor_rms=100.0,
                        descent_ratio=10.0,
                        descent_consistency=0.95,
                    ),
                ),
                incoming_fades=(
                    IncomingFadeEvidence(
                        start_time_seconds=101.05,
                        established_time_seconds=101.2,
                        start_rms=100.0,
                        established_rms=600.0,
                        ascent_ratio=6.0,
                        ascent_consistency=0.95,
                        persistence_ratio=5.0,
                    ),
                ),
            ),
        ),
        selected=original.selected,
        profile=original.profile,
        window=original.window,
        pcm8=original.pcm8,
        pcm16=original.pcm16,
    )

    result = FadeSpecialist.resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.CROSSFADE,
        incoming_start_seconds=101.1,
        outgoing_end_seconds=101.3,
    )


def test_fade_specialist_keeps_conservative_pair_without_two_sided_support() -> None:
    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.fade_specialist import FadeSpecialist
    from brynse.transition_evidence import (
        CrossfadeRiseEvidence,
        FadeRoutingEvidence,
        OutgoingFadeEvidence,
    )

    outgoing_boundary = BoundaryHypothesis(
        time_seconds=101.3,
        proposals=(
            BoundaryProposal(
                time_seconds=101.3,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
    )

    original = _observation(boundary=100.0)

    observation = TransitionSpecialistInput(
        routing=TransitionRoutingContext(
            reference_time_seconds=101.0,
            evidence=TransitionEvidenceState(),
            hypotheses=(outgoing_boundary,),
            fade=FadeRoutingEvidence(
                rise=CrossfadeRiseEvidence(
                    time_seconds=101.5,
                    jump_ratio=3.0,
                    persistence_ratio=4.0,
                    after_global_ratio=3.0,
                ),
                outgoing_fades=(
                    OutgoingFadeEvidence(
                        start_time_seconds=99.0,
                        floor_time_seconds=101.25,
                        rise_time_seconds=101.5,
                        start_rms=1000.0,
                        floor_rms=100.0,
                        descent_ratio=10.0,
                        descent_consistency=0.95,
                    ),
                ),
            ),
        ),
        selected=original.selected,
        profile=original.profile,
        window=original.window,
        pcm8=original.pcm8,
        pcm16=original.pcm16,
    )

    result = FadeSpecialist.resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.CROSSFADE,
        incoming_start_seconds=100.0,
        outgoing_end_seconds=101.5,
    )


def test_fade_specialist_preserves_crossfade_coherence_rule() -> None:
    from brynse.fade_specialist import FadeSpecialist
    from brynse.transition_evidence import (
        CrossfadeRiseEvidence,
        FadeRoutingEvidence,
    )

    original = _observation(boundary=100.5)

    observation = TransitionSpecialistInput(
        routing=TransitionRoutingContext(
            reference_time_seconds=101.0,
            evidence=TransitionEvidenceState(),
            fade=FadeRoutingEvidence(
                rise=CrossfadeRiseEvidence(
                    time_seconds=102.0,
                    jump_ratio=3.0,
                    persistence_ratio=4.0,
                    after_global_ratio=3.0,
                ),
            ),
        ),
        selected=original.selected,
        profile=original.profile,
        window=original.window,
        pcm8=original.pcm8,
        pcm16=original.pcm16,
    )

    result = FadeSpecialist.resolve(observation=observation)

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=100.5,
        outgoing_end_seconds=100.5,
    )


def test_fade_specialist_uses_observed_evidence_without_reobserving() -> None:
    from brynse.fade_specialist import FadeSpecialist

    observation = _observation(boundary=100.0)

    result = FadeSpecialist.resolve(observation=observation)

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=100.0,
        outgoing_end_seconds=100.0,
    )


def test_routed_transition_resolver_dispatches_to_exactly_one_specialist() -> None:
    from brynse.transition_router import TransitionKind
    from brynse.transition_specialists import RoutedTransitionResolver

    expected = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=100.0,
    )

    class Router:
        def __init__(self):
            self.contexts = []

        def classify(self, *, context):
            self.contexts.append(context)
            return TransitionKind.FADE

    class Specialist:
        def __init__(self, *, result=None):
            self.calls = []
            self.result = result

        def resolve(self, *, observation):
            self.calls.append(observation)
            return TransitionSpecialistResult(
                temporal=self.result,
            )

    router = Router()

    hard_cut = Specialist()
    fade = Specialist(result=expected)
    interstitial = Specialist()
    bounded_insert = Specialist()
    ambiguous = Specialist()

    resolver = RoutedTransitionResolver(
        router=router,
        hard_cut=hard_cut,
        fade=fade,
        interstitial=interstitial,
        bounded_insert=bounded_insert,
        ambiguous=ambiguous,
    )

    observation = _observation()

    kind, decision = resolver.resolve(
        observation=observation,
    )

    assert kind is TransitionKind.FADE
    assert decision.temporal is expected

    assert router.contexts == [observation.routing]

    assert hard_cut.calls == []
    assert fade.calls == [observation]
    assert interstitial.calls == []
    assert bounded_insert.calls == []
    assert ambiguous.calls == []


def test_routed_transition_resolver_dispatches_every_router_kind() -> None:
    from brynse.transition_router import TransitionKind
    from brynse.transition_specialists import RoutedTransitionResolver

    class Router:
        def __init__(self, kind):
            self.kind = kind

        def classify(self, *, context):
            _ = context
            return self.kind

    class Specialist:
        def __init__(self, name):
            self.name = name
            self.calls = []

        def resolve(self, *, observation):
            self.calls.append(observation)
            return TransitionSpecialistResult(
                temporal=None,
                diagnostics=self.name,
            )

    observation = _observation()

    for routed_kind, expected_name in (
        (TransitionKind.HARD_CUT, "hard-cut"),
        (TransitionKind.FADE, "fade"),
        (TransitionKind.INTERSTITIAL, "interstitial"),
        (TransitionKind.BOUNDED_INSERT, "bounded-insert"),
        (TransitionKind.AMBIGUOUS, "ambiguous"),
    ):
        specialists = {
            "hard-cut": Specialist("hard-cut"),
            "fade": Specialist("fade"),
            "interstitial": Specialist("interstitial"),
            "bounded-insert": Specialist("bounded-insert"),
            "ambiguous": Specialist("ambiguous"),
        }

        resolver = RoutedTransitionResolver(
            router=Router(routed_kind),
            hard_cut=specialists["hard-cut"],
            fade=specialists["fade"],
            interstitial=specialists["interstitial"],
            bounded_insert=specialists["bounded-insert"],
            ambiguous=specialists["ambiguous"],
        )

        kind, result = resolver.resolve(
            observation=observation,
        )

        assert kind is routed_kind
        assert result.diagnostics == expected_name

        for name, specialist in specialists.items():
            if name == expected_name:
                assert specialist.calls == [observation]
            else:
                assert specialist.calls == []


def _bounded_insert_specialist_observation(
    *,
    regions,
):
    from types import SimpleNamespace

    from brynse.transition_evidence import TransitionEvidenceState
    from brynse.transition_router import TransitionRoutingContext

    return TransitionSpecialistInput(
        routing=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=TransitionEvidenceState(),
            bounded_insert_regions=tuple(regions),
        ),
        selected=SimpleNamespace(time_seconds=100.0),
        profile=None,
        window=None,
        pcm8=None,
        pcm16=None,
    )


def _bounded_insert_test_region(
    *,
    left_time: float,
    right_time: float,
):
    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.bounded_insert import BoundedInsertRegionEvidence

    def edge(time_seconds: float) -> BoundaryHypothesis:
        basin = BoundaryProposal(
            time_seconds=time_seconds,
            source=BoundaryProposalSource.BASIN,
        )
        structural = BoundaryProposal(
            time_seconds=time_seconds,
            source=BoundaryProposalSource.STRUCTURAL,
            strength=0.2,
        )

        return BoundaryHypothesis(
            time_seconds=time_seconds,
            proposals=(basin, structural),
            evidence=(
                BoundaryEvidence(
                    proposal=basin,
                    basin_start_seconds=time_seconds - 0.5,
                    basin_recovery_seconds=time_seconds + 0.5,
                ),
            ),
        )

    return BoundedInsertRegionEvidence(
        reference_time_seconds=100.0,
        left=edge(left_time),
        right=edge(right_time),
    )


def test_bounded_insert_specialist_excludes_unique_region() -> None:
    from brynse.transition_specialists import BoundedInsertSpecialist

    observation = _bounded_insert_specialist_observation(
        regions=(
            _bounded_insert_test_region(
                left_time=100.25,
                right_time=108.75,
            ),
        ),
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.25,
        incoming_start_seconds=108.75,
    )


def test_bounded_insert_specialist_abstains_without_region() -> None:
    from brynse.transition_specialists import BoundedInsertSpecialist

    observation = _bounded_insert_specialist_observation(
        regions=(),
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal is None


def test_bounded_insert_specialist_abstains_with_multiple_regions() -> None:
    from brynse.transition_specialists import BoundedInsertSpecialist

    observation = _bounded_insert_specialist_observation(
        regions=(
            _bounded_insert_test_region(
                left_time=100.25,
                right_time=108.75,
            ),
            _bounded_insert_test_region(
                left_time=108.75,
                right_time=116.0,
            ),
        ),
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal is None


def _asymmetric_specialist_observation(region):
    from types import SimpleNamespace

    from brynse.transition_evidence import TransitionEvidenceState
    from brynse.transition_router import TransitionRoutingContext

    return TransitionSpecialistInput(
        routing=TransitionRoutingContext(
            reference_time_seconds=region.reference_time_seconds,
            evidence=TransitionEvidenceState(),
            hypotheses=(
                region.primary,
                *region.interior_hypotheses,
                region.complementary,
            ),
            bounded_insert_asymmetric_regions=(region,),
        ),
        selected=SimpleNamespace(
            time_seconds=region.reference_time_seconds,
        ),
        profile=object(),
        window=object(),
        pcm8=object(),
        pcm16=object(),
    )


def _asymmetric_specialist_primary(
    *,
    time_seconds,
    basin_start_seconds,
    basin_recovery_seconds,
):
    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )

    basin = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.BASIN,
    )
    structural = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.8,
    )

    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=(basin, structural),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_start_seconds=basin_start_seconds,
                basin_recovery_seconds=basin_recovery_seconds,
            ),
        ),
    )


def _asymmetric_structural_complement(
    *,
    time_seconds,
):
    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )

    proposal = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.3,
    )

    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=(proposal,),
    )


def _asymmetric_basin_complement(
    *,
    time_seconds,
    basin_start_seconds,
    basin_recovery_seconds,
):
    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )

    proposal = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.BASIN,
    )

    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=(proposal,),
        evidence=(
            BoundaryEvidence(
                proposal=proposal,
                basin_start_seconds=basin_start_seconds,
                basin_recovery_seconds=basin_recovery_seconds,
            ),
        ),
    )


def test_bounded_insert_specialist_resolves_ernie_h_geometry() -> None:
    from brynse.bounded_insert import (
        BoundedInsertAsymmetricRegionEvidence,
    )
    from brynse.transition_specialists import BoundedInsertSpecialist

    primary = _asymmetric_specialist_primary(
        time_seconds=1344.876,
        basin_start_seconds=1344.251,
        basin_recovery_seconds=1348.051,
    )
    complementary = _asymmetric_structural_complement(
        time_seconds=1364.451,
    )

    region = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=1363.958,
        primary=primary,
        complementary=complementary,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=_asymmetric_specialist_observation(region),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=1348.051,
        incoming_start_seconds=1364.451,
    )


def test_bounded_insert_specialist_resolves_banda_kenny_geometry() -> None:
    from brynse.bounded_insert import (
        BoundedInsertAsymmetricRegionEvidence,
    )
    from brynse.transition_specialists import BoundedInsertSpecialist

    primary = _asymmetric_specialist_primary(
        time_seconds=1778.519,
        basin_start_seconds=1773.069,
        basin_recovery_seconds=1778.769,
    )
    complementary = _asymmetric_basin_complement(
        time_seconds=1781.919,
        basin_start_seconds=1781.169,
        basin_recovery_seconds=1782.069,
    )

    region = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=1780.376,
        primary=primary,
        complementary=complementary,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=_asymmetric_specialist_observation(region),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=1778.769,
        incoming_start_seconds=1782.069,
    )


def test_bounded_insert_specialist_preserves_symmetric_support_times() -> None:
    from brynse.transition_specialists import BoundedInsertSpecialist

    observation = _bounded_insert_specialist_observation(
        regions=(
            _bounded_insert_test_region(
                left_time=100.25,
                right_time=108.75,
            ),
        ),
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.25,
        incoming_start_seconds=108.75,
    )


def _phase2_supported_edge(
    *,
    time_seconds,
    basin_start_seconds,
    basin_recovery_seconds,
    structural_strength=0.5,
):
    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )

    basin = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.BASIN,
    )
    structural = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=structural_strength,
    )

    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=(basin, structural),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_start_seconds=basin_start_seconds,
                basin_recovery_seconds=basin_recovery_seconds,
            ),
        ),
    )


def _phase2_structural_only(*, time_seconds):
    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )

    proposal = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.2,
    )

    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=(proposal,),
    )


def test_bounded_insert_phase2_jazzy_chuck_phase_aware_left_edge() -> None:
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_specialists import BoundedInsertSpecialist

    left = _phase2_supported_edge(
        time_seconds=1624.206,
        basin_start_seconds=1623.631,
        basin_recovery_seconds=1623.731,
    )
    right = _phase2_supported_edge(
        time_seconds=1635.106,
        basin_start_seconds=1634.8,
        basin_recovery_seconds=1635.3,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=1630.0,
        left=left,
        right=right,
    )

    proposal = (
        BoundedInsertSpecialist.propose_symmetric_phase_aware(
            region
        )
    )

    assert proposal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=1623.731,
        incoming_start_seconds=1635.106,
    )


def test_bounded_insert_phase2_symmetric_keeps_earlier_support() -> None:
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_specialists import BoundedInsertSpecialist

    left = _phase2_supported_edge(
        time_seconds=100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
    )
    right = _phase2_supported_edge(
        time_seconds=108.0,
        basin_start_seconds=107.5,
        basin_recovery_seconds=108.5,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=104.0,
        left=left,
        right=right,
    )

    proposal = (
        BoundedInsertSpecialist.propose_symmetric_phase_aware(
            region
        )
    )

    assert proposal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=108.0,
    )


def test_bounded_insert_phase2_earl_dj_expands_outer_extent() -> None:
    from brynse.bounded_insert import (
        BoundedInsertAsymmetricRegionEvidence,
        BoundedInsertRegionEvidence,
    )
    from brynse.transition_specialists import BoundedInsertSpecialist

    predecessor = _phase2_supported_edge(
        time_seconds=2448.978,
        basin_start_seconds=2448.3,
        basin_recovery_seconds=2449.2,
    )

    internal_a = _phase2_structural_only(
        time_seconds=2452.628,
    )
    internal_b = _phase2_structural_only(
        time_seconds=2458.128,
    )

    anchor = _phase2_supported_edge(
        time_seconds=2464.753,
        basin_start_seconds=2464.2,
        basin_recovery_seconds=2465.0,
        structural_strength=0.8,
    )

    complementary = _phase2_structural_only(
        time_seconds=2468.128,
    )

    outer_right = _phase2_supported_edge(
        time_seconds=2486.953,
        basin_start_seconds=2486.4,
        basin_recovery_seconds=2487.2,
    )

    asymmetric = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=2467.9,
        primary=anchor,
        complementary=complementary,
    )

    symmetric = BoundedInsertRegionEvidence(
        reference_time_seconds=2475.0,
        left=anchor,
        right=outer_right,
    )

    proposal = (
        BoundedInsertSpecialist.propose_asymmetric_outer_extent(
            asymmetric_region=asymmetric,
            symmetric_regions=(symmetric,),
            hypotheses=(
                predecessor,
                internal_a,
                internal_b,
                anchor,
                complementary,
                outer_right,
            ),
        )
    )

    assert proposal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=2448.978,
        incoming_start_seconds=2486.953,
    )


def test_bounded_insert_phase2_outer_extent_rejects_intervening_supported_edge() -> None:
    from brynse.bounded_insert import (
        BoundedInsertAsymmetricRegionEvidence,
        BoundedInsertRegionEvidence,
    )
    from brynse.transition_specialists import BoundedInsertSpecialist

    predecessor = _phase2_supported_edge(
        time_seconds=100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
    )

    intervening = _phase2_supported_edge(
        time_seconds=105.0,
        basin_start_seconds=104.5,
        basin_recovery_seconds=105.5,
    )

    anchor = _phase2_supported_edge(
        time_seconds=110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.8,
    )

    complementary = _phase2_structural_only(
        time_seconds=111.0,
    )

    right = _phase2_supported_edge(
        time_seconds=120.0,
        basin_start_seconds=119.5,
        basin_recovery_seconds=120.5,
    )

    asymmetric = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=110.5,
        primary=anchor,
        complementary=complementary,
    )

    symmetric = BoundedInsertRegionEvidence(
        reference_time_seconds=115.0,
        left=anchor,
        right=right,
    )

    proposal = (
        BoundedInsertSpecialist.propose_asymmetric_outer_extent(
            asymmetric_region=asymmetric,
            symmetric_regions=(symmetric,),
            hypotheses=(
                predecessor,
                intervening,
                anchor,
                complementary,
                right,
            ),
        )
    )

    # The nearest supported predecessor is the intervening edge,
    # so expansion must not jump across it to the earlier one.
    assert proposal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=105.0,
        incoming_start_seconds=120.0,
    )


def test_bounded_insert_phase2_outer_extent_rejects_nonstructural_interior() -> None:
    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.bounded_insert import (
        BoundedInsertAsymmetricRegionEvidence,
        BoundedInsertRegionEvidence,
    )
    from brynse.transition_specialists import BoundedInsertSpecialist

    predecessor = _phase2_supported_edge(
        time_seconds=100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
    )

    basin = BoundaryProposal(
        time_seconds=105.0,
        source=BoundaryProposalSource.BASIN,
    )
    basin_only = BoundaryHypothesis(
        time_seconds=105.0,
        proposals=(basin,),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_start_seconds=104.5,
                basin_recovery_seconds=105.5,
            ),
        ),
    )

    anchor = _phase2_supported_edge(
        time_seconds=110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.8,
    )

    complementary = _phase2_structural_only(
        time_seconds=111.0,
    )

    right = _phase2_supported_edge(
        time_seconds=120.0,
        basin_start_seconds=119.5,
        basin_recovery_seconds=120.5,
    )

    asymmetric = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=110.5,
        primary=anchor,
        complementary=complementary,
    )

    symmetric = BoundedInsertRegionEvidence(
        reference_time_seconds=115.0,
        left=anchor,
        right=right,
    )

    proposal = (
        BoundedInsertSpecialist.propose_asymmetric_outer_extent(
            asymmetric_region=asymmetric,
            symmetric_regions=(symmetric,),
            hypotheses=(
                predecessor,
                basin_only,
                anchor,
                complementary,
                right,
            ),
        )
    )

    assert proposal is None


def test_bounded_insert_phase2_shadow_does_not_change_resolve() -> None:
    from brynse.transition_specialists import BoundedInsertSpecialist

    observation = _bounded_insert_specialist_observation(
        regions=(
            _bounded_insert_test_region(
                left_time=100.25,
                right_time=108.75,
            ),
        ),
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.25,
        incoming_start_seconds=108.75,
    )


def test_routed_resolver_passes_bounded_insert_ownership_provenance() -> None:
    from brynse.transition_router import (
        BoundedInsertOwnership,
        TransitionKind,
    )
    from brynse.transition_specialists import RoutedTransitionResolver

    class Router:
        def classify(self, *, context):
            _ = context
            return TransitionKind.BOUNDED_INSERT

        def bounded_insert_ownership(self, *, context):
            _ = context
            return BoundedInsertOwnership.ASYMMETRIC

    class Specialist:
        def __init__(self):
            self.observation = None

        def resolve(self, *, observation):
            self.observation = observation
            return TransitionSpecialistResult(
                temporal=None,
            )

    bounded_insert = Specialist()

    resolver = RoutedTransitionResolver(
        router=Router(),
        hard_cut=Specialist(),
        fade=Specialist(),
        interstitial=Specialist(),
        bounded_insert=bounded_insert,
        ambiguous=Specialist(),
    )

    original = _observation()

    kind, result = resolver.resolve(
        observation=original,
    )

    assert kind is TransitionKind.BOUNDED_INSERT
    assert result.temporal is None

    routed = bounded_insert.observation

    assert routed is not None
    assert routed is not original
    assert (
        routed.bounded_insert_ownership
        is BoundedInsertOwnership.ASYMMETRIC
    )

    assert routed.routing is original.routing
    assert routed.selected is original.selected
    assert routed.window is original.window
    assert routed.pcm8 is original.pcm8
    assert routed.pcm16 is original.pcm16


def test_routed_resolver_keeps_generic_router_compatible() -> None:
    from brynse.transition_router import TransitionKind
    from brynse.transition_specialists import RoutedTransitionResolver

    class Router:
        def classify(self, *, context):
            _ = context
            return TransitionKind.BOUNDED_INSERT

    class Specialist:
        def __init__(self):
            self.observation = None

        def resolve(self, *, observation):
            self.observation = observation
            return TransitionSpecialistResult(
                temporal=None,
            )

    bounded_insert = Specialist()

    resolver = RoutedTransitionResolver(
        router=Router(),
        hard_cut=Specialist(),
        fade=Specialist(),
        interstitial=Specialist(),
        bounded_insert=bounded_insert,
        ambiguous=Specialist(),
    )

    original = _observation()

    resolver.resolve(
        observation=original,
    )

    assert bounded_insert.observation is original
    assert original.bounded_insert_ownership is None


def test_bounded_insert_phase2_symmetric_provenance_gets_authority() -> None:
    from dataclasses import replace

    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_router import BoundedInsertOwnership
    from brynse.transition_specialists import BoundedInsertSpecialist

    left = _phase2_supported_edge(
        time_seconds=1624.206,
        basin_start_seconds=1623.631,
        basin_recovery_seconds=1623.731,
    )
    right = _phase2_supported_edge(
        time_seconds=1635.106,
        basin_start_seconds=1634.8,
        basin_recovery_seconds=1635.3,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=1630.0,
        left=left,
        right=right,
    )

    base = _bounded_insert_specialist_observation(
        regions=(region,),
    )

    observation = replace(
        base,
        bounded_insert_ownership=BoundedInsertOwnership.SYMMETRIC,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=1623.731,
        incoming_start_seconds=1635.106,
    )


def test_bounded_insert_without_provenance_preserves_symmetric_legacy() -> None:
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_specialists import BoundedInsertSpecialist

    left = _phase2_supported_edge(
        time_seconds=1624.206,
        basin_start_seconds=1623.631,
        basin_recovery_seconds=1623.731,
    )
    right = _phase2_supported_edge(
        time_seconds=1635.106,
        basin_start_seconds=1634.8,
        basin_recovery_seconds=1635.3,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=1630.0,
        left=left,
        right=right,
    )

    observation = _bounded_insert_specialist_observation(
        regions=(region,),
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=1624.206,
        incoming_start_seconds=1635.106,
    )


def test_bounded_insert_phase2_asymmetric_provenance_gets_outer_extent() -> None:
    from dataclasses import replace

    from brynse.bounded_insert import (
        BoundedInsertAsymmetricRegionEvidence,
        BoundedInsertRegionEvidence,
    )
    from brynse.transition_router import BoundedInsertOwnership
    from brynse.transition_specialists import BoundedInsertSpecialist

    predecessor = _phase2_supported_edge(
        time_seconds=2448.978,
        basin_start_seconds=2448.3,
        basin_recovery_seconds=2449.2,
    )

    internal_a = _phase2_structural_only(
        time_seconds=2452.628,
    )
    internal_b = _phase2_structural_only(
        time_seconds=2458.128,
    )

    anchor = _phase2_supported_edge(
        time_seconds=2464.753,
        basin_start_seconds=2464.2,
        basin_recovery_seconds=2465.0,
        structural_strength=0.8,
    )

    complementary = _phase2_structural_only(
        time_seconds=2468.128,
    )

    outer_right = _phase2_supported_edge(
        time_seconds=2486.953,
        basin_start_seconds=2486.4,
        basin_recovery_seconds=2487.2,
    )

    asymmetric = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=2467.9,
        primary=anchor,
        complementary=complementary,
    )

    symmetric = BoundedInsertRegionEvidence(
        reference_time_seconds=2475.0,
        left=anchor,
        right=outer_right,
    )

    base = _bounded_insert_specialist_observation(
        regions=(symmetric,),
    )

    routing = replace(
        base.routing,
        bounded_insert_asymmetric_regions=(asymmetric,),
        hypotheses=(
            predecessor,
            internal_a,
            internal_b,
            anchor,
            complementary,
            outer_right,
        ),
    )

    observation = replace(
        base,
        routing=routing,
        bounded_insert_ownership=BoundedInsertOwnership.ASYMMETRIC,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=2448.978,
        incoming_start_seconds=2486.953,
    )


def test_bounded_insert_phase2_asymmetric_falls_back_when_outer_extent_absent() -> None:
    from dataclasses import replace

    from brynse.bounded_insert import (
        BoundedInsertAsymmetricRegionEvidence,
    )
    from brynse.transition_router import BoundedInsertOwnership
    from brynse.transition_specialists import BoundedInsertSpecialist

    primary = _phase2_supported_edge(
        time_seconds=100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.8,
    )

    complementary = _phase2_structural_only(
        time_seconds=119.0,
    )

    asymmetric = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=118.5,
        primary=primary,
        complementary=complementary,
    )

    base = _bounded_insert_specialist_observation(
        regions=(),
    )

    routing = replace(
        base.routing,
        bounded_insert_asymmetric_regions=(asymmetric,),
        hypotheses=(primary, complementary),
    )

    observation = replace(
        base,
        routing=routing,
        bounded_insert_ownership=BoundedInsertOwnership.ASYMMETRIC,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.5,
        incoming_start_seconds=119.0,
    )


def test_phase3_hard_cut_uses_recovery_when_selected_is_inside_basin() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    base = _observation(boundary=100.75)

    hypothesis = SimpleNamespace(
        time_seconds=100.5,
        basin_geometry=SimpleNamespace(
            start_seconds=99.5,
            minimum_seconds=100.5,
            recovery_seconds=101.5,
        ),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=102.0,
            hypotheses=(hypothesis,),
        ),
    )

    result = HardCutSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=101.5,
        incoming_start_seconds=101.5,
    )


def test_phase3_hard_cut_uses_recovery_when_reference_is_inside_basin() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    base = _observation(boundary=98.0)

    hypothesis = SimpleNamespace(
        time_seconds=102.5,
        basin_geometry=SimpleNamespace(
            start_seconds=101.0,
            minimum_seconds=102.5,
            recovery_seconds=103.0,
        ),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=102.0,
            hypotheses=(hypothesis,),
        ),
    )

    result = HardCutSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=102.5,
        incoming_start_seconds=102.5,
    )


def test_phase3_hard_cut_uses_reference_nearest_hypothesis_without_basins() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    base = _observation(boundary=95.0)

    farther = SimpleNamespace(
        time_seconds=97.0,
        basin_geometry=None,
    )
    nearest = SimpleNamespace(
        time_seconds=101.9,
        basin_geometry=None,
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=102.0,
            hypotheses=(farther, nearest),
        ),
    )

    result = HardCutSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=101.9,
        incoming_start_seconds=101.9,
    )


def test_phase3_hard_cut_does_not_jump_to_structural_when_unrelated_basin_exists() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    base = _observation(boundary=100.0)

    unrelated_basin = SimpleNamespace(
        time_seconds=80.0,
        basin_geometry=SimpleNamespace(
            start_seconds=79.0,
            minimum_seconds=80.0,
            recovery_seconds=81.0,
        ),
    )
    near_reference = SimpleNamespace(
        time_seconds=101.9,
        basin_geometry=None,
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=102.0,
            hypotheses=(unrelated_basin, near_reference),
        ),
    )

    result = HardCutSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=100.0,
    )


def test_phase3_interstitial_recovers_exclusion_when_supported_edges_enclose_selected() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    def supported(
        *,
        time_seconds,
        start_seconds,
        minimum_seconds,
        recovery_seconds,
    ):
        return SimpleNamespace(
            time_seconds=time_seconds,
            proposals=(
                SimpleNamespace(
                    source=BoundaryProposalSource.BASIN,
                ),
                SimpleNamespace(
                    source=BoundaryProposalSource.STRUCTURAL,
                ),
            ),
            basin_geometry=SimpleNamespace(
                start_seconds=start_seconds,
                minimum_seconds=minimum_seconds,
                recovery_seconds=recovery_seconds,
            ),
        )

    left = supported(
        time_seconds=98.0,
        start_seconds=97.5,
        minimum_seconds=98.0,
        recovery_seconds=98.5,
    )
    right = supported(
        time_seconds=102.0,
        start_seconds=101.5,
        minimum_seconds=102.0,
        recovery_seconds=102.5,
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return kwargs["decision"]

    base = _observation(boundary=100.0)

    observation = replace(
        base,
        routing=replace(
            base.routing,
            hypotheses=(left, right),
        ),
    )

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=98.0,
        incoming_start_seconds=102.5,
    )


def test_phase3_interstitial_does_not_invent_exclusion_without_enclosing_pair() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    supported = SimpleNamespace(
        time_seconds=98.0,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=97.5,
            minimum_seconds=98.0,
            recovery_seconds=98.5,
        ),
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return kwargs["decision"]

    base = _observation(boundary=100.0)

    observation = replace(
        base,
        routing=replace(
            base.routing,
            hypotheses=(supported,),
        ),
    )

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=100.0,
    )



def test_phase3_hard_cut_preserves_selected_when_reference_has_left_basin() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    base = _observation(boundary=100.0)

    hypothesis = SimpleNamespace(
        time_seconds=101.0,
        basin_geometry=SimpleNamespace(
            start_seconds=99.0,
            minimum_seconds=101.0,
            recovery_seconds=102.0,
        ),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=103.0,
            hypotheses=(hypothesis,),
        ),
    )

    result = HardCutSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=100.0,
    )


def test_phase3_interstitial_recovery_rejects_distant_supported_edge() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    def supported(time_seconds):
        return SimpleNamespace(
            time_seconds=time_seconds,
            proposals=(
                SimpleNamespace(
                    source=BoundaryProposalSource.BASIN,
                ),
                SimpleNamespace(
                    source=BoundaryProposalSource.STRUCTURAL,
                ),
            ),
            basin_geometry=SimpleNamespace(
                start_seconds=time_seconds - 0.5,
                minimum_seconds=time_seconds,
                recovery_seconds=time_seconds + 0.5,
            ),
        )

    left = supported(80.0)

    intervening = SimpleNamespace(
        time_seconds=99.0,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
        basin_geometry=None,
    )

    right = supported(102.0)

    class IncomingPipeline:
        def refine(self, **kwargs):
            return kwargs["decision"]

    base = _observation(boundary=100.0)

    observation = replace(
        base,
        routing=replace(
            base.routing,
            hypotheses=(
                left,
                intervening,
                right,
            ),
        ),
    )

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=100.0,
    )



def test_phase3_interstitial_preserves_outgoing_when_right_edge_is_closer() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    def supported(time_seconds, recovery_seconds):
        return SimpleNamespace(
            time_seconds=time_seconds,
            proposals=(
                SimpleNamespace(
                    source=BoundaryProposalSource.BASIN,
                ),
                SimpleNamespace(
                    source=BoundaryProposalSource.STRUCTURAL,
                ),
            ),
            basin_geometry=SimpleNamespace(
                start_seconds=time_seconds - 0.5,
                minimum_seconds=time_seconds,
                recovery_seconds=recovery_seconds,
            ),
        )

    left = supported(
        84.0,
        84.5,
    )
    right = supported(
        102.0,
        102.5,
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return kwargs["decision"]

    base = _observation(
        boundary=100.0,
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            hypotheses=(
                left,
                right,
            ),
        ),
    )

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=102.5,
    )
