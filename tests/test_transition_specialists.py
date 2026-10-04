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
    ambiguous = Specialist()

    resolver = RoutedTransitionResolver(
        router=router,
        hard_cut=hard_cut,
        fade=fade,
        interstitial=interstitial,
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
        (TransitionKind.AMBIGUOUS, "ambiguous"),
    ):
        specialists = {
            "hard-cut": Specialist("hard-cut"),
            "fade": Specialist("fade"),
            "interstitial": Specialist("interstitial"),
            "ambiguous": Specialist("ambiguous"),
        }

        resolver = RoutedTransitionResolver(
            router=Router(routed_kind),
            hard_cut=specialists["hard-cut"],
            fade=specialists["fade"],
            interstitial=specialists["interstitial"],
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
