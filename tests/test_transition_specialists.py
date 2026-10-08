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


def test_interstitial_shared_handoff_uses_routed_fade_hard_cut_geometry() -> None:
    from dataclasses import replace

    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.transition_evidence import (
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        IncomingFadeRoutingAssociation,
        OutgoingFadeRoutingAssociation,
        TransitionSemanticRoles,
    )

    basin = BoundaryProposal(
        time_seconds=100.2,
        source=BoundaryProposalSource.BASIN,
    )
    structural = BoundaryProposal(
        time_seconds=100.3,
        source=BoundaryProposalSource.STRUCTURAL,
    )
    shared = BoundaryHypothesis(
        time_seconds=100.25,
        proposals=(basin, structural),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_start_seconds=99.5,
                basin_recovery_seconds=101.0,
            ),
        ),
    )

    fade_assessment = FadeRoutingAssessment(
        rise_present=False,
        basin_count=1,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=OutgoingFadeEvidence(
                start_time_seconds=96.0,
                floor_time_seconds=100.0,
                rise_time_seconds=None,
                start_rms=1000.0,
                floor_rms=100.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
            hypothesis=shared,
            support_distance_seconds=0.25,
        ),
        incoming_association=IncomingFadeRoutingAssociation(
            fade=IncomingFadeEvidence(
                start_time_seconds=100.0,
                established_time_seconds=101.5,
                start_rms=100.0,
                established_rms=900.0,
                ascent_ratio=9.0,
                ascent_consistency=0.9,
                persistence_ratio=3.0,
            ),
            hypothesis=shared,
            support_distance_seconds=0.25,
        ),
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            raise AssertionError(
                "shared routed handoff must resolve before rediscovery"
            )

    base = _observation(boundary=95.0)
    observation = replace(
        base,
        routing=replace(
            base.routing,
            hypotheses=(shared,),
        ),
        fade_assessment=fade_assessment,
        semantic_roles=TransitionSemanticRoles(
            outgoing_handoff=shared,
            incoming_handoff=shared,
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


def test_interstitial_shared_handoff_uses_routed_fade_crossfade_geometry() -> None:
    from dataclasses import replace

    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.transition_evidence import (
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        IncomingFadeRoutingAssociation,
        OutgoingFadeRoutingAssociation,
        TransitionSemanticRoles,
    )

    basin = BoundaryProposal(
        time_seconds=100.2,
        source=BoundaryProposalSource.BASIN,
    )
    structural = BoundaryProposal(
        time_seconds=100.3,
        source=BoundaryProposalSource.STRUCTURAL,
    )
    shared = BoundaryHypothesis(
        time_seconds=100.25,
        proposals=(basin, structural),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_start_seconds=99.5,
                basin_recovery_seconds=101.0,
            ),
        ),
    )

    fade_assessment = FadeRoutingAssessment(
        rise_present=False,
        basin_count=1,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=OutgoingFadeEvidence(
                start_time_seconds=96.0,
                floor_time_seconds=100.3,
                rise_time_seconds=None,
                start_rms=1000.0,
                floor_rms=100.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
            hypothesis=shared,
            support_distance_seconds=0.05,
        ),
        incoming_association=IncomingFadeRoutingAssociation(
            fade=IncomingFadeEvidence(
                start_time_seconds=99.8,
                established_time_seconds=101.5,
                start_rms=100.0,
                established_rms=900.0,
                ascent_ratio=9.0,
                ascent_consistency=0.9,
                persistence_ratio=3.0,
            ),
            hypothesis=shared,
            support_distance_seconds=0.45,
        ),
    )

    base = _observation(boundary=95.0)
    observation = replace(
        base,
        fade_assessment=fade_assessment,
        semantic_roles=TransitionSemanticRoles(
            outgoing_handoff=shared,
            incoming_handoff=shared,
        ),
    )

    result = InterstitialSpecialist(
        incoming_pipeline=object(),
    ).resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.CROSSFADE,
        outgoing_end_seconds=100.3,
        incoming_start_seconds=99.8,
    )


def test_interstitial_distinct_roles_preserve_existing_pipeline() -> None:
    from dataclasses import replace

    from brynse.transition_router import TransitionSemanticRoles

    outgoing = object()
    incoming = object()

    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=108.0,
    )

    class IncomingPipeline:
        def __init__(self):
            self.calls = 0

        def refine(self, **kwargs):
            self.calls += 1
            return refined

    pipeline = IncomingPipeline()

    base = _observation(boundary=100.0)
    observation = replace(
        base,
        semantic_roles=TransitionSemanticRoles(
            outgoing_handoff=outgoing,
            incoming_handoff=incoming,
        ),
    )

    result = InterstitialSpecialist(
        incoming_pipeline=pipeline,
    ).resolve(
        observation=observation,
    )

    assert result.temporal is refined
    assert pipeline.calls == 1


def test_interstitial_complete_roles_recover_supported_exclusion() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.transition_router import (
        InterstitialRoutingAssessment,
        TransitionSemanticRoles,
    )

    outgoing = SimpleNamespace(
        time_seconds=100.0,
        basin_geometry=SimpleNamespace(
            start_seconds=99.0,
            minimum_seconds=100.0,
            recovery_seconds=101.0,
        ),
    )
    incoming = SimpleNamespace(
        time_seconds=108.0,
        basin_geometry=SimpleNamespace(
            start_seconds=107.0,
            minimum_seconds=107.5,
            recovery_seconds=107.8,
        ),
    )

    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=105.0,
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

    base = _observation(boundary=100.0)
    observation = replace(
        base,
        routing=replace(
            base.routing,
            hypotheses=(outgoing, incoming),
        ),
        interstitial_assessment=InterstitialRoutingAssessment(
            supported_handoffs=(outgoing, incoming),
        ),
        semantic_roles=TransitionSemanticRoles(
            outgoing_handoff=outgoing,
            incoming_handoff=incoming,
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
        incoming_start_seconds=108.0,
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


def test_interstitial_specialist_uses_routed_outgoing_role_instead_of_nearest_high() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.transition_router import TransitionSemanticRoles

    def hypothesis(time_seconds):
        basin = BoundaryProposal(
            time_seconds=time_seconds,
            source=BoundaryProposalSource.BASIN,
        )
        structural = BoundaryProposal(
            time_seconds=time_seconds,
            source=BoundaryProposalSource.STRUCTURAL,
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

    routed_outgoing = hypothesis(96.0)
    nearer_high = hypothesis(104.0)

    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=108.0,
    )

    def analysis(item):
        return SimpleNamespace(
            hypothesis=item,
            assessment=SimpleNamespace(
                candidate_confidence=BoundaryConfidence.HIGH,
            ),
        )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

        def analyze_hypotheses(self, **kwargs):
            return (
                analysis(routed_outgoing),
                analysis(nearer_high),
            )

    base = _observation(boundary=100.0)
    observation = replace(
        base,
        routing=replace(
            base.routing,
            hypotheses=(routed_outgoing, nearer_high),
        ),
        semantic_roles=TransitionSemanticRoles(
            outgoing_handoff=routed_outgoing,
        ),
    )

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=96.0,
        incoming_start_seconds=108.0,
    )


def test_interstitial_specialist_routed_outgoing_role_must_still_be_high() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.transition_router import TransitionSemanticRoles

    routed_outgoing = BoundaryHypothesis(
        time_seconds=96.0,
        proposals=(
            BoundaryProposal(
                time_seconds=96.0,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
    )

    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=108.0,
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

        def analyze_hypotheses(self, **kwargs):
            return (
                SimpleNamespace(
                    hypothesis=routed_outgoing,
                    assessment=SimpleNamespace(
                        candidate_confidence=BoundaryConfidence.UNRESOLVED,
                    ),
                ),
            )

    base = _observation(boundary=100.0)
    observation = replace(
        base,
        semantic_roles=TransitionSemanticRoles(
            outgoing_handoff=routed_outgoing,
        ),
    )

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=observation,
    )

    assert result.temporal is refined


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



def test_phase4_ambiguous_uses_recovery_of_spanning_structural_basin() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    original = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=1519.23,
        incoming_start_seconds=1519.23,
    )

    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=1519.23,
        incoming_start_seconds=1523.87306122297,
    )

    hypothesis = SimpleNamespace(
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                time_seconds=1513.9730612229698,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
                time_seconds=1516.62306122297,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                time_seconds=1516.9730612229698,
            ),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=1509.4730612229698,
            minimum_seconds=1516.62306122297,
            recovery_seconds=1519.0730612229697,
        ),
    )

    class TemporalResolver:
        def resolve(self, **kwargs):
            return original

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

    base = _observation(
        boundary=1519.23,
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=1520.9795918351758,
            hypotheses=(hypothesis,),
        ),
    )

    result = LegacyAmbiguousSpecialist(
        temporal_resolver=TemporalResolver(),
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=1519.0730612229697,
        incoming_start_seconds=1523.87306122297,
    )


def test_phase4_ambiguous_keeps_outgoing_without_structure_after_minimum() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    original = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=447.676,
        incoming_start_seconds=447.676,
    )

    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=447.676,
        incoming_start_seconds=448.59398,
    )

    hypothesis = SimpleNamespace(
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                time_seconds=436.6689795918642,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
                time_seconds=439.0189795918642,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                time_seconds=439.6689795918642,
            ),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=438.6689795918642,
            minimum_seconds=439.0189795918642,
            recovery_seconds=439.1689795918642,
        ),
    )

    class TemporalResolver:
        def resolve(self, **kwargs):
            return original

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

    base = _observation(
        boundary=447.676,
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=449.175510,
            hypotheses=(hypothesis,),
        ),
    )

    result = LegacyAmbiguousSpecialist(
        temporal_resolver=TemporalResolver(),
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=observation,
    )

    assert result.temporal is refined



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


def _phase6_fade_hypothesis(*, time: float):
    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )

    return BoundaryHypothesis(
        time_seconds=time,
        proposals=(
            BoundaryProposal(
                time_seconds=time,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
        evidence=(),
    )


def _phase6_replace(instance, /, **changes):
    from dataclasses import replace

    return replace(instance, **changes)


def test_fade_specialist_uses_shared_reconciled_hard_cut_edge() -> None:
    from brynse.fade_specialist import FadeSpecialist
    from brynse.transition_evidence import (
        CrossfadeRiseEvidence,
        FadeGeometryEvidence,
        FadeRoutingEvidence,
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        IncomingFadeRoutingAssociation,
        OutgoingFadeRoutingAssociation,
    )

    outgoing = OutgoingFadeEvidence(
        start_time_seconds=99.0,
        floor_time_seconds=101.0,
        rise_time_seconds=105.0,
        start_rms=1000.0,
        floor_rms=50.0,
        descent_ratio=20.0,
        descent_consistency=0.95,
    )
    incoming = IncomingFadeEvidence(
        start_time_seconds=101.0,
        established_time_seconds=104.0,
        start_rms=50.0,
        established_rms=1000.0,
        ascent_ratio=20.0,
        ascent_consistency=0.95,
        persistence_ratio=10.0,
    )

    outgoing_support = _phase6_fade_hypothesis(time=100.2)
    incoming_support = _phase6_fade_hypothesis(time=103.0)

    original = _observation(boundary=100.0)

    observation = _phase6_replace(
        original,
        routing=_phase6_replace(
            original.routing,
            reference_time_seconds=101.5,
            hypotheses=(
                outgoing_support,
                incoming_support,
            ),
            fade=FadeRoutingEvidence(
                rise=CrossfadeRiseEvidence(
                    time_seconds=105.0,
                    jump_ratio=3.0,
                    persistence_ratio=4.0,
                    after_global_ratio=3.0,
                ),
                outgoing_fades=(outgoing,),
                incoming_fades=(incoming,),
                geometries=(
                    FadeGeometryEvidence(
                        outgoing=outgoing,
                        incoming=incoming,
                    ),
                ),
            ),
        ),
        fade_assessment=FadeRoutingAssessment(
            rise_present=True,
            basin_count=0,
            geometry_count=1,
            outgoing_association=OutgoingFadeRoutingAssociation(
                fade=outgoing,
                hypothesis=outgoing_support,
                support_distance_seconds=0.8,
            ),
            incoming_association=IncomingFadeRoutingAssociation(
                fade=incoming,
                hypothesis=incoming_support,
                support_distance_seconds=2.0,
            ),
        ),
    )

    result = FadeSpecialist.resolve(observation=observation)

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=101.0,
        outgoing_end_seconds=101.0,
    )


def test_fade_specialist_uses_shared_reconciled_crossfade_edges() -> None:
    from brynse.fade_specialist import FadeSpecialist
    from brynse.transition_evidence import (
        CrossfadeRiseEvidence,
        FadeGeometryEvidence,
        FadeRoutingEvidence,
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        IncomingFadeRoutingAssociation,
        OutgoingFadeRoutingAssociation,
    )

    outgoing = OutgoingFadeEvidence(
        start_time_seconds=99.0,
        floor_time_seconds=101.4,
        rise_time_seconds=105.0,
        start_rms=1000.0,
        floor_rms=50.0,
        descent_ratio=20.0,
        descent_consistency=0.95,
    )
    incoming = IncomingFadeEvidence(
        start_time_seconds=101.0,
        established_time_seconds=104.0,
        start_rms=50.0,
        established_rms=1000.0,
        ascent_ratio=20.0,
        ascent_consistency=0.95,
        persistence_ratio=10.0,
    )

    outgoing_support = _phase6_fade_hypothesis(time=100.2)
    incoming_support = _phase6_fade_hypothesis(time=103.0)

    original = _observation(boundary=100.0)

    observation = _phase6_replace(
        original,
        routing=_phase6_replace(
            original.routing,
            reference_time_seconds=101.5,
            hypotheses=(
                outgoing_support,
                incoming_support,
            ),
            fade=FadeRoutingEvidence(
                rise=CrossfadeRiseEvidence(
                    time_seconds=105.0,
                    jump_ratio=3.0,
                    persistence_ratio=4.0,
                    after_global_ratio=3.0,
                ),
                outgoing_fades=(outgoing,),
                incoming_fades=(incoming,),
                geometries=(
                    FadeGeometryEvidence(
                        outgoing=outgoing,
                        incoming=incoming,
                    ),
                ),
            ),
        ),
        fade_assessment=FadeRoutingAssessment(
            rise_present=True,
            basin_count=0,
            geometry_count=1,
            outgoing_association=OutgoingFadeRoutingAssociation(
                fade=outgoing,
                hypothesis=outgoing_support,
                support_distance_seconds=1.2,
            ),
            incoming_association=IncomingFadeRoutingAssociation(
                fade=incoming,
                hypothesis=incoming_support,
                support_distance_seconds=2.0,
            ),
        ),
    )

    result = FadeSpecialist.resolve(observation=observation)

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.CROSSFADE,
        incoming_start_seconds=101.0,
        outgoing_end_seconds=101.4,
    )


def test_fade_specialist_uses_shared_reconciled_exclusion_edges() -> None:
    from brynse.fade_specialist import FadeSpecialist
    from brynse.transition_evidence import (
        CrossfadeRiseEvidence,
        FadeGeometryEvidence,
        FadeRoutingEvidence,
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        IncomingFadeRoutingAssociation,
        OutgoingFadeRoutingAssociation,
    )

    outgoing = OutgoingFadeEvidence(
        start_time_seconds=99.0,
        floor_time_seconds=101.0,
        rise_time_seconds=105.0,
        start_rms=1000.0,
        floor_rms=50.0,
        descent_ratio=20.0,
        descent_consistency=0.95,
    )
    incoming = IncomingFadeEvidence(
        start_time_seconds=101.4,
        established_time_seconds=104.0,
        start_rms=50.0,
        established_rms=1000.0,
        ascent_ratio=20.0,
        ascent_consistency=0.95,
        persistence_ratio=10.0,
    )

    outgoing_support = _phase6_fade_hypothesis(time=100.2)
    incoming_support = _phase6_fade_hypothesis(time=103.0)

    original = _observation(boundary=100.0)

    observation = _phase6_replace(
        original,
        routing=_phase6_replace(
            original.routing,
            reference_time_seconds=101.5,
            hypotheses=(
                outgoing_support,
                incoming_support,
            ),
            fade=FadeRoutingEvidence(
                rise=CrossfadeRiseEvidence(
                    time_seconds=105.0,
                    jump_ratio=3.0,
                    persistence_ratio=4.0,
                    after_global_ratio=3.0,
                ),
                outgoing_fades=(outgoing,),
                incoming_fades=(incoming,),
                geometries=(
                    FadeGeometryEvidence(
                        outgoing=outgoing,
                        incoming=incoming,
                    ),
                ),
            ),
        ),
        fade_assessment=FadeRoutingAssessment(
            rise_present=True,
            basin_count=0,
            geometry_count=1,
            outgoing_association=OutgoingFadeRoutingAssociation(
                fade=outgoing,
                hypothesis=outgoing_support,
                support_distance_seconds=0.8,
            ),
            incoming_association=IncomingFadeRoutingAssociation(
                fade=incoming,
                hypothesis=incoming_support,
                support_distance_seconds=1.6,
            ),
        ),
    )

    result = FadeSpecialist.resolve(observation=observation)

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        incoming_start_seconds=101.4,
        outgoing_end_seconds=101.0,
    )


def test_fade_specialist_preserves_policy_without_shared_reconciled_geometry() -> None:
    from brynse.fade_specialist import FadeSpecialist
    from brynse.transition_evidence import (
        CrossfadeRiseEvidence,
        FadeGeometryEvidence,
        FadeRoutingEvidence,
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        IncomingFadeRoutingAssociation,
        OutgoingFadeRoutingAssociation,
    )

    outgoing = OutgoingFadeEvidence(
        start_time_seconds=99.0,
        floor_time_seconds=101.0,
        rise_time_seconds=105.0,
        start_rms=1000.0,
        floor_rms=50.0,
        descent_ratio=20.0,
        descent_consistency=0.95,
    )
    selected_incoming = IncomingFadeEvidence(
        start_time_seconds=101.0,
        established_time_seconds=104.0,
        start_rms=50.0,
        established_rms=1000.0,
        ascent_ratio=20.0,
        ascent_consistency=0.95,
        persistence_ratio=10.0,
    )
    other_incoming = IncomingFadeEvidence(
        start_time_seconds=100.0,
        established_time_seconds=100.8,
        start_rms=80.0,
        established_rms=500.0,
        ascent_ratio=6.25,
        ascent_consistency=0.8,
        persistence_ratio=4.0,
    )

    outgoing_support = _phase6_fade_hypothesis(time=100.2)
    incoming_support = _phase6_fade_hypothesis(time=103.0)

    original = _observation(boundary=100.0)

    observation = _phase6_replace(
        original,
        routing=_phase6_replace(
            original.routing,
            reference_time_seconds=101.5,
            hypotheses=(
                outgoing_support,
                incoming_support,
            ),
            fade=FadeRoutingEvidence(
                rise=CrossfadeRiseEvidence(
                    time_seconds=105.0,
                    jump_ratio=3.0,
                    persistence_ratio=4.0,
                    after_global_ratio=3.0,
                ),
                outgoing_fades=(outgoing,),
                incoming_fades=(
                    selected_incoming,
                    other_incoming,
                ),
                geometries=(
                    FadeGeometryEvidence(
                        outgoing=outgoing,
                        incoming=other_incoming,
                    ),
                ),
            ),
        ),
        fade_assessment=FadeRoutingAssessment(
            rise_present=True,
            basin_count=0,
            geometry_count=1,
            outgoing_association=OutgoingFadeRoutingAssociation(
                fade=outgoing,
                hypothesis=outgoing_support,
                support_distance_seconds=0.8,
            ),
            incoming_association=IncomingFadeRoutingAssociation(
                fade=selected_incoming,
                hypothesis=incoming_support,
                support_distance_seconds=2.0,
            ),
        ),
    )

    result = FadeSpecialist.resolve(observation=observation)

    # The old reference/rise coherence rule remains authoritative when
    # Router-v2 does not provide one exact shared reconciled geometry.
    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=100.0,
        outgoing_end_seconds=100.0,
    )


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


def test_routed_resolver_exposes_exact_transition_route() -> None:
    from types import SimpleNamespace

    from brynse.transition_evidence import TransitionEvidenceState
    from brynse.transition_router import (
        TransitionKind,
        TransitionRoute,
        TransitionRoutingContext,
        TransitionSemanticRoles,
    )
    from brynse.transition_specialists import RoutedTransitionResolver

    outgoing = SimpleNamespace(time_seconds=90.0)
    incoming = SimpleNamespace(time_seconds=110.0)
    roles = TransitionSemanticRoles(
        outgoing_handoff=outgoing,
        incoming_handoff=incoming,
    )

    route = TransitionRoute(
        kind=TransitionKind.INTERSTITIAL,
        semantic_roles=roles,
    )

    class Router:
        def route(self, *, context):
            _ = context
            return route

    class Specialist:
        def resolve(self, *, observation):
            assert observation.semantic_roles is roles
            return TransitionSpecialistResult(
                temporal=TemporalSplitDecision(
                    kind=TemporalSplitKind.EXCLUSION,
                    outgoing_end_seconds=90.0,
                    incoming_start_seconds=110.0,
                )
            )

    unused = Specialist()

    resolver = RoutedTransitionResolver(
        router=Router(),
        hard_cut=unused,
        fade=unused,
        interstitial=unused,
        bounded_insert=unused,
        ambiguous=unused,
    )

    observation = TransitionSpecialistInput(
        routing=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=TransitionEvidenceState(),
        ),
        selected=SimpleNamespace(time_seconds=100.0),
        profile=SimpleNamespace(),
        window=SimpleNamespace(),
        pcm8=SimpleNamespace(),
        pcm16=SimpleNamespace(),
    )

    kind, result, resolved_route = resolver.resolve_with_route(
        observation=observation,
    )

    assert kind is TransitionKind.INTERSTITIAL
    assert result.temporal is not None
    assert resolved_route is route


def test_routed_resolver_passes_semantic_roles() -> None:
    from types import SimpleNamespace

    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.transition_router import (
        TransitionKind,
        TransitionRoute,
        TransitionRoutingContext,
        TransitionSemanticRoles,
    )
    from brynse.transition_specialists import RoutedTransitionResolver

    outgoing = BoundaryHypothesis(
        time_seconds=99.0,
        proposals=(
            BoundaryProposal(
                time_seconds=99.0,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
    )
    incoming = BoundaryHypothesis(
        time_seconds=101.0,
        proposals=(
            BoundaryProposal(
                time_seconds=101.0,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
    )

    roles = TransitionSemanticRoles(
        outgoing_handoff=outgoing,
        incoming_handoff=incoming,
    )

    class Router:
        def route(self, *, context):
            return TransitionRoute(
                kind=TransitionKind.INTERSTITIAL,
                semantic_roles=roles,
            )

        def classify(self, *, context):
            return TransitionKind.INTERSTITIAL

    class Specialist:
        observation = None

        def resolve(self, *, observation):
            self.observation = observation
            return TransitionSpecialistResult(
                temporal=None,
            )

    interstitial = Specialist()

    resolver = RoutedTransitionResolver(
        router=Router(),
        hard_cut=Specialist(),
        fade=Specialist(),
        interstitial=interstitial,
        bounded_insert=Specialist(),
        ambiguous=Specialist(),
    )

    observation = TransitionSpecialistInput(
        routing=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=SimpleNamespace(),
        ),
        selected=SimpleNamespace(time_seconds=100.0),
        profile=None,
        window=None,
        pcm8=None,
        pcm16=None,
    )

    kind, result = resolver.resolve(
        observation=observation,
    )

    assert kind is TransitionKind.INTERSTITIAL
    assert result.temporal is None
    assert interstitial.observation is not None
    assert interstitial.observation.semantic_roles is roles


def test_routed_resolver_passes_interstitial_assessment_provenance() -> None:
    from brynse.transition_router import (
        InterstitialRoutingAssessment,
        TransitionKind,
        TransitionRoute,
    )
    from brynse.transition_specialists import RoutedTransitionResolver

    assessment = InterstitialRoutingAssessment()

    class Router:
        def route(self, *, context):
            _ = context
            return TransitionRoute(
                kind=TransitionKind.INTERSTITIAL,
                interstitial_assessment=assessment,
            )

    class Specialist:
        def __init__(self):
            self.observation = None

        def resolve(self, *, observation):
            self.observation = observation
            return TransitionSpecialistResult(
                temporal=None,
            )

    interstitial = Specialist()

    resolver = RoutedTransitionResolver(
        router=Router(),
        hard_cut=Specialist(),
        fade=Specialist(),
        interstitial=interstitial,
        bounded_insert=Specialist(),
        ambiguous=Specialist(),
    )

    original = _observation()

    kind, result = resolver.resolve(
        observation=original,
    )

    assert kind is TransitionKind.INTERSTITIAL
    assert result.temporal is None

    routed = interstitial.observation

    assert routed is not None
    assert routed is not original
    assert routed.interstitial_assessment is assessment
    assert routed.routing is original.routing
    assert routed.selected is original.selected


def test_routed_resolver_passes_fade_assessment_provenance() -> None:
    from brynse.transition_router import (
        FadeRoutingAssessment,
        TransitionKind,
        TransitionRoute,
    )
    from brynse.transition_specialists import RoutedTransitionResolver

    assessment = FadeRoutingAssessment(
        rise_present=True,
        basin_count=1,
        geometry_count=0,
    )

    class Router:
        def route(self, *, context):
            _ = context
            return TransitionRoute(
                kind=TransitionKind.FADE,
                fade_assessment=assessment,
            )

    class Specialist:
        def __init__(self):
            self.observation = None

        def resolve(self, *, observation):
            self.observation = observation
            return TransitionSpecialistResult(
                temporal=None,
            )

    fade = Specialist()

    resolver = RoutedTransitionResolver(
        router=Router(),
        hard_cut=Specialist(),
        fade=fade,
        interstitial=Specialist(),
        bounded_insert=Specialist(),
        ambiguous=Specialist(),
    )

    original = _observation()

    kind, result = resolver.resolve(
        observation=original,
    )

    assert kind is TransitionKind.FADE
    assert result.temporal is None

    routed = fade.observation

    assert routed is not None
    assert routed is not original
    assert routed.fade_assessment is assessment
    assert routed.routing is original.routing
    assert routed.selected is original.selected


def test_routed_resolver_passes_hard_cut_assessment_provenance() -> None:
    from brynse.transition_router import (
        HardCutRoutingAssessment,
        TransitionKind,
        TransitionRoute,
    )
    from brynse.transition_specialists import RoutedTransitionResolver

    assessment = HardCutRoutingAssessment(
        local_discontinuity_complete=True,
        fade_rise_present=False,
    )

    class Router:
        def route(self, *, context):
            _ = context
            return TransitionRoute(
                kind=TransitionKind.HARD_CUT,
                hard_cut=assessment,
            )

    class Specialist:
        def __init__(self):
            self.observation = None

        def resolve(self, *, observation):
            self.observation = observation
            return TransitionSpecialistResult(
                temporal=None,
            )

    hard_cut = Specialist()

    resolver = RoutedTransitionResolver(
        router=Router(),
        hard_cut=hard_cut,
        fade=Specialist(),
        interstitial=Specialist(),
        bounded_insert=Specialist(),
        ambiguous=Specialist(),
    )

    original = _observation()

    kind, result = resolver.resolve(
        observation=original,
    )

    assert kind is TransitionKind.HARD_CUT
    assert result.temporal is None

    routed = hard_cut.observation

    assert routed is not None
    assert routed is not original
    assert routed.hard_cut_assessment is assessment

    assert routed.routing is original.routing
    assert routed.selected is original.selected
    assert routed.window is original.window
    assert routed.pcm8 is original.pcm8
    assert routed.pcm16 is original.pcm16


def test_routed_resolver_consumes_route_provenance_without_reclassification() -> None:
    from brynse.transition_router import (
        BoundedInsertOwnership,
        TransitionKind,
        TransitionRoute,
    )
    from brynse.transition_specialists import RoutedTransitionResolver

    class Router:
        def __init__(self):
            self.route_calls = []
            self.classify_calls = []
            self.ownership_calls = []

        def route(self, *, context):
            self.route_calls.append(context)
            return TransitionRoute(
                kind=TransitionKind.BOUNDED_INSERT,
                bounded_insert_ownership=(
                    BoundedInsertOwnership.ASYMMETRIC
                ),
            )

        def classify(self, *, context):
            self.classify_calls.append(context)
            raise AssertionError(
                "route-aware resolver must not call classify()"
            )

        def bounded_insert_ownership(self, *, context):
            self.ownership_calls.append(context)
            raise AssertionError(
                "route already carries bounded-insert provenance"
            )

    class Specialist:
        def __init__(self):
            self.observation = None

        def resolve(self, *, observation):
            self.observation = observation
            return TransitionSpecialistResult(
                temporal=None,
            )

    router = Router()
    bounded_insert = Specialist()

    resolver = RoutedTransitionResolver(
        router=router,
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

    assert router.route_calls == [original.routing]
    assert router.classify_calls == []
    assert router.ownership_calls == []

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


def test_routed_resolver_passes_bounded_insert_routing_provenance() -> None:
    from brynse.transition_router import (
        BoundedInsertAsymmetricRoutingAssessment,
        BoundedInsertOwnership,
        BoundedInsertRoutingAssessment,
        BoundedInsertRoutingProvenance,
        TransitionKind,
        TransitionRoute,
    )
    from brynse.transition_specialists import RoutedTransitionResolver

    provenance = BoundedInsertRoutingProvenance(
        ownership=BoundedInsertOwnership.ASYMMETRIC,
        symmetric_assessment=BoundedInsertRoutingAssessment(
            candidate=False,
            related_region_count=0,
            unique_region=False,
        ),
        asymmetric_assessment=BoundedInsertAsymmetricRoutingAssessment(
            candidate=True,
            related_region_count=1,
            unique_region=True,
            primary_is_context_structural_peak=True,
        ),
    )

    class Router:
        def route(self, *, context):
            _ = context
            return TransitionRoute(
                kind=TransitionKind.BOUNDED_INSERT,
                bounded_insert_ownership=provenance.ownership,
                bounded_insert=provenance,
            )

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
    assert routed.bounded_insert_assessment is provenance
    assert routed.routing is original.routing
    assert routed.selected is original.selected


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


def test_bounded_insert_routing_provenance_owns_symmetric_region_identity() -> None:
    from dataclasses import replace

    from brynse.transition_router import (
        BoundedInsertAsymmetricRoutingAssessment,
        BoundedInsertOwnership,
        BoundedInsertRoutingAssessment,
        BoundedInsertRoutingProvenance,
    )
    from brynse.transition_specialists import BoundedInsertSpecialist

    owned = _bounded_insert_test_region(
        left_time=100.0,
        right_time=108.0,
    )
    unrelated = _bounded_insert_test_region(
        left_time=200.0,
        right_time=208.0,
    )

    base = _bounded_insert_specialist_observation(
        regions=(unrelated,),
    )

    provenance = BoundedInsertRoutingProvenance(
        ownership=BoundedInsertOwnership.SYMMETRIC,
        symmetric_assessment=BoundedInsertRoutingAssessment(
            candidate=True,
            related_region_count=1,
            unique_region=True,
        ),
        asymmetric_assessment=BoundedInsertAsymmetricRoutingAssessment(
            candidate=False,
            related_region_count=0,
            unique_region=False,
        ),
        symmetric_region=owned,
    )

    observation = replace(
        base,
        bounded_insert_ownership=BoundedInsertOwnership.SYMMETRIC,
        bounded_insert_assessment=provenance,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=108.0,
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


def test_bounded_insert_routed_symmetric_roles_prevent_predecessor_rediscovery() -> None:
    from dataclasses import replace

    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_router import (
        BoundedInsertOwnership,
        TransitionSemanticRoles,
    )
    from brynse.transition_specialists import BoundedInsertSpecialist

    predecessor = _phase2_supported_edge(
        time_seconds=90.0,
        basin_start_seconds=89.5,
        basin_recovery_seconds=90.5,
        structural_strength=0.9,
    )
    bridge = _phase2_structural_only(
        time_seconds=95.0,
    )
    left = _phase2_supported_edge(
        time_seconds=100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.3,
    )
    right = _phase2_supported_edge(
        time_seconds=110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.2,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=105.0,
        left=left,
        right=right,
    )

    base = _bounded_insert_specialist_observation(
        regions=(region,),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=105.0,
            hypotheses=(predecessor, bridge, left, right),
            bounded_insert_regions=(region,),
        ),
        bounded_insert_ownership=BoundedInsertOwnership.SYMMETRIC,
        semantic_roles=TransitionSemanticRoles(
            inserted_region=region,
            incoming_handoff=right,
        ),
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=110.0,
    )


def test_bounded_insert_routed_symmetric_roles_use_assigned_predecessor() -> None:
    from dataclasses import replace

    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_router import (
        BoundedInsertOwnership,
        TransitionSemanticRoles,
    )
    from brynse.transition_specialists import BoundedInsertSpecialist

    predecessor = _phase2_supported_edge(
        time_seconds=90.0,
        basin_start_seconds=89.5,
        basin_recovery_seconds=90.5,
        structural_strength=0.9,
    )
    bridge = _phase2_structural_only(
        time_seconds=95.0,
    )
    left = _phase2_supported_edge(
        time_seconds=100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.3,
    )
    right = _phase2_supported_edge(
        time_seconds=110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.2,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=105.0,
        left=left,
        right=right,
    )

    base = _bounded_insert_specialist_observation(
        regions=(region,),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=105.0,
            hypotheses=(predecessor, bridge, left, right),
            bounded_insert_regions=(region,),
        ),
        bounded_insert_ownership=BoundedInsertOwnership.SYMMETRIC,
        semantic_roles=TransitionSemanticRoles(
            outgoing_handoff=predecessor,
            inserted_region=region,
            incoming_handoff=right,
        ),
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=90.0,
        incoming_start_seconds=110.0,
    )


def test_bounded_insert_asymmetric_outer_extent_consumes_routed_roles() -> None:
    from brynse.bounded_insert import (
        BoundedInsertAsymmetricRegionEvidence,
    )
    from brynse.transition_router import TransitionSemanticRoles
    from brynse.transition_specialists import BoundedInsertSpecialist

    predecessor = _phase2_supported_edge(
        time_seconds=90.0,
        basin_start_seconds=89.5,
        basin_recovery_seconds=90.5,
        structural_strength=0.5,
    )
    primary = _asymmetric_specialist_primary(
        time_seconds=100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
    )
    complementary = _asymmetric_structural_complement(
        time_seconds=105.0,
    )
    outer_right = _phase2_supported_edge(
        time_seconds=110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.2,
    )

    asymmetric = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=105.0,
        primary=primary,
        complementary=complementary,
    )

    roles = TransitionSemanticRoles(
        outgoing_handoff=predecessor,
        inserted_region=asymmetric,
        incoming_handoff=outer_right,
    )

    proposal = BoundedInsertSpecialist.propose_asymmetric_outer_extent(
        asymmetric_region=asymmetric,
        symmetric_regions=(),
        hypotheses=(
            predecessor,
            primary,
            complementary,
            outer_right,
        ),
        semantic_roles=roles,
    )

    assert proposal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=90.0,
        incoming_start_seconds=110.0,
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



def test_phase4_hard_cut_uses_unique_basin_fully_between_selected_and_reference() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    base = _observation(boundary=100.0)

    transition = SimpleNamespace(
        time_seconds=105.0,
        basin_geometry=SimpleNamespace(
            start_seconds=103.0,
            minimum_seconds=105.0,
            recovery_seconds=106.0,
        ),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=107.0,
            hypotheses=(transition,),
        ),
    )

    result = HardCutSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=106.0,
        incoming_start_seconds=106.0,
    )


def test_phase4_hard_cut_does_not_choose_between_multiple_enclosed_basins() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    base = _observation(boundary=100.0)

    first = SimpleNamespace(
        time_seconds=103.0,
        basin_geometry=SimpleNamespace(
            start_seconds=102.0,
            minimum_seconds=103.0,
            recovery_seconds=104.0,
        ),
    )

    second = SimpleNamespace(
        time_seconds=106.0,
        basin_geometry=SimpleNamespace(
            start_seconds=105.0,
            minimum_seconds=106.0,
            recovery_seconds=107.0,
        ),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=108.0,
            hypotheses=(first, second),
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



def test_phase4_interstitial_uses_start_of_dominant_incoming_basin() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    incoming_edge = SimpleNamespace(
        time_seconds=101.0,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
                strength=None,
                time_seconds=100.0,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                strength=1.0,
                time_seconds=100.5,
            ),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=99.0,
            minimum_seconds=100.0,
            recovery_seconds=101.0,
        ),
    )

    earlier = SimpleNamespace(
        time_seconds=92.0,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                strength=0.4,
            ),
        ),
        basin_geometry=None,
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=90.0,
                incoming_start_seconds=101.0,
            )

    base = _observation(boundary=95.0)

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=100.5,
            hypotheses=(
                earlier,
                incoming_edge,
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
        outgoing_end_seconds=99.0,
        incoming_start_seconds=101.0,
    )





def test_phase4_interstitial_uses_minimum_when_peak_is_at_recovery() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    earlier = SimpleNamespace(
        time_seconds=123.333061224491,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                strength=0.18796907145190353,
                time_seconds=123.333061224491,
            ),
        ),
        basin_geometry=None,
    )

    incoming_edge = SimpleNamespace(
        time_seconds=128.208061224491,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
                strength=None,
                time_seconds=128.083061224491,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                strength=0.19455395654277668,
                time_seconds=128.333061224491,
            ),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=124.233061224491,
            minimum_seconds=128.083061224491,
            recovery_seconds=128.333061224491,
        ),
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=122.308,
                incoming_start_seconds=128.333061224491,
            )

    base = _observation(
        boundary=122.308,
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=128.339591836736,
            hypotheses=(
                earlier,
                incoming_edge,
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
        outgoing_end_seconds=128.083061224491,
        incoming_start_seconds=128.333061224491,
    )



def test_phase4_interstitial_preserves_exclusion_when_incoming_is_not_context_peak() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    dominant_elsewhere = SimpleNamespace(
        time_seconds=92.0,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                strength=1.0,
            ),
        ),
        basin_geometry=None,
    )

    incoming_edge = SimpleNamespace(
        time_seconds=101.0,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
                strength=None,
                time_seconds=100.0,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                strength=0.4,
                time_seconds=100.5,
            ),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=99.0,
            minimum_seconds=100.0,
            recovery_seconds=101.0,
        ),
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=90.0,
                incoming_start_seconds=101.0,
            )

    base = _observation(boundary=95.0)

    observation = replace(
        base,
        routing=replace(
            base.routing,
            hypotheses=(
                dominant_elsewhere,
                incoming_edge,
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
        outgoing_end_seconds=90.0,
        incoming_start_seconds=101.0,
    )



def test_phase4_interstitial_preserves_exclusion_before_incoming_basin_proposal() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    incoming_edge = SimpleNamespace(
        time_seconds=100.0,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
                strength=None,
                time_seconds=100.0,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                strength=1.0,
                time_seconds=99.5,
            ),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=99.0,
            minimum_seconds=100.0,
            recovery_seconds=101.0,
        ),
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=95.0,
                incoming_start_seconds=101.0,
            )

    base = _observation(boundary=95.0)

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=99.75,
            hypotheses=(incoming_edge,),
        ),
    )

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=95.0,
        incoming_start_seconds=101.0,
    )



def test_phase4_asymmetric_fallback_uses_trailing_basin_proposal() -> None:
    from dataclasses import replace

    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.bounded_insert import BoundedInsertAsymmetricRegionEvidence
    from brynse.transition_router import BoundedInsertOwnership
    from brynse.transition_specialists import BoundedInsertSpecialist

    structural = BoundaryProposal(
        time_seconds=100.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.8,
    )
    basin = BoundaryProposal(
        time_seconds=101.0,
        source=BoundaryProposalSource.BASIN,
    )

    primary = BoundaryHypothesis(
        time_seconds=100.5,
        proposals=(
            structural,
            basin,
        ),
        evidence=(
            BoundaryEvidence(
                proposal=structural,
                structural_novelty=0.8,
            ),
            BoundaryEvidence(
                proposal=basin,
                basin_start_seconds=99.5,
                basin_recovery_seconds=104.0,
            ),
        ),
    )

    complementary = _phase2_structural_only(
        time_seconds=119.0,
    )

    asymmetric = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=118.0,
        primary=primary,
        complementary=complementary,
    )

    base = _bounded_insert_specialist_observation(
        regions=(),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            hypotheses=(
                primary,
                complementary,
            ),
            bounded_insert_asymmetric_regions=(
                asymmetric,
            ),
        ),
        bounded_insert_ownership=BoundedInsertOwnership.ASYMMETRIC,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=101.0,
        incoming_start_seconds=119.0,
    )


def test_phase4_asymmetric_fallback_preserves_recovery_when_basin_leads() -> None:
    from dataclasses import replace

    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.bounded_insert import BoundedInsertAsymmetricRegionEvidence
    from brynse.transition_router import BoundedInsertOwnership
    from brynse.transition_specialists import BoundedInsertSpecialist

    basin = BoundaryProposal(
        time_seconds=100.0,
        source=BoundaryProposalSource.BASIN,
    )
    structural = BoundaryProposal(
        time_seconds=101.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.8,
    )

    primary = BoundaryHypothesis(
        time_seconds=100.5,
        proposals=(
            basin,
            structural,
        ),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_start_seconds=99.5,
                basin_recovery_seconds=104.0,
            ),
            BoundaryEvidence(
                proposal=structural,
                structural_novelty=0.8,
            ),
        ),
    )

    complementary = _phase2_structural_only(
        time_seconds=119.0,
    )

    asymmetric = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=118.0,
        primary=primary,
        complementary=complementary,
    )

    base = _bounded_insert_specialist_observation(
        regions=(),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            hypotheses=(
                primary,
                complementary,
            ),
            bounded_insert_asymmetric_regions=(
                asymmetric,
            ),
        ),
        bounded_insert_ownership=BoundedInsertOwnership.ASYMMETRIC,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=104.0,
        incoming_start_seconds=119.0,
    )




def test_phase4_kinkajous_symmetric_ownership_excludes_terminal_insert() -> None:
    from dataclasses import replace

    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_router import BoundedInsertOwnership
    from brynse.transition_specialists import BoundedInsertSpecialist

    left = _phase2_supported_edge(
        time_seconds=2832.6468367310245,
        basin_start_seconds=2832.171836731024,
        basin_recovery_seconds=2832.9718367310243,
        structural_strength=0.37617136803949863,
    )
    interior = _phase2_structural_only(
        time_seconds=2837.071836731024,
    )
    right = _phase2_supported_edge(
        time_seconds=2849.421836731024,
        basin_start_seconds=2849.271836731024,
        basin_recovery_seconds=2849.671836731024,
        structural_strength=0.799025627372707,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=2848.07836734323,
        left=left,
        right=right,
        interior_hypotheses=(interior,),
    )

    base = _bounded_insert_specialist_observation(
        regions=(region,),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=2848.07836734323,
            hypotheses=(
                left,
                interior,
                right,
            ),
            bounded_insert_regions=(region,),
        ),
        bounded_insert_ownership=BoundedInsertOwnership.SYMMETRIC,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=2832.6468367310245,
        incoming_start_seconds=2849.421836731024,
    )



def test_phase4_symmetric_uses_later_basin_containing_reference() -> None:
    from dataclasses import replace

    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_router import BoundedInsertOwnership
    from brynse.transition_specialists import BoundedInsertSpecialist

    structural = BoundaryProposal(
        time_seconds=99.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.8,
    )
    first_basin = BoundaryProposal(
        time_seconds=100.0,
        source=BoundaryProposalSource.BASIN,
    )
    later_basin = BoundaryProposal(
        time_seconds=101.0,
        source=BoundaryProposalSource.BASIN,
    )

    left = BoundaryHypothesis(
        time_seconds=100.0,
        proposals=(
            structural,
            first_basin,
            later_basin,
        ),
        evidence=(
            BoundaryEvidence(
                proposal=structural,
                structural_novelty=0.8,
            ),
            BoundaryEvidence(
                proposal=first_basin,
                basin_start_seconds=99.5,
                basin_recovery_seconds=100.5,
            ),
            BoundaryEvidence(
                proposal=later_basin,
                basin_start_seconds=100.75,
                basin_recovery_seconds=104.0,
            ),
        ),
    )

    right = _phase2_supported_edge(
        time_seconds=110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=103.0,
        left=left,
        right=right,
    )

    base = _bounded_insert_specialist_observation(
        regions=(region,),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=103.0,
            hypotheses=(
                left,
                right,
            ),
            bounded_insert_regions=(region,),
        ),
        bounded_insert_ownership=BoundedInsertOwnership.SYMMETRIC,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=104.0,
        incoming_start_seconds=110.0,
    )


def test_phase4_symmetric_preserves_support_when_reference_misses_later_basin() -> None:
    from dataclasses import replace

    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_router import BoundedInsertOwnership
    from brynse.transition_specialists import BoundedInsertSpecialist

    structural = BoundaryProposal(
        time_seconds=99.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.8,
    )
    first_basin = BoundaryProposal(
        time_seconds=100.0,
        source=BoundaryProposalSource.BASIN,
    )
    later_basin = BoundaryProposal(
        time_seconds=102.0,
        source=BoundaryProposalSource.BASIN,
    )

    left = BoundaryHypothesis(
        time_seconds=100.0,
        proposals=(
            structural,
            first_basin,
            later_basin,
        ),
        evidence=(
            BoundaryEvidence(
                proposal=structural,
                structural_novelty=0.8,
            ),
            BoundaryEvidence(
                proposal=first_basin,
                basin_start_seconds=99.5,
                basin_recovery_seconds=100.5,
            ),
            BoundaryEvidence(
                proposal=later_basin,
                basin_start_seconds=101.5,
                basin_recovery_seconds=104.0,
            ),
        ),
    )

    right = _phase2_supported_edge(
        time_seconds=110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=105.0,
        left=left,
        right=right,
    )

    base = _bounded_insert_specialist_observation(
        regions=(region,),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=105.0,
            hypotheses=(
                left,
                right,
            ),
            bounded_insert_regions=(region,),
        ),
        bounded_insert_ownership=BoundedInsertOwnership.SYMMETRIC,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=110.0,
    )



# ---------------------------------------------------------------------------
# Phase 4e: production replay refinements
# ---------------------------------------------------------------------------


def test_phase4e_hard_cut_keeps_basin_only_reference_time() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    base = _observation(boundary=203.0)

    basin_only = SimpleNamespace(
        time_seconds=205.443,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
            ),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=204.693,
            minimum_seconds=205.443,
            recovery_seconds=208.593,
        ),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=204.800,
            hypotheses=(basin_only,),
        ),
    )

    result = HardCutSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=204.800,
        incoming_start_seconds=204.800,
    )


def test_phase4e_hard_cut_completes_supported_selected_basin() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    base = _observation(boundary=938.337)

    supported = SimpleNamespace(
        time_seconds=938.581,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=938.231,
            minimum_seconds=938.581,
            recovery_seconds=938.831,
        ),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=939.337,
            hypotheses=(supported,),
        ),
    )

    result = HardCutSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=938.831,
        incoming_start_seconds=938.831,
    )


def test_phase4e_bounded_insert_collapses_reference_structural_bridge() -> None:
    from dataclasses import replace

    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_router import BoundedInsertOwnership
    from brynse.transition_specialists import BoundedInsertSpecialist

    left = _phase2_supported_edge(
        time_seconds=492.139,
        basin_start_seconds=491.189,
        basin_recovery_seconds=495.789,
    )
    bridge = _phase2_structural_only(
        time_seconds=495.589,
    )
    right = _phase2_supported_edge(
        time_seconds=499.914,
        basin_start_seconds=499.489,
        basin_recovery_seconds=500.789,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=495.595,
        left=left,
        right=right,
        interior_hypotheses=(bridge,),
    )

    base = _bounded_insert_specialist_observation(
        regions=(region,),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=495.595,
            hypotheses=(
                left,
                bridge,
                right,
            ),
            bounded_insert_regions=(region,),
        ),
        bounded_insert_ownership=BoundedInsertOwnership.SYMMETRIC,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=495.589,
        incoming_start_seconds=495.589,
    )


def test_phase4e_bounded_insert_extends_symmetric_structural_chain() -> None:
    from dataclasses import replace

    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_router import BoundedInsertOwnership
    from brynse.transition_specialists import BoundedInsertSpecialist

    predecessor = _phase2_supported_edge(
        time_seconds=2231.217,
        basin_start_seconds=2230.892,
        basin_recovery_seconds=2232.692,
    )
    structural_a = _phase2_structural_only(
        time_seconds=2235.192,
    )
    structural_b = _phase2_structural_only(
        time_seconds=2240.192,
    )
    anchor = _phase2_supported_edge(
        time_seconds=2247.967,
        basin_start_seconds=2246.192,
        basin_recovery_seconds=2248.092,
    )
    right = _phase2_supported_edge(
        time_seconds=2264.517,
        basin_start_seconds=2264.092,
        basin_recovery_seconds=2264.592,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=2248.699,
        left=anchor,
        right=right,
    )

    base = _bounded_insert_specialist_observation(
        regions=(region,),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=2248.699,
            hypotheses=(
                predecessor,
                structural_a,
                structural_b,
                anchor,
                right,
            ),
            bounded_insert_regions=(region,),
        ),
        bounded_insert_ownership=BoundedInsertOwnership.SYMMETRIC,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=2231.217,
        incoming_start_seconds=2264.517,
    )


def test_phase4e_bounded_insert_fade_blocks_asymmetric_outer_extent() -> None:
    from dataclasses import replace

    from brynse.bounded_insert import (
        BoundedInsertAsymmetricRegionEvidence,
        BoundedInsertRegionEvidence,
    )
    from brynse.transition_evidence import (
        CrossfadeRiseEvidence,
        FadeRoutingEvidence,
    )
    from brynse.transition_router import BoundedInsertOwnership
    from brynse.transition_specialists import BoundedInsertSpecialist

    predecessor = _phase2_supported_edge(
        time_seconds=90.0,
        basin_start_seconds=89.5,
        basin_recovery_seconds=90.5,
    )
    structural = _phase2_structural_only(
        time_seconds=95.0,
    )
    primary = _phase2_supported_edge(
        time_seconds=100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.8,
    )
    complementary = _phase2_structural_only(
        time_seconds=119.0,
    )
    outer_right = _phase2_supported_edge(
        time_seconds=120.0,
        basin_start_seconds=119.5,
        basin_recovery_seconds=120.5,
    )

    asymmetric = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=110.0,
        primary=primary,
        complementary=complementary,
    )
    symmetric = BoundedInsertRegionEvidence(
        reference_time_seconds=110.0,
        left=primary,
        right=outer_right,
    )

    base = _bounded_insert_specialist_observation(
        regions=(symmetric,),
    )

    routing = replace(
        base.routing,
        reference_time_seconds=110.0,
        hypotheses=(
            predecessor,
            structural,
            primary,
            complementary,
            outer_right,
        ),
        bounded_insert_regions=(symmetric,),
        bounded_insert_asymmetric_regions=(asymmetric,),
        fade=FadeRoutingEvidence(
            rise=CrossfadeRiseEvidence(
                time_seconds=100.0,
                jump_ratio=2.0,
                persistence_ratio=2.0,
                after_global_ratio=2.0,
            ),
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

    # Without the fade guard, outer extent would claim predecessor=90.
    # With fade ownership evidence, the existing local asymmetric fallback
    # remains responsible for the bounded-insert geometry.
    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.5,
        incoming_start_seconds=119.0,
    )



def test_phase4e_hard_cut_keeps_basin_only_reference_when_selected_is_inside_same_basin() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    # Lowjazz-like alternate execution path:
    # selected and semantic reference both land before the minimum of the
    # same basin-only geometry.  A basin without structural support must
    # not drag the cut forward to its minimum.
    base = _observation(boundary=204.750)

    basin_only = SimpleNamespace(
        time_seconds=205.443,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
            ),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=204.693,
            minimum_seconds=205.443,
            recovery_seconds=208.593,
        ),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=204.800,
            hypotheses=(basin_only,),
        ),
    )

    result = HardCutSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=204.800,
        incoming_start_seconds=204.800,
    )



# ---------------------------------------------------------------------------
# Phase 4f: asymmetric pre-basin structural extent
# ---------------------------------------------------------------------------


def _phase4f_asymmetric_observation(
    *,
    incoming_established_seconds,
    incoming_start_seconds=98.8,
):
    from dataclasses import replace

    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.bounded_insert import (
        BoundedInsertAsymmetricRegionEvidence,
    )
    from brynse.transition_evidence import (
        FadeRoutingEvidence,
        IncomingFadeEvidence,
    )
    from brynse.transition_router import (
        BoundedInsertOwnership,
    )

    structural = BoundaryProposal(
        time_seconds=98.7,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.48,
    )
    basin = BoundaryProposal(
        time_seconds=100.0,
        source=BoundaryProposalSource.BASIN,
    )

    primary = BoundaryHypothesis(
        time_seconds=100.0,
        proposals=(
            structural,
            basin,
        ),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_depth=0.2,
                basin_start_seconds=99.5,
                basin_recovery_seconds=100.5,
            ),
        ),
    )

    complementary_proposal = BoundaryProposal(
        time_seconds=103.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.2,
    )
    complementary = BoundaryHypothesis(
        time_seconds=103.0,
        proposals=(complementary_proposal,),
    )

    region = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=102.0,
        primary=primary,
        complementary=complementary,
    )

    base = _asymmetric_specialist_observation(
        region
    )

    return replace(
        base,
        routing=replace(
            base.routing,
            fade=FadeRoutingEvidence(
                incoming_fades=(
                    IncomingFadeEvidence(
                        start_time_seconds=(
                            incoming_start_seconds
                        ),
                        established_time_seconds=(
                            incoming_established_seconds
                        ),
                        start_rms=100.0,
                        established_rms=500.0,
                        ascent_ratio=5.0,
                        ascent_consistency=0.5,
                        persistence_ratio=3.0,
                    ),
                ),
            ),
        ),
        bounded_insert_ownership=(
            BoundedInsertOwnership.ASYMMETRIC
        ),
    )


def test_phase4f_asymmetric_pre_basin_structural_owns_outer_left_extent() -> None:
    from brynse.transition_specialists import (
        BoundedInsertSpecialist,
    )

    observation = _phase4f_asymmetric_observation(
        incoming_established_seconds=99.2,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=98.7,
        incoming_start_seconds=103.0,
    )


def test_phase4f_asymmetric_pre_basin_structural_requires_local_incoming_phase() -> None:
    from brynse.transition_specialists import (
        BoundedInsertSpecialist,
    )

    observation = _phase4f_asymmetric_observation(
        incoming_start_seconds=90.0,
        incoming_established_seconds=91.0,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.5,
        incoming_start_seconds=103.0,
    )


def test_phase4f_compatibility_fallback_does_not_use_incoming_phase() -> None:
    from dataclasses import replace

    from brynse.transition_specialists import (
        BoundedInsertSpecialist,
    )

    observation = _phase4f_asymmetric_observation(
        incoming_established_seconds=99.2,
    )

    # Remove Router provenance.  This exercises the compatibility
    # path, which deliberately calls the fallback without fade
    # evidence and must preserve its historical result.
    observation = replace(
        observation,
        bounded_insert_ownership=None,
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.5,
        incoming_start_seconds=103.0,
    )


# ---------------------------------------------------------------------------
# Phase 4g: failed direct-transition right-basin recovery
# ---------------------------------------------------------------------------


def _phase4g_right_basin_observation(
    *,
    boundary=100.0,
    reference_inside_right_basin=True,
):
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    base = _observation(boundary=boundary)

    right_edge = SimpleNamespace(
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=101.5,
            minimum_seconds=102.0,
            recovery_seconds=103.0,
        ),
    )

    region = SimpleNamespace(
        reference_inside_right_basin=reference_inside_right_basin,
        right_edge=right_edge,
    )

    return replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=102.25,
            hypotheses=(),
            bounded_insert_regions=(region,),
        ),
    )


def test_phase4g_interstitial_failed_hard_cut_uses_supported_right_basin() -> None:
    from brynse.transition_specialists import InterstitialSpecialist

    class IncomingPipeline:
        def refine(self, **kwargs):
            return kwargs["decision"]

    observation = _phase4g_right_basin_observation()

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=102.0,
        incoming_start_seconds=102.0,
    )


def test_phase4g_ambiguous_failed_hard_cut_uses_supported_right_basin() -> None:
    from brynse.transition_specialists import LegacyAmbiguousSpecialist

    class TemporalResolver:
        def resolve(self, **kwargs):
            return TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                outgoing_end_seconds=100.0,
                incoming_start_seconds=100.0,
            )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return kwargs["decision"]

    observation = _phase4g_right_basin_observation()

    result = LegacyAmbiguousSpecialist(
        temporal_resolver=TemporalResolver(),
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=102.0,
        incoming_start_seconds=102.0,
    )


def test_phase4g_right_basin_recovery_requires_reference_ownership() -> None:
    from brynse.transition_specialists import InterstitialSpecialist

    class IncomingPipeline:
        def refine(self, **kwargs):
            return kwargs["decision"]

    observation = _phase4g_right_basin_observation(
        reference_inside_right_basin=False,
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


# ---------------------------------------------------------------------------
# Phase 4h: interstitial terminal-fade outgoing recovery
# ---------------------------------------------------------------------------


def _phase4h_larry_observation(
    *,
    reference_inside_right=True,
    left_peak=True,
    right_peak=False,
    fade_starts_before_outgoing=True,
):
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource

    base = _observation(boundary=100.0)

    peak_strength = 0.9

    left_strength = peak_strength if left_peak else 0.4
    right_strength = peak_strength if right_peak else 0.5

    left = SimpleNamespace(
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
                strength=None,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                strength=left_strength,
            ),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=102.0,
            minimum_seconds=102.5,
            recovery_seconds=103.0,
        ),
    )

    right_geometry = SimpleNamespace(
        start_seconds=106.0,
        minimum_seconds=107.0,
        recovery_seconds=108.0,
    )

    right = SimpleNamespace(
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
                strength=None,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                strength=right_strength,
            ),
        ),
        basin_geometry=right_geometry,
    )

    region = SimpleNamespace(
        left_edge=left,
        right_edge=right,
        reference_inside_left_basin=False,
        reference_inside_right_basin=reference_inside_right,
        structural_only_interior_count=0,
    )

    fade_start = 99.0 if fade_starts_before_outgoing else 101.0

    fade = SimpleNamespace(
        rise=None,
        outgoing_fades=(
            SimpleNamespace(
                start_time_seconds=fade_start,
                floor_time_seconds=106.8,
            ),
        ),
    )

    context_peak = SimpleNamespace(
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
                strength=peak_strength,
            ),
        ),
        basin_geometry=None,
    )

    hypotheses = (
        SimpleNamespace(
            proposals=left.proposals,
            basin_geometry=left.basin_geometry,
        ),
        SimpleNamespace(
            proposals=right.proposals,
            basin_geometry=right.basin_geometry,
        ),
        context_peak,
    )

    return replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=107.2,
            hypotheses=hypotheses,
            fade=fade,
            bounded_insert_regions=(region,),
        ),
    )


def test_phase4h_interstitial_extends_outgoing_to_confirmed_terminal_fade() -> None:
    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=107.5,
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

        def analyze_hypotheses(self, **kwargs):
            return ()

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=_phase4h_larry_observation(),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=106.8,
        incoming_start_seconds=107.5,
    )


def test_phase4h_terminal_fade_requires_reference_inside_right_basin() -> None:
    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=107.5,
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

        def analyze_hypotheses(self, **kwargs):
            return ()

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=_phase4h_larry_observation(
            reference_inside_right=False,
        ),
    )

    assert result.temporal is refined


def test_phase4h_terminal_fade_requires_left_structural_peak() -> None:
    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=107.5,
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

        def analyze_hypotheses(self, **kwargs):
            return ()

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=_phase4h_larry_observation(
            left_peak=False,
        ),
    )

    assert result.temporal is refined


def test_phase4h_terminal_fade_rejects_right_structural_peak() -> None:
    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=107.5,
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

        def analyze_hypotheses(self, **kwargs):
            return ()

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=_phase4h_larry_observation(
            right_peak=True,
        ),
    )

    assert result.temporal is refined


def test_phase4h_terminal_fade_requires_current_outgoing_inside_fade() -> None:
    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=107.5,
    )

    class IncomingPipeline:
        def refine(self, **kwargs):
            return refined

        def analyze_hypotheses(self, **kwargs):
            return ()

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(
        observation=_phase4h_larry_observation(
            fade_starts_before_outgoing=False,
        ),
    )

    assert result.temporal is refined


# ---------------------------------------------------------------------------
# Phase 4i: symmetric bounded-insert predecessor terminal fade
# ---------------------------------------------------------------------------


def _phase4i_mark_observation(
    *,
    incoming_phase=True,
    floor_inside_predecessor=True,
    left_peak=False,
):
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_router import BoundedInsertOwnership

    predecessor = _phase2_supported_edge(
        time_seconds=98.0,
        basin_start_seconds=97.0,
        basin_recovery_seconds=99.5,
        structural_strength=0.9,
    )

    left = _phase2_supported_edge(
        time_seconds=102.4,
        basin_start_seconds=101.5,
        basin_recovery_seconds=102.2,
        structural_strength=(
            0.9 if left_peak else 0.4
        ),
    )

    interior = _phase2_structural_only(
        time_seconds=106.0,
    )

    right = _phase2_supported_edge(
        time_seconds=112.0,
        basin_start_seconds=111.5,
        basin_recovery_seconds=112.4,
        structural_strength=0.5,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=103.0,
        left=left,
        right=right,
        interior_hypotheses=(interior,),
    )

    floor = 98.5 if floor_inside_predecessor else 100.0

    incoming_fades = ()

    if incoming_phase:
        incoming_fades = (
            SimpleNamespace(
                start_time_seconds=floor,
                established_time_seconds=105.0,
            ),
        )

    fade = SimpleNamespace(
        rise=None,
        outgoing_fades=(
            SimpleNamespace(
                start_time_seconds=90.0,
                floor_time_seconds=floor,
            ),
        ),
        incoming_fades=incoming_fades,
    )

    base = _bounded_insert_specialist_observation(
        regions=(region,),
    )

    return replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=103.0,
            hypotheses=(
                predecessor,
                left,
                interior,
                right,
            ),
            fade=fade,
            bounded_insert_regions=(region,),
        ),
        bounded_insert_ownership=(
            BoundedInsertOwnership.SYMMETRIC
        ),
    )


def test_phase4i_symmetric_insert_uses_predecessor_terminal_fade() -> None:
    from brynse.transition_specialists import BoundedInsertSpecialist

    result = BoundedInsertSpecialist().resolve(
        observation=_phase4i_mark_observation(),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=98.5,
        incoming_start_seconds=112.0,
    )


def test_phase4i_terminal_fade_requires_matching_incoming_phase() -> None:
    from brynse.transition_specialists import BoundedInsertSpecialist

    result = BoundedInsertSpecialist().resolve(
        observation=_phase4i_mark_observation(
            incoming_phase=False,
        ),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=102.2,
        incoming_start_seconds=112.0,
    )


def test_phase4i_terminal_fade_must_belong_to_predecessor_basin() -> None:
    from brynse.transition_specialists import BoundedInsertSpecialist

    result = BoundedInsertSpecialist().resolve(
        observation=_phase4i_mark_observation(
            floor_inside_predecessor=False,
        ),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=102.2,
        incoming_start_seconds=112.0,
    )


def test_phase4i_terminal_fade_rejects_dominant_left_edge() -> None:
    from brynse.transition_specialists import BoundedInsertSpecialist

    result = BoundedInsertSpecialist().resolve(
        observation=_phase4i_mark_observation(
            left_peak=True,
        ),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=102.2,
        incoming_start_seconds=112.0,
    )


# ---------------------------------------------------------------------------
# Phase 4j: hard-cut terminal acoustic handoff
# ---------------------------------------------------------------------------


def _phase4j_nikki_observation(
    *,
    incoming_phase=True,
    floor_after_minimum=True,
    supported=True,
):
    from dataclasses import replace
    from types import SimpleNamespace

    from brynse.boundaries import (
        BoundaryProposal,
        BoundaryProposalSource,
    )

    base = _observation(boundary=100.75)

    proposals = [
        BoundaryProposal(
            time_seconds=100.5,
            source=BoundaryProposalSource.BASIN,
        ),
    ]

    if supported:
        proposals.append(
            BoundaryProposal(
                time_seconds=100.4,
                source=BoundaryProposalSource.STRUCTURAL,
                strength=0.8,
            )
        )

    hypothesis = SimpleNamespace(
        time_seconds=100.5,
        proposals=tuple(proposals),
        basin_geometry=SimpleNamespace(
            start_seconds=99.5,
            minimum_seconds=100.5,
            recovery_seconds=101.5,
        ),
    )

    floor = 101.25 if floor_after_minimum else 100.25

    incoming_fades = ()

    if incoming_phase:
        incoming_fades = (
            SimpleNamespace(
                start_time_seconds=floor,
                established_time_seconds=102.0,
            ),
        )

    fade = SimpleNamespace(
        rise=None,
        outgoing_fades=(
            SimpleNamespace(
                start_time_seconds=95.0,
                floor_time_seconds=floor,
            ),
        ),
        incoming_fades=incoming_fades,
    )

    return replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=103.0,
            hypotheses=(hypothesis,),
            fade=fade,
        ),
    )


def test_phase4j_hard_cut_uses_terminal_acoustic_handoff() -> None:
    result = HardCutSpecialist().resolve(
        observation=_phase4j_nikki_observation(),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=101.25,
        incoming_start_seconds=101.25,
    )


def test_phase4j_terminal_handoff_requires_matching_incoming_phase() -> None:
    result = HardCutSpecialist().resolve(
        observation=_phase4j_nikki_observation(
            incoming_phase=False,
        ),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=101.5,
        incoming_start_seconds=101.5,
    )


def test_phase4j_terminal_handoff_must_follow_basin_minimum() -> None:
    result = HardCutSpecialist().resolve(
        observation=_phase4j_nikki_observation(
            floor_after_minimum=False,
        ),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=101.5,
        incoming_start_seconds=101.5,
    )


def test_phase4j_terminal_handoff_requires_supported_basin() -> None:
    result = HardCutSpecialist().resolve(
        observation=_phase4j_nikki_observation(
            supported=False,
        ),
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=101.5,
        incoming_start_seconds=101.5,
    )


def test_bounded_insert_partial_roles_exclude_complete_edge_envelopes() -> None:
    from dataclasses import replace

    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_router import (
        BoundedInsertOwnership,
        TransitionSemanticRoles,
    )
    from brynse.transition_specialists import BoundedInsertSpecialist

    left = _phase2_supported_edge(
        time_seconds=100.25,
        basin_start_seconds=99.0,
        basin_recovery_seconds=100.0,
    )
    interior_a = _phase2_structural_only(
        time_seconds=104.0,
    )
    interior_b = _phase2_structural_only(
        time_seconds=106.0,
    )
    right = _phase2_supported_edge(
        time_seconds=110.25,
        basin_start_seconds=110.0,
        basin_recovery_seconds=112.0,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=109.5,
        left=left,
        right=right,
        interior_hypotheses=(
            interior_a,
            interior_b,
        ),
    )

    base = _bounded_insert_specialist_observation(
        regions=(region,),
    )

    observation = replace(
        base,
        routing=replace(
            base.routing,
            reference_time_seconds=109.5,
            hypotheses=(
                left,
                interior_a,
                interior_b,
                right,
            ),
            bounded_insert_regions=(region,),
        ),
        bounded_insert_ownership=BoundedInsertOwnership.SYMMETRIC,
        semantic_roles=TransitionSemanticRoles(
            inserted_region=region,
            incoming_handoff=right,
        ),
    )

    result = BoundedInsertSpecialist().resolve(
        observation=observation,
    )

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=99.0,
        incoming_start_seconds=112.0,
    )
