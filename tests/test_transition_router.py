import pytest

from brynse.boundaries import (
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
)
from brynse.bounded_insert import BoundedInsertRegionEvidence
from brynse.transition_evidence import (
    CrossfadeBasinEvidence,
    CrossfadeRiseEvidence,
    FadeRoutingEvidence,
    OutgoingFadeEvidence,
    TransitionEvidenceState,
)
from brynse.transition_router import (
    BoundedInsertOwnership,
    DefaultTransitionRouter,
    HardCutRoutingAssessment,
    TransitionKind,
    TransitionRoute,
    TransitionRouter,
    TransitionRoutingContext,
    TransitionSemanticRoles,
)


class StubTransitionRouter:
    def __init__(self, kind: TransitionKind) -> None:
        self._kind = kind

    def classify(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> TransitionKind:
        _ = context
        return self._kind


def test_transition_kind_exposes_only_specialist_routing_classes() -> None:
    assert tuple(TransitionKind) == (
        TransitionKind.HARD_CUT,
        TransitionKind.FADE,
        TransitionKind.INTERSTITIAL,
        TransitionKind.BOUNDED_INSERT,
        TransitionKind.AMBIGUOUS,
    )


def test_transition_routing_context_preserves_descriptive_inputs() -> None:
    evidence = TransitionEvidenceState(
        spectral_change=0.42,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=120.5,
        evidence=evidence,
    )

    assert context.reference_time_seconds == 120.5
    assert context.evidence is evidence
    assert context.hypotheses == ()
    assert context.fade == FadeRoutingEvidence()


def test_transition_routing_context_rejects_negative_reference_time() -> None:
    with pytest.raises(
        ValueError,
        match="reference_time_seconds",
    ):
        TransitionRoutingContext(
            reference_time_seconds=-0.01,
            evidence=TransitionEvidenceState(),
        )


def test_transition_router_contract_classifies_context_without_boundaries() -> None:
    router: TransitionRouter = StubTransitionRouter(
        TransitionKind.AMBIGUOUS,
    )

    result = router.classify(
        context=TransitionRoutingContext(
            reference_time_seconds=42.0,
            evidence=TransitionEvidenceState(
                spectral_change=0.42,
            ),
        )
    )

    assert result is TransitionKind.AMBIGUOUS


def test_transition_kind_carries_no_temporal_boundary() -> None:
    result = TransitionKind.HARD_CUT

    assert not hasattr(result, "incoming_start_seconds")
    assert not hasattr(result, "outgoing_end_seconds")


def test_transition_routing_context_carries_no_resolved_boundary() -> None:
    context = TransitionRoutingContext(
        reference_time_seconds=42.0,
        evidence=TransitionEvidenceState(),
    )

    assert not hasattr(context, "incoming_start_seconds")
    assert not hasattr(context, "outgoing_end_seconds")



def test_transition_route_carries_ownership_without_temporal_boundary() -> None:
    route = TransitionRoute(
        kind=TransitionKind.BOUNDED_INSERT,
        bounded_insert_ownership=BoundedInsertOwnership.ASYMMETRIC,
    )

    assert route.kind is TransitionKind.BOUNDED_INSERT
    assert (
        route.bounded_insert_ownership
        is BoundedInsertOwnership.ASYMMETRIC
    )
    assert not hasattr(route, "incoming_start_seconds")
    assert not hasattr(route, "outgoing_end_seconds")


def test_transition_semantic_roles_reference_evidence_without_temporal_cuts() -> None:
    from brynse.boundaries import BoundaryProposal, BoundaryProposalSource
    from brynse.bounded_insert import BoundedInsertRegionEvidence

    outgoing = BoundaryHypothesis(
        time_seconds=98.0,
        proposals=(
            BoundaryProposal(
                time_seconds=98.0,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
    )
    incoming = BoundaryHypothesis(
        time_seconds=102.0,
        proposals=(
            BoundaryProposal(
                time_seconds=102.0,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
    )
    region = BoundedInsertRegionEvidence(
        reference_time_seconds=100.0,
        left=outgoing,
        right=incoming,
    )

    roles = TransitionSemanticRoles(
        outgoing_handoff=outgoing,
        incoming_handoff=incoming,
        inserted_region=region,
    )

    assert roles.outgoing_handoff is outgoing
    assert roles.incoming_handoff is incoming
    assert roles.inserted_region is region

    for temporal_name in (
        "incoming_start_seconds",
        "outgoing_end_seconds",
        "boundary_time_seconds",
        "cut_time_seconds",
    ):
        assert not hasattr(roles, temporal_name)


def test_transition_route_semantic_roles_default_to_none() -> None:
    route = TransitionRoute(
        kind=TransitionKind.AMBIGUOUS,
    )

    assert route.semantic_roles is None


def test_default_transition_router_classify_matches_route_kind() -> None:
    class Router(DefaultTransitionRouter):
        def bounded_insert_ownership(self, *, context):
            _ = context
            return BoundedInsertOwnership.ASYMMETRIC

    router = Router()

    context = TransitionRoutingContext(
        reference_time_seconds=42.0,
        evidence=TransitionEvidenceState(),
    )

    route = router.route(context=context)

    assert route.kind is TransitionKind.BOUNDED_INSERT
    assert (
        route.bounded_insert_ownership
        is BoundedInsertOwnership.ASYMMETRIC
    )
    assert route.fade_assessment is not None
    assert route.fade_assessment.legacy_candidate is False
    assert router.classify(context=context) is route.kind


def test_transition_routing_context_preserves_fade_evidence() -> None:
    rise = CrossfadeRiseEvidence(
        time_seconds=431.15,
        jump_ratio=3.5,
        persistence_ratio=4.0,
        after_global_ratio=3.0,
    )
    basin = CrossfadeBasinEvidence(
        center_time_seconds=426.75,
        lead_seconds=0.85,
        width_seconds=0.7,
        prominence=1.5,
        quietness=1.8,
    )
    fade = FadeRoutingEvidence(
        rise=rise,
        basins=(basin,),
    )

    context = TransitionRoutingContext(
        reference_time_seconds=427.6,
        evidence=TransitionEvidenceState(),
        fade=fade,
    )

    assert context.fade is fade
    assert context.fade.rise is rise
    assert context.fade.basins == (basin,)
    assert not hasattr(context.fade, "incoming_start_seconds")
    assert not hasattr(context.fade, "outgoing_end_seconds")
    assert not hasattr(context.fade, "kind")


def _interstitial_hypothesis(
    *,
    basin_time: float,
    structural_time: float,
    basin_start: float,
    basin_recovery: float,
):
    from brynse.boundaries import (
        BoundaryEvidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )

    basin = BoundaryProposal(
        time_seconds=basin_time,
        source=BoundaryProposalSource.BASIN,
    )
    structural = BoundaryProposal(
        time_seconds=structural_time,
        source=BoundaryProposalSource.STRUCTURAL,
    )

    return BoundaryHypothesis(
        time_seconds=(basin_time + structural_time) / 2,
        proposals=(basin, structural),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_start_seconds=basin_start,
                basin_recovery_seconds=basin_recovery,
            ),
        ),
    )


def test_fade_assessment_correlates_boundary_support_without_cutting() -> None:
    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.transition_evidence import (
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )

    hypothesis = BoundaryHypothesis(
        time_seconds=104.0,
        proposals=(
            BoundaryProposal(
                time_seconds=104.0,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
        evidence=(),
    )

    outgoing = OutgoingFadeEvidence(
        start_time_seconds=100.0,
        floor_time_seconds=104.0,
        rise_time_seconds=None,
        start_rms=1000.0,
        floor_rms=200.0,
        descent_ratio=5.0,
        descent_consistency=0.9,
    )

    incoming = IncomingFadeEvidence(
        start_time_seconds=104.0,
        established_time_seconds=105.0,
        start_rms=200.0,
        established_rms=1000.0,
        ascent_ratio=5.0,
        ascent_consistency=0.9,
        persistence_ratio=4.0,
    )

    router = DefaultTransitionRouter()

    assessment = router.assess_fade(
        context=TransitionRoutingContext(
            reference_time_seconds=103.0,
            evidence=TransitionEvidenceState(),
            hypotheses=(hypothesis,),
            fade=FadeRoutingEvidence(
                outgoing_fades=(outgoing,),
                incoming_fades=(incoming,),
            ),
        )
    )

    assert assessment.rise_present is False
    assert assessment.basin_count == 0
    assert assessment.geometry_count == 0
    assert assessment.outgoing_association is not None
    assert assessment.outgoing_association.fade is outgoing
    assert assessment.outgoing_association.hypothesis is hypothesis
    assert assessment.outgoing_association.support_distance_seconds == 0.0

    assert assessment.incoming_association is not None
    assert assessment.incoming_association.fade is incoming
    assert assessment.incoming_association.hypothesis is hypothesis
    assert assessment.incoming_association.support_distance_seconds == 0.0

    assert assessment.has_two_sided_support is True
    assert assessment.legacy_candidate is False

    assert not hasattr(assessment, "outgoing_end_seconds")
    assert not hasattr(assessment, "incoming_start_seconds")


def test_fade_assessment_prefers_outgoing_reconciled_with_selected_incoming() -> None:
    from brynse.boundaries import (
        BoundaryConfidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.transition_evidence import (
        FadeGeometryEvidence,
        FadeRoutingEvidence,
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
        TransitionEvidenceState,
    )
    from brynse.transition_router import (
        DefaultTransitionRouter,
        TransitionRoutingContext,
    )

    def hypothesis(time: float) -> BoundaryHypothesis:
        return BoundaryHypothesis(
            time_seconds=time,
            proposals=(
                BoundaryProposal(
                    time_seconds=time,
                    source=BoundaryProposalSource.STRUCTURAL,
                ),
            ),
            evidence=(),
            confidence=BoundaryConfidence.UNRESOLVED,
        )

    stale_outgoing = OutgoingFadeEvidence(
        start_time_seconds=90.0,
        floor_time_seconds=95.0,
        rise_time_seconds=None,
        start_rms=1000.0,
        floor_rms=100.0,
        descent_ratio=10.0,
        descent_consistency=0.9,
    )
    coherent_outgoing = OutgoingFadeEvidence(
        start_time_seconds=98.0,
        floor_time_seconds=101.0,
        rise_time_seconds=104.0,
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

    stale_support = hypothesis(95.0)
    coherent_support = hypothesis(101.2)

    router = DefaultTransitionRouter()

    context = TransitionRoutingContext(
        reference_time_seconds=100.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(
            stale_support,
            coherent_support,
        ),
        fade=FadeRoutingEvidence(
            outgoing_fades=(
                stale_outgoing,
                coherent_outgoing,
            ),
            incoming_fades=(incoming,),
            geometries=(
                FadeGeometryEvidence(
                    outgoing=coherent_outgoing,
                    incoming=incoming,
                ),
            ),
        ),
    )

    legacy_outgoing = router.associate_outgoing_fade(
        reference_time_seconds=context.reference_time_seconds,
        outgoing_fades=context.fade.outgoing_fades,
        hypotheses=context.hypotheses,
    )

    assert legacy_outgoing is not None
    assert legacy_outgoing.fade is stale_outgoing

    assessment = router.assess_fade(context=context)

    assert assessment.incoming_association is not None
    assert assessment.incoming_association.fade is incoming

    assert assessment.outgoing_association is not None
    assert assessment.outgoing_association.fade is coherent_outgoing
    assert assessment.outgoing_association.hypothesis is coherent_support


def test_fade_assessment_keeps_legacy_outgoing_without_selected_geometry() -> None:
    from brynse.boundaries import (
        BoundaryConfidence,
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.transition_evidence import (
        FadeGeometryEvidence,
        FadeRoutingEvidence,
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
        TransitionEvidenceState,
    )
    from brynse.transition_router import (
        DefaultTransitionRouter,
        TransitionRoutingContext,
    )

    def hypothesis(time: float) -> BoundaryHypothesis:
        return BoundaryHypothesis(
            time_seconds=time,
            proposals=(
                BoundaryProposal(
                    time_seconds=time,
                    source=BoundaryProposalSource.STRUCTURAL,
                ),
            ),
            evidence=(),
            confidence=BoundaryConfidence.UNRESOLVED,
        )

    legacy_outgoing = OutgoingFadeEvidence(
        start_time_seconds=98.0,
        floor_time_seconds=101.0,
        rise_time_seconds=None,
        start_rms=1000.0,
        floor_rms=100.0,
        descent_ratio=10.0,
        descent_consistency=0.9,
    )
    unrelated_outgoing = OutgoingFadeEvidence(
        start_time_seconds=97.0,
        floor_time_seconds=99.0,
        rise_time_seconds=None,
        start_rms=900.0,
        floor_rms=100.0,
        descent_ratio=9.0,
        descent_consistency=0.8,
    )

    selected_incoming = IncomingFadeEvidence(
        start_time_seconds=102.0,
        established_time_seconds=104.0,
        start_rms=100.0,
        established_rms=900.0,
        ascent_ratio=9.0,
        ascent_consistency=0.9,
        persistence_ratio=5.0,
    )
    unrelated_incoming = IncomingFadeEvidence(
        start_time_seconds=98.5,
        established_time_seconds=99.5,
        start_rms=100.0,
        established_rms=500.0,
        ascent_ratio=5.0,
        ascent_consistency=0.8,
        persistence_ratio=3.0,
    )

    support = hypothesis(101.0)
    incoming_support = hypothesis(102.1)

    router = DefaultTransitionRouter()

    context = TransitionRoutingContext(
        reference_time_seconds=100.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(
            support,
            incoming_support,
        ),
        fade=FadeRoutingEvidence(
            outgoing_fades=(
                legacy_outgoing,
                unrelated_outgoing,
            ),
            incoming_fades=(
                selected_incoming,
                unrelated_incoming,
            ),
            geometries=(
                FadeGeometryEvidence(
                    outgoing=unrelated_outgoing,
                    incoming=unrelated_incoming,
                ),
            ),
        ),
    )

    assessment = router.assess_fade(context=context)

    assert assessment.incoming_association is not None
    assert assessment.incoming_association.fade is selected_incoming

    assert assessment.outgoing_association is not None
    assert assessment.outgoing_association.fade is legacy_outgoing
    assert assessment.outgoing_association.hypothesis is support


def test_fade_route_keeps_legacy_candidate_semantics_with_rich_assessment() -> None:
    fade = FadeRoutingEvidence(
        rise=CrossfadeRiseEvidence(
            time_seconds=431.15,
            jump_ratio=3.5,
            persistence_ratio=4.0,
            after_global_ratio=3.0,
        ),
        basins=(
            CrossfadeBasinEvidence(
                center_time_seconds=426.75,
                lead_seconds=0.85,
                width_seconds=0.7,
                prominence=1.5,
                quietness=1.8,
            ),
        ),
    )

    route = DefaultTransitionRouter().route(
        context=TransitionRoutingContext(
            reference_time_seconds=427.6,
            evidence=TransitionEvidenceState(),
            fade=fade,
        )
    )

    assert route.kind is TransitionKind.FADE
    assert route.fade_assessment is not None
    assert route.fade_assessment.legacy_candidate is True


def test_fade_route_rejects_completed_geometry_from_earlier_transition() -> None:
    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.transition_evidence import (
        FadeGeometryEvidence,
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )

    reference = 120.0

    outgoing_support = BoundaryHypothesis(
        time_seconds=101.0,
        proposals=(
            BoundaryProposal(
                time_seconds=101.0,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
        evidence=(),
    )

    later_support = BoundaryHypothesis(
        time_seconds=130.0,
        proposals=(
            BoundaryProposal(
                time_seconds=130.0,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
        evidence=(),
    )

    outgoing = OutgoingFadeEvidence(
        start_time_seconds=96.0,
        floor_time_seconds=101.0,
        rise_time_seconds=104.0,
        start_rms=1000.0,
        floor_rms=100.0,
        descent_ratio=10.0,
        descent_consistency=0.9,
    )

    completed_incoming = IncomingFadeEvidence(
        start_time_seconds=101.0,
        established_time_seconds=106.0,
        start_rms=100.0,
        established_rms=900.0,
        ascent_ratio=9.0,
        ascent_consistency=0.9,
        persistence_ratio=4.0,
    )

    fade = FadeRoutingEvidence(
        rise=CrossfadeRiseEvidence(
            time_seconds=130.0,
            jump_ratio=3.5,
            persistence_ratio=4.0,
            after_global_ratio=3.0,
        ),
        basins=(
            CrossfadeBasinEvidence(
                center_time_seconds=101.0,
                lead_seconds=0.85,
                width_seconds=0.7,
                prominence=1.5,
                quietness=1.8,
            ),
        ),
        outgoing_fades=(outgoing,),
        incoming_fades=(completed_incoming,),
        geometries=(
            FadeGeometryEvidence(
                outgoing=outgoing,
                incoming=completed_incoming,
            ),
        ),
    )

    router = DefaultTransitionRouter()
    context = TransitionRoutingContext(
        reference_time_seconds=reference,
        evidence=TransitionEvidenceState(),
        hypotheses=(outgoing_support, later_support),
        fade=fade,
    )

    assessment = router.assess_fade(context=context)

    assert assessment.legacy_candidate is True
    assert assessment.outgoing_association is not None
    assert assessment.outgoing_association.fade is outgoing
    assert assessment.incoming_association is None
    assert assessment.stale_completed_geometry is True
    assert assessment.ownership_candidate is False

    route = router.route(context=context)

    assert route.kind is TransitionKind.AMBIGUOUS
    assert route.fade_assessment is not None
    assert route.fade_assessment.legacy_candidate is True
    assert route.fade_assessment.ownership_candidate is False


def test_fade_assessment_distinguishes_two_sided_support_from_reconciled_geometry() -> None:
    from dataclasses import replace

    from brynse.transition_router import FadeRoutingAssessment

    assessment = FadeRoutingAssessment(
        rise_present=True,
        basin_count=1,
        geometry_count=2,
        outgoing_association=object(),
        incoming_association=object(),
        reconciled_geometry_count=0,
    )

    assert assessment.has_two_sided_support is True
    assert assessment.has_reconciled_geometry is False

    reconciled = replace(
        assessment,
        reconciled_geometry_count=1,
    )

    assert reconciled.has_two_sided_support is True
    assert reconciled.has_reconciled_geometry is True


def test_default_transition_router_routes_clear_fade() -> None:
    from brynse.transition_router import DefaultTransitionRouter

    fade = FadeRoutingEvidence(
        rise=CrossfadeRiseEvidence(
            time_seconds=431.15,
            jump_ratio=3.5,
            persistence_ratio=4.0,
            after_global_ratio=3.0,
        ),
        basins=(
            CrossfadeBasinEvidence(
                center_time_seconds=426.75,
                lead_seconds=0.85,
                width_seconds=0.7,
                prominence=1.5,
                quietness=1.8,
            ),
        ),
    )

    result = DefaultTransitionRouter().classify(
        context=TransitionRoutingContext(
            reference_time_seconds=427.6,
            evidence=TransitionEvidenceState(),
            fade=fade,
        )
    )

    assert result is TransitionKind.FADE


def test_interstitial_assessment_orders_supported_handoffs_without_cutting() -> None:
    router = DefaultTransitionRouter()

    left = _interstitial_hypothesis(
        basin_time=100.5,
        structural_time=100.7,
        basin_start=100.2,
        basin_recovery=101.0,
    )

    right = _interstitial_hypothesis(
        basin_time=102.0,
        structural_time=102.2,
        basin_start=101.7,
        basin_recovery=102.6,
    )

    assessment = router.assess_interstitial(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=TransitionEvidenceState(),
            hypotheses=(right, left),
        )
    )

    assert assessment.routing_handoffs == (left, right)
    assert assessment.supported_handoffs == (left, right)
    assert assessment.legacy_candidate is True
    assert assessment.absorbed_right_edge_fade is False
    assert assessment.terminal_fade_region is None
    assert assessment.terminal_fade_candidates == ()

    assert not hasattr(assessment, "outgoing_end_seconds")
    assert not hasattr(assessment, "incoming_start_seconds")


def test_interstitial_assessment_keeps_single_handoff_non_candidate() -> None:
    router = DefaultTransitionRouter()

    hypothesis = _interstitial_hypothesis(
        basin_time=101.0,
        structural_time=101.2,
        basin_start=100.7,
        basin_recovery=101.6,
    )

    assessment = router.assess_interstitial(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=TransitionEvidenceState(),
            hypotheses=(hypothesis,),
        )
    )

    assert assessment.routing_handoffs == (hypothesis,)
    assert assessment.supported_handoffs == (hypothesis,)
    assert assessment.legacy_candidate is False


def test_interstitial_roles_use_owned_routing_handoffs() -> None:
    from brynse.transition_evidence import (
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        IncomingFadeRoutingAssociation,
        InterstitialRoutingAssessment,
        OutgoingFadeRoutingAssociation,
    )

    outgoing = _interstitial_hypothesis(
        basin_time=98.0,
        structural_time=98.2,
        basin_start=97.5,
        basin_recovery=98.5,
    )
    incoming = _interstitial_hypothesis(
        basin_time=102.0,
        structural_time=102.2,
        basin_start=101.5,
        basin_recovery=102.5,
    )

    fade = FadeRoutingAssessment(
        rise_present=False,
        basin_count=0,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=OutgoingFadeEvidence(
                start_time_seconds=96.0,
                floor_time_seconds=98.0,
                rise_time_seconds=None,
                start_rms=1000.0,
                floor_rms=100.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
            hypothesis=outgoing,
            support_distance_seconds=0.0,
        ),
        incoming_association=IncomingFadeRoutingAssociation(
            fade=IncomingFadeEvidence(
                start_time_seconds=101.0,
                established_time_seconds=102.0,
                start_rms=100.0,
                established_rms=800.0,
                ascent_ratio=8.0,
                ascent_consistency=0.9,
                persistence_ratio=3.0,
            ),
            hypothesis=incoming,
            support_distance_seconds=0.0,
        ),
    )
    interstitial = InterstitialRoutingAssessment(
        routing_handoffs=(outgoing, incoming),
        supported_handoffs=(outgoing, incoming),
    )

    roles = DefaultTransitionRouter.assign_interstitial_roles(
        fade=fade,
        interstitial=interstitial,
    )

    assert roles is not None
    assert roles.outgoing_handoff is outgoing
    assert roles.incoming_handoff is incoming


def test_interstitial_roles_allow_partial_supported_association() -> None:
    from brynse.transition_evidence import OutgoingFadeEvidence
    from brynse.transition_router import (
        FadeRoutingAssessment,
        InterstitialRoutingAssessment,
        OutgoingFadeRoutingAssociation,
    )

    outgoing = _interstitial_hypothesis(
        basin_time=98.0,
        structural_time=98.2,
        basin_start=97.5,
        basin_recovery=98.5,
    )

    fade = FadeRoutingAssessment(
        rise_present=False,
        basin_count=0,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=OutgoingFadeEvidence(
                start_time_seconds=96.0,
                floor_time_seconds=98.0,
                rise_time_seconds=None,
                start_rms=1000.0,
                floor_rms=100.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
            hypothesis=outgoing,
            support_distance_seconds=0.0,
        ),
    )
    interstitial = InterstitialRoutingAssessment(
        supported_handoffs=(outgoing,),
    )

    roles = DefaultTransitionRouter.assign_interstitial_roles(
        fade=fade,
        interstitial=interstitial,
    )

    assert roles is None


def test_interstitial_roles_complete_incoming_from_unique_observed_region() -> None:
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_evidence import (
        OutgoingFadeEvidence,
        TransitionEvidenceState,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        InterstitialRoutingAssessment,
        OutgoingFadeRoutingAssociation,
        TransitionRoutingContext,
    )

    outgoing = _interstitial_hypothesis(
        basin_time=98.0,
        structural_time=98.2,
        basin_start=97.5,
        basin_recovery=98.5,
    )
    incoming = _interstitial_hypothesis(
        basin_time=102.0,
        structural_time=102.2,
        basin_start=101.5,
        basin_recovery=102.5,
    )

    fade = FadeRoutingAssessment(
        rise_present=False,
        basin_count=0,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=OutgoingFadeEvidence(
                start_time_seconds=96.0,
                floor_time_seconds=98.0,
                rise_time_seconds=None,
                start_rms=1000.0,
                floor_rms=100.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
            hypothesis=outgoing,
            support_distance_seconds=0.0,
        ),
    )
    interstitial = InterstitialRoutingAssessment(
        supported_handoffs=(outgoing, incoming),
    )
    region = BoundedInsertRegionEvidence(
        reference_time_seconds=100.0,
        left=outgoing,
        right=incoming,
    )
    context = TransitionRoutingContext(
        reference_time_seconds=100.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(outgoing, incoming),
        bounded_insert_regions=(region,),
    )

    roles = DefaultTransitionRouter.assign_interstitial_roles(
        fade=fade,
        interstitial=interstitial,
        context=context,
    )

    assert roles is not None
    assert roles.outgoing_handoff is outgoing
    assert roles.incoming_handoff is incoming


def test_interstitial_roles_keep_incoming_unknown_with_multiple_observed_regions() -> None:
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_evidence import (
        OutgoingFadeEvidence,
        TransitionEvidenceState,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        InterstitialRoutingAssessment,
        OutgoingFadeRoutingAssociation,
        TransitionRoutingContext,
    )

    outgoing = _interstitial_hypothesis(
        basin_time=80.0,
        structural_time=80.2,
        basin_start=79.5,
        basin_recovery=80.5,
    )
    middle = _interstitial_hypothesis(
        basin_time=100.0,
        structural_time=100.2,
        basin_start=98.0,
        basin_recovery=101.0,
    )
    later = _interstitial_hypothesis(
        basin_time=115.0,
        structural_time=115.2,
        basin_start=114.5,
        basin_recovery=115.5,
    )

    fade = FadeRoutingAssessment(
        rise_present=False,
        basin_count=0,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=OutgoingFadeEvidence(
                start_time_seconds=76.0,
                floor_time_seconds=80.0,
                rise_time_seconds=None,
                start_rms=1000.0,
                floor_rms=100.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
            hypothesis=outgoing,
            support_distance_seconds=0.0,
        ),
    )

    interstitial = InterstitialRoutingAssessment(
        supported_handoffs=(
            outgoing,
            middle,
            later,
        ),
    )

    context = TransitionRoutingContext(
        reference_time_seconds=100.1,
        evidence=TransitionEvidenceState(),
        hypotheses=(
            outgoing,
            middle,
            later,
        ),
        bounded_insert_regions=(
            BoundedInsertRegionEvidence(
                reference_time_seconds=100.1,
                left=outgoing,
                right=middle,
            ),
            BoundedInsertRegionEvidence(
                reference_time_seconds=100.1,
                left=middle,
                right=later,
            ),
        ),
    )

    roles = DefaultTransitionRouter.assign_interstitial_roles(
        fade=fade,
        interstitial=interstitial,
        context=context,
    )

    assert roles is not None
    assert roles.outgoing_handoff is outgoing
    assert roles.incoming_handoff is None


def test_interstitial_roles_keep_incoming_unknown_when_region_is_ambiguous() -> None:
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_evidence import (
        OutgoingFadeEvidence,
        TransitionEvidenceState,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        InterstitialRoutingAssessment,
        OutgoingFadeRoutingAssociation,
        TransitionRoutingContext,
    )

    outgoing = _interstitial_hypothesis(
        basin_time=98.0,
        structural_time=98.2,
        basin_start=97.5,
        basin_recovery=98.5,
    )
    incoming_a = _interstitial_hypothesis(
        basin_time=102.0,
        structural_time=102.2,
        basin_start=101.5,
        basin_recovery=102.5,
    )
    incoming_b = _interstitial_hypothesis(
        basin_time=106.0,
        structural_time=106.2,
        basin_start=105.5,
        basin_recovery=106.5,
    )

    fade = FadeRoutingAssessment(
        rise_present=False,
        basin_count=0,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=OutgoingFadeEvidence(
                start_time_seconds=96.0,
                floor_time_seconds=98.0,
                rise_time_seconds=None,
                start_rms=1000.0,
                floor_rms=100.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
            hypothesis=outgoing,
            support_distance_seconds=0.0,
        ),
    )
    interstitial = InterstitialRoutingAssessment(
        supported_handoffs=(
            outgoing,
            incoming_a,
            incoming_b,
        ),
    )

    context = TransitionRoutingContext(
        reference_time_seconds=100.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(
            outgoing,
            incoming_a,
            incoming_b,
        ),
        bounded_insert_regions=(
            BoundedInsertRegionEvidence(
                reference_time_seconds=100.0,
                left=outgoing,
                right=incoming_a,
            ),
            BoundedInsertRegionEvidence(
                reference_time_seconds=100.0,
                left=outgoing,
                right=incoming_b,
            ),
        ),
    )

    roles = DefaultTransitionRouter.assign_interstitial_roles(
        fade=fade,
        interstitial=interstitial,
        context=context,
    )

    assert roles is not None
    assert roles.outgoing_handoff is outgoing
    assert roles.incoming_handoff is None


def test_interstitial_roles_do_not_invent_shared_handoff_from_fade() -> None:
    from brynse.transition_evidence import (
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        IncomingFadeRoutingAssociation,
        InterstitialRoutingAssessment,
        OutgoingFadeRoutingAssociation,
    )

    shared = _interstitial_hypothesis(
        basin_time=100.0,
        structural_time=100.2,
        basin_start=99.0,
        basin_recovery=101.0,
    )

    fade = FadeRoutingAssessment(
        rise_present=False,
        basin_count=0,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=OutgoingFadeEvidence(
                start_time_seconds=98.0,
                floor_time_seconds=100.0,
                rise_time_seconds=None,
                start_rms=1000.0,
                floor_rms=100.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
            hypothesis=shared,
            support_distance_seconds=0.0,
        ),
        incoming_association=IncomingFadeRoutingAssociation(
            fade=IncomingFadeEvidence(
                start_time_seconds=99.0,
                established_time_seconds=100.0,
                start_rms=100.0,
                established_rms=800.0,
                ascent_ratio=8.0,
                ascent_consistency=0.9,
                persistence_ratio=3.0,
            ),
            hypothesis=shared,
            support_distance_seconds=0.0,
        ),
    )
    interstitial = InterstitialRoutingAssessment(
        supported_handoffs=(shared,),
    )

    roles = DefaultTransitionRouter.assign_interstitial_roles(
        fade=fade,
        interstitial=interstitial,
    )

    assert roles is None


def test_interstitial_roles_ignore_fade_association_without_interstitial_support() -> None:
    from brynse.boundaries import BoundaryProposal, BoundaryProposalSource
    from brynse.transition_evidence import OutgoingFadeEvidence
    from brynse.transition_router import (
        FadeRoutingAssessment,
        InterstitialRoutingAssessment,
        OutgoingFadeRoutingAssociation,
    )

    unrelated = BoundaryHypothesis(
        time_seconds=98.0,
        proposals=(
            BoundaryProposal(
                time_seconds=98.0,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
    )

    fade = FadeRoutingAssessment(
        rise_present=False,
        basin_count=0,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=OutgoingFadeEvidence(
                start_time_seconds=96.0,
                floor_time_seconds=98.0,
                rise_time_seconds=None,
                start_rms=1000.0,
                floor_rms=100.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
            hypothesis=unrelated,
            support_distance_seconds=0.0,
        ),
    )

    assert (
        DefaultTransitionRouter.assign_interstitial_roles(
            fade=fade,
            interstitial=InterstitialRoutingAssessment(),
        )
        is None
    )


def test_default_transition_router_routes_clear_interstitial_pattern() -> None:
    from brynse.transition_router import DefaultTransitionRouter

    hypotheses = (
        _interstitial_hypothesis(
            basin_time=100.5,
            structural_time=100.7,
            basin_start=100.2,
            basin_recovery=101.0,
        ),
        _interstitial_hypothesis(
            basin_time=102.0,
            structural_time=102.2,
            basin_start=101.7,
            basin_recovery=102.6,
        ),
    )

    router = DefaultTransitionRouter()

    context = TransitionRoutingContext(
        reference_time_seconds=100.0,
        evidence=TransitionEvidenceState(),
        hypotheses=hypotheses,
    )

    route = router.route(
        context=context,
    )

    assert route.kind is TransitionKind.INTERSTITIAL
    assert route.interstitial_assessment is not None
    assert route.interstitial_assessment.legacy_candidate is True
    assert router.classify(context=context) is route.kind


def test_default_transition_router_keeps_single_interstitial_event_ambiguous() -> None:
    from brynse.transition_router import DefaultTransitionRouter

    result = DefaultTransitionRouter().classify(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=TransitionEvidenceState(),
            hypotheses=(
                _interstitial_hypothesis(
                    basin_time=101.0,
                    structural_time=101.2,
                    basin_start=100.7,
                    basin_recovery=101.6,
                ),
            ),
        )
    )

    assert result is TransitionKind.AMBIGUOUS


def test_default_transition_router_prefers_interstitial_when_fade_lacks_two_sided_owner() -> None:
    from brynse.transition_router import DefaultTransitionRouter

    fade = FadeRoutingEvidence(
        rise=CrossfadeRiseEvidence(
            time_seconds=105.0,
            jump_ratio=2.5,
            persistence_ratio=3.0,
            after_global_ratio=2.0,
        ),
        basins=(
            CrossfadeBasinEvidence(
                center_time_seconds=99.0,
                lead_seconds=1.0,
                width_seconds=0.6,
                prominence=1.2,
                quietness=1.4,
            ),
        ),
    )

    hypotheses = (
        _interstitial_hypothesis(
            basin_time=101.0,
            structural_time=101.2,
            basin_start=100.7,
            basin_recovery=101.6,
        ),
        _interstitial_hypothesis(
            basin_time=103.0,
            structural_time=103.2,
            basin_start=102.7,
            basin_recovery=103.6,
        ),
    )

    result = DefaultTransitionRouter().classify(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=TransitionEvidenceState(),
            hypotheses=hypotheses,
            fade=fade,
        )
    )

    assert result is TransitionKind.INTERSTITIAL



def test_fade_interstitial_conflict_prefers_two_sided_fade_owner() -> None:
    from brynse.transition_evidence import (
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        IncomingFadeRoutingAssociation,
        InterstitialRoutingAssessment,
        OutgoingFadeRoutingAssociation,
    )

    outgoing_hypothesis = _interstitial_hypothesis(
        basin_time=101.0,
        structural_time=101.2,
        basin_start=100.7,
        basin_recovery=101.6,
    )
    incoming_hypothesis = _interstitial_hypothesis(
        basin_time=103.0,
        structural_time=103.2,
        basin_start=102.7,
        basin_recovery=103.6,
    )

    fade = FadeRoutingAssessment(
        rise_present=True,
        basin_count=1,
        geometry_count=1,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=OutgoingFadeEvidence(
                start_time_seconds=99.0,
                floor_time_seconds=101.0,
                rise_time_seconds=102.0,
                start_rms=1000.0,
                floor_rms=100.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
            hypothesis=outgoing_hypothesis,
            support_distance_seconds=0.0,
        ),
        incoming_association=IncomingFadeRoutingAssociation(
            fade=IncomingFadeEvidence(
                start_time_seconds=102.0,
                established_time_seconds=103.0,
                start_rms=100.0,
                established_rms=800.0,
                ascent_ratio=8.0,
                ascent_consistency=0.9,
                persistence_ratio=3.0,
            ),
            hypothesis=incoming_hypothesis,
            support_distance_seconds=0.0,
        ),
    )

    interstitial = InterstitialRoutingAssessment(
        routing_handoffs=(
            outgoing_hypothesis,
            incoming_hypothesis,
        ),
        supported_handoffs=(
            outgoing_hypothesis,
            incoming_hypothesis,
        ),
    )

    assert fade.legacy_candidate is True
    assert fade.has_two_sided_support is True
    assert interstitial.legacy_candidate is True

    assert (
        DefaultTransitionRouter.resolve_fade_interstitial_conflict(
            fade=fade,
            interstitial=interstitial,
        )
        is TransitionKind.FADE
    )


def test_fade_interstitial_conflict_keeps_absorbed_fade_interstitial() -> None:
    from brynse.transition_evidence import (
        IncomingFadeEvidence,
        OutgoingFadeEvidence,
    )
    from brynse.transition_router import (
        FadeRoutingAssessment,
        IncomingFadeRoutingAssociation,
        InterstitialRoutingAssessment,
        OutgoingFadeRoutingAssociation,
    )

    outgoing_hypothesis = _interstitial_hypothesis(
        basin_time=101.0,
        structural_time=101.2,
        basin_start=100.7,
        basin_recovery=101.6,
    )
    incoming_hypothesis = _interstitial_hypothesis(
        basin_time=103.0,
        structural_time=103.2,
        basin_start=102.7,
        basin_recovery=103.6,
    )

    fade = FadeRoutingAssessment(
        rise_present=True,
        basin_count=1,
        geometry_count=1,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=OutgoingFadeEvidence(
                start_time_seconds=99.0,
                floor_time_seconds=101.0,
                rise_time_seconds=102.0,
                start_rms=1000.0,
                floor_rms=100.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
            hypothesis=outgoing_hypothesis,
            support_distance_seconds=0.0,
        ),
        incoming_association=IncomingFadeRoutingAssociation(
            fade=IncomingFadeEvidence(
                start_time_seconds=102.0,
                established_time_seconds=103.0,
                start_rms=100.0,
                established_rms=800.0,
                ascent_ratio=8.0,
                ascent_consistency=0.9,
                persistence_ratio=3.0,
            ),
            hypothesis=incoming_hypothesis,
            support_distance_seconds=0.0,
        ),
    )

    interstitial = InterstitialRoutingAssessment(
        routing_handoffs=(
            outgoing_hypothesis,
            incoming_hypothesis,
        ),
        supported_handoffs=(
            outgoing_hypothesis,
            incoming_hypothesis,
        ),
        absorbed_right_edge_fade=True,
    )

    assert (
        DefaultTransitionRouter.resolve_fade_interstitial_conflict(
            fade=fade,
            interstitial=interstitial,
        )
        is TransitionKind.INTERSTITIAL
    )


def test_fade_interstitial_conflict_prefers_interstitial_without_complete_fade_owner() -> None:
    from brynse.transition_router import (
        FadeRoutingAssessment,
        InterstitialRoutingAssessment,
    )

    left = _interstitial_hypothesis(
        basin_time=101.0,
        structural_time=101.2,
        basin_start=100.7,
        basin_recovery=101.6,
    )
    right = _interstitial_hypothesis(
        basin_time=103.0,
        structural_time=103.2,
        basin_start=102.7,
        basin_recovery=103.6,
    )

    fade = FadeRoutingAssessment(
        rise_present=True,
        basin_count=1,
        geometry_count=1,
    )

    interstitial = InterstitialRoutingAssessment(
        routing_handoffs=(left, right),
        supported_handoffs=(left, right),
    )

    assert fade.legacy_candidate is True
    assert fade.has_two_sided_support is False
    assert interstitial.legacy_candidate is True

    assert (
        DefaultTransitionRouter.resolve_fade_interstitial_conflict(
            fade=fade,
            interstitial=interstitial,
        )
        is TransitionKind.INTERSTITIAL
    )

def test_default_transition_router_keeps_unresolved_context_ambiguous() -> None:
    from brynse.transition_router import DefaultTransitionRouter

    result = DefaultTransitionRouter().classify(
        context=TransitionRoutingContext(
            reference_time_seconds=42.0,
            evidence=TransitionEvidenceState(),
        )
    )

    assert result is TransitionKind.AMBIGUOUS


def _complete_local_evidence() -> TransitionEvidenceState:
    return TransitionEvidenceState(
        local_discontinuity_250ms=0.21,
        local_discontinuity_500ms=0.43,
        local_discontinuity_1s=0.65,
        local_discontinuity_2s=0.87,
    )


def test_hard_cut_roles_require_same_unique_supported_handoff() -> None:
    from types import SimpleNamespace

    from brynse.transition_router import (
        FadeRoutingAssessment,
        HardCutRoutingAssessment,
        IncomingFadeRoutingAssociation,
        OutgoingFadeRoutingAssociation,
    )

    shared = _interstitial_hypothesis(
        basin_time=100.0,
        structural_time=100.2,
        basin_start=99.5,
        basin_recovery=100.5,
    )

    fade = FadeRoutingAssessment(
        rise_present=False,
        basin_count=0,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=SimpleNamespace(),
            hypothesis=shared,
            support_distance_seconds=0.0,
        ),
        incoming_association=IncomingFadeRoutingAssociation(
            fade=SimpleNamespace(),
            hypothesis=shared,
            support_distance_seconds=0.0,
        ),
    )

    hard_cut = HardCutRoutingAssessment(
        local_discontinuity_complete=True,
        fade_rise_present=False,
        supported_handoffs=(shared,),
    )

    roles = DefaultTransitionRouter.assign_hard_cut_roles(
        fade=fade,
        hard_cut=hard_cut,
    )

    assert roles is not None
    assert roles.outgoing_handoff is shared
    assert roles.incoming_handoff is shared
    assert roles.outgoing_handoff is roles.incoming_handoff


def test_hard_cut_roles_reject_different_semantic_handoffs() -> None:
    from types import SimpleNamespace

    from brynse.transition_router import (
        FadeRoutingAssessment,
        HardCutRoutingAssessment,
        IncomingFadeRoutingAssociation,
        OutgoingFadeRoutingAssociation,
    )

    outgoing = _interstitial_hypothesis(
        basin_time=100.0,
        structural_time=100.2,
        basin_start=99.5,
        basin_recovery=100.5,
    )
    incoming = _interstitial_hypothesis(
        basin_time=108.0,
        structural_time=108.2,
        basin_start=107.5,
        basin_recovery=108.5,
    )

    hard_cut = HardCutRoutingAssessment(
        local_discontinuity_complete=True,
        fade_rise_present=False,
        supported_handoffs=(outgoing,),
    )

    fade = FadeRoutingAssessment(
        rise_present=False,
        basin_count=0,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=SimpleNamespace(),
            hypothesis=outgoing,
            support_distance_seconds=0.0,
        ),
        incoming_association=IncomingFadeRoutingAssociation(
            fade=SimpleNamespace(),
            hypothesis=incoming,
            support_distance_seconds=0.0,
        ),
    )

    assert (
        DefaultTransitionRouter.assign_hard_cut_roles(
            fade=fade,
            hard_cut=hard_cut,
        )
        is None
    )


def test_hard_cut_roles_reject_incomplete_semantic_evidence() -> None:
    from types import SimpleNamespace

    from brynse.transition_router import (
        FadeRoutingAssessment,
        HardCutRoutingAssessment,
        OutgoingFadeRoutingAssociation,
    )

    shared = _interstitial_hypothesis(
        basin_time=100.0,
        structural_time=100.2,
        basin_start=99.5,
        basin_recovery=100.5,
    )

    hard_cut = HardCutRoutingAssessment(
        local_discontinuity_complete=True,
        fade_rise_present=False,
        supported_handoffs=(shared,),
    )

    fade = FadeRoutingAssessment(
        rise_present=False,
        basin_count=0,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=SimpleNamespace(),
            hypothesis=shared,
            support_distance_seconds=0.0,
        ),
    )

    assert (
        DefaultTransitionRouter.assign_hard_cut_roles(
            fade=fade,
            hard_cut=hard_cut,
        )
        is None
    )


def test_hard_cut_roles_reject_multiple_supported_handoffs() -> None:
    from types import SimpleNamespace

    from brynse.transition_router import (
        FadeRoutingAssessment,
        HardCutRoutingAssessment,
        IncomingFadeRoutingAssociation,
        OutgoingFadeRoutingAssociation,
    )

    first = _interstitial_hypothesis(
        basin_time=100.0,
        structural_time=100.2,
        basin_start=99.5,
        basin_recovery=100.5,
    )
    second = _interstitial_hypothesis(
        basin_time=108.0,
        structural_time=108.2,
        basin_start=107.5,
        basin_recovery=108.5,
    )

    fade = FadeRoutingAssessment(
        rise_present=False,
        basin_count=0,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=SimpleNamespace(),
            hypothesis=first,
            support_distance_seconds=0.0,
        ),
        incoming_association=IncomingFadeRoutingAssociation(
            fade=SimpleNamespace(),
            hypothesis=first,
            support_distance_seconds=0.0,
        ),
    )

    hard_cut = HardCutRoutingAssessment(
        local_discontinuity_complete=True,
        fade_rise_present=False,
        supported_handoffs=(first, second),
    )

    assert (
        DefaultTransitionRouter.assign_hard_cut_roles(
            fade=fade,
            hard_cut=hard_cut,
        )
        is None
    )


def test_hard_cut_route_requires_positive_shared_handoff_proof() -> None:
    from types import SimpleNamespace

    from brynse.transition_router import (
        FadeRoutingAssessment,
        HardCutRoutingAssessment,
        IncomingFadeRoutingAssociation,
        InterstitialRoutingAssessment,
        OutgoingFadeRoutingAssociation,
    )

    shared = _interstitial_hypothesis(
        basin_time=100.0,
        structural_time=100.2,
        basin_start=99.5,
        basin_recovery=100.5,
    )

    fade = FadeRoutingAssessment(
        rise_present=False,
        basin_count=0,
        geometry_count=0,
        outgoing_association=OutgoingFadeRoutingAssociation(
            fade=SimpleNamespace(),
            hypothesis=shared,
            support_distance_seconds=0.0,
        ),
        incoming_association=IncomingFadeRoutingAssociation(
            fade=SimpleNamespace(),
            hypothesis=shared,
            support_distance_seconds=0.0,
        ),
    )

    hard_cut = HardCutRoutingAssessment(
        local_discontinuity_complete=True,
        fade_rise_present=False,
        supported_handoffs=(shared,),
    )

    class Router(DefaultTransitionRouter):
        def assess_fade(self, *, context):
            return fade

        def assess_interstitial(self, *, context):
            return InterstitialRoutingAssessment()

        def assess_hard_cut(self, *, context):
            return hard_cut

    route = Router().route(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=TransitionEvidenceState(),
        ),
    )

    assert route.kind is TransitionKind.HARD_CUT
    assert route.hard_cut is hard_cut
    assert route.semantic_roles is not None
    assert route.semantic_roles.outgoing_handoff is shared
    assert route.semantic_roles.incoming_handoff is shared


def test_hard_cut_assessment_preserves_descriptive_fallback_evidence() -> None:
    router = DefaultTransitionRouter()

    assessment = router.assess_hard_cut(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=_complete_local_evidence(),
        )
    )

    assert assessment == HardCutRoutingAssessment(
        local_discontinuity_complete=True,
        fade_rise_present=False,
        supported_handoffs=(),
    )
    assert assessment.fallback_eligible is True


def test_hard_cut_assessment_does_not_invent_strength_threshold() -> None:
    router = DefaultTransitionRouter()

    assessment = router.assess_hard_cut(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=TransitionEvidenceState(
                local_discontinuity_250ms=0.0,
                local_discontinuity_500ms=0.0,
                local_discontinuity_1s=0.0,
                local_discontinuity_2s=0.0,
            ),
        )
    )

    assert assessment.local_discontinuity_complete is True
    assert assessment.fallback_eligible is True


def test_ambiguous_route_carries_hard_cut_assessment_provenance() -> None:
    router = DefaultTransitionRouter()

    route = router.route(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=_complete_local_evidence(),
        )
    )

    assert route.kind is TransitionKind.AMBIGUOUS
    assert route.hard_cut is not None
    assert route.hard_cut.local_discontinuity_complete is True
    assert route.hard_cut.fade_rise_present is False
    assert route.hard_cut.fallback_eligible is True


def test_default_router_keeps_complete_local_boundary_ambiguous_without_direct_proof() -> None:
    from brynse.transition_router import DefaultTransitionRouter

    result = DefaultTransitionRouter().classify(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=_complete_local_evidence(),
        )
    )

    assert result is TransitionKind.AMBIGUOUS


def test_default_transition_router_keeps_incomplete_local_boundary_ambiguous() -> None:
    from brynse.transition_router import DefaultTransitionRouter

    result = DefaultTransitionRouter().classify(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=TransitionEvidenceState(
                local_discontinuity_250ms=0.21,
                local_discontinuity_500ms=0.43,
                local_discontinuity_1s=0.65,
            ),
        )
    )

    assert result is TransitionKind.AMBIGUOUS


def test_default_transition_router_keeps_unpaired_fade_rise_ambiguous() -> None:
    from brynse.transition_router import DefaultTransitionRouter

    result = DefaultTransitionRouter().classify(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=_complete_local_evidence(),
            fade=FadeRoutingEvidence(
                rise=CrossfadeRiseEvidence(
                    time_seconds=104.0,
                    jump_ratio=2.5,
                    persistence_ratio=3.0,
                    after_global_ratio=2.0,
                ),
            ),
        )
    )

    assert result is TransitionKind.AMBIGUOUS


def test_default_transition_router_prefers_clear_interstitial_over_hard_cut_fallback() -> None:
    from brynse.transition_router import DefaultTransitionRouter

    hypotheses = (
        _interstitial_hypothesis(
            basin_time=101.0,
            structural_time=101.2,
            basin_start=100.7,
            basin_recovery=101.6,
        ),
        _interstitial_hypothesis(
            basin_time=103.0,
            structural_time=103.2,
            basin_start=102.7,
            basin_recovery=103.6,
        ),
    )

    result = DefaultTransitionRouter().classify(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=_complete_local_evidence(),
            hypotheses=hypotheses,
        )
    )

    assert result is TransitionKind.INTERSTITIAL


def _bounded_insert_edge(
    time_seconds: float,
    *,
    basin_start_seconds: float,
    basin_recovery_seconds: float,
    structural_strength: float,
) -> BoundaryHypothesis:
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


def _bounded_insert_region(
    *,
    reference_time_seconds: float,
    left: BoundaryHypothesis,
    right: BoundaryHypothesis,
    interior_hypotheses: tuple[BoundaryHypothesis, ...] = (),
) -> BoundedInsertRegionEvidence:
    return BoundedInsertRegionEvidence(
        reference_time_seconds=reference_time_seconds,
        left=left,
        right=right,
        interior_hypotheses=interior_hypotheses,
    )


def _structural_only_hypothesis(
    time_seconds: float,
    *,
    structural_strength: float,
) -> BoundaryHypothesis:
    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=(
            BoundaryProposal(
                time_seconds=time_seconds,
                source=BoundaryProposalSource.STRUCTURAL,
                strength=structural_strength,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("fade_rise_time_seconds", "expected"),
    [
        (110.1, TransitionKind.INTERSTITIAL),
        (112.0, TransitionKind.INTERSTITIAL),
    ],
)
def test_transition_router_correlates_right_edge_fade_with_interstitial(
    fade_rise_time_seconds: float,
    expected: TransitionKind,
) -> None:
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=1.0,
    )
    right = _bounded_insert_edge(
        110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.8,
    )
    interior = _structural_only_hypothesis(
        105.0,
        structural_strength=0.9,
    )

    region = _bounded_insert_region(
        reference_time_seconds=108.0,
        left=left,
        right=right,
        interior_hypotheses=(interior,),
    )

    fade = FadeRoutingEvidence(
        rise=CrossfadeRiseEvidence(
            time_seconds=130.0,
            jump_ratio=3.5,
            persistence_ratio=4.0,
            after_global_ratio=3.0,
        ),
        basins=(
            CrossfadeBasinEvidence(
                center_time_seconds=107.0,
                lead_seconds=1.0,
                width_seconds=0.7,
                prominence=1.5,
                quietness=1.8,
            ),
        ),
        outgoing_fades=(
            OutgoingFadeEvidence(
                start_time_seconds=106.0,
                floor_time_seconds=108.5,
                rise_time_seconds=fade_rise_time_seconds,
                start_rms=1200.0,
                floor_rms=120.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
        ),
    )

    result = DefaultTransitionRouter().classify(
        context=TransitionRoutingContext(
            reference_time_seconds=108.0,
            evidence=TransitionEvidenceState(),
            hypotheses=(left, interior, right),
            fade=fade,
            bounded_insert_regions=(region,),
        )
    )

    assert result is expected


def test_transition_router_does_not_absorb_fade_without_right_edge_geometry() -> None:
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=1.0,
    )

    right = BoundaryHypothesis(
        time_seconds=110.0,
        proposals=(
            BoundaryProposal(
                time_seconds=110.0,
                source=BoundaryProposalSource.STRUCTURAL,
                strength=0.8,
            ),
        ),
    )

    interior = _structural_only_hypothesis(
        105.0,
        structural_strength=0.9,
    )

    region = _bounded_insert_region(
        reference_time_seconds=108.0,
        left=left,
        right=right,
        interior_hypotheses=(interior,),
    )

    fade = FadeRoutingEvidence(
        rise=CrossfadeRiseEvidence(
            time_seconds=130.0,
            jump_ratio=3.5,
            persistence_ratio=4.0,
            after_global_ratio=3.0,
        ),
        basins=(
            CrossfadeBasinEvidence(
                center_time_seconds=107.0,
                lead_seconds=1.0,
                width_seconds=0.7,
                prominence=1.5,
                quietness=1.8,
            ),
        ),
        outgoing_fades=(
            OutgoingFadeEvidence(
                start_time_seconds=106.0,
                floor_time_seconds=108.5,
                rise_time_seconds=110.1,
                start_rms=1200.0,
                floor_rms=120.0,
                descent_ratio=10.0,
                descent_consistency=0.9,
            ),
        ),
    )

    context = TransitionRoutingContext(
        reference_time_seconds=108.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(left, interior, right),
        fade=fade,
        bounded_insert_regions=(region,),
    )

    router = DefaultTransitionRouter()

    assert region.right_geometry is None
    assert router._has_interstitial_absorbed_right_edge_fade(
        context=context,
    ) is False


def test_bounded_insert_shadow_accepts_unique_non_dominant_region() -> None:
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.2,
    )
    right = _bounded_insert_edge(
        108.0,
        basin_start_seconds=107.5,
        basin_recovery_seconds=108.5,
        structural_strength=0.3,
    )
    dominant = _bounded_insert_edge(
        120.0,
        basin_start_seconds=119.5,
        basin_recovery_seconds=120.5,
        structural_strength=0.9,
    )

    region = _bounded_insert_region(
        reference_time_seconds=101.0,
        left=left,
        right=right,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=101.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(left, right, dominant),
        bounded_insert_regions=(region,),
    )

    result = DefaultTransitionRouter().assess_bounded_insert(
        context=context,
    )

    assert result.candidate is True
    assert result.unique_region is True
    assert result.related_region_count == 1
    assert result.left_is_context_structural_peak is False
    assert result.right_is_context_structural_peak is False


def test_bounded_insert_rejects_left_basin_shared_handoff() -> None:
    from types import SimpleNamespace

    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.0,
        basin_recovery_seconds=102.0,
        structural_strength=0.2,
    )
    right = _bounded_insert_edge(
        112.0,
        basin_start_seconds=111.5,
        basin_recovery_seconds=112.5,
        structural_strength=0.3,
    )
    interior = _structural_only_hypothesis(
        106.0,
        structural_strength=0.25,
    )
    dominant = _bounded_insert_edge(
        120.0,
        basin_start_seconds=119.5,
        basin_recovery_seconds=120.5,
        structural_strength=0.9,
    )

    region = _bounded_insert_region(
        reference_time_seconds=100.2,
        left=left,
        right=right,
        interior_hypotheses=(interior,),
    )

    shared_outgoing = SimpleNamespace(
        floor_time_seconds=100.4,
    )
    shared_incoming = SimpleNamespace(
        start_time_seconds=100.4,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=100.2,
        evidence=TransitionEvidenceState(),
        hypotheses=(left, interior, right, dominant),
        fade=SimpleNamespace(
            geometries=(
                SimpleNamespace(
                    outgoing=shared_outgoing,
                    incoming=shared_incoming,
                ),
            ),
        ),
        bounded_insert_regions=(region,),
    )

    assessment = DefaultTransitionRouter().assess_bounded_insert(
        context=context,
    )

    assert region.reference_inside_left_basin is True
    assert assessment.candidate is False
    assert assessment.unique_region is True


def test_bounded_insert_preserves_region_when_shared_handoff_precedes_reference() -> None:
    from types import SimpleNamespace

    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.0,
        basin_recovery_seconds=102.0,
        structural_strength=0.2,
    )
    right = _bounded_insert_edge(
        112.0,
        basin_start_seconds=111.5,
        basin_recovery_seconds=112.5,
        structural_strength=0.3,
    )
    interior = _structural_only_hypothesis(
        106.0,
        structural_strength=0.25,
    )
    dominant = _bounded_insert_edge(
        120.0,
        basin_start_seconds=119.5,
        basin_recovery_seconds=120.5,
        structural_strength=0.9,
    )

    region = _bounded_insert_region(
        reference_time_seconds=100.2,
        left=left,
        right=right,
        interior_hypotheses=(interior,),
    )

    earlier_outgoing = SimpleNamespace(
        floor_time_seconds=99.9,
    )
    earlier_incoming = SimpleNamespace(
        start_time_seconds=99.9,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=100.2,
        evidence=TransitionEvidenceState(),
        hypotheses=(left, interior, right, dominant),
        fade=SimpleNamespace(
            geometries=(
                SimpleNamespace(
                    outgoing=earlier_outgoing,
                    incoming=earlier_incoming,
                ),
            ),
        ),
        bounded_insert_regions=(region,),
    )

    assessment = DefaultTransitionRouter().assess_bounded_insert(
        context=context,
    )

    assert region.reference_inside_left_basin is True
    assert assessment.candidate is True


def test_bounded_insert_preserves_region_without_shared_left_handoff() -> None:
    from types import SimpleNamespace

    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.0,
        basin_recovery_seconds=102.0,
        structural_strength=0.2,
    )
    right = _bounded_insert_edge(
        112.0,
        basin_start_seconds=111.5,
        basin_recovery_seconds=112.5,
        structural_strength=0.3,
    )
    interior = _structural_only_hypothesis(
        106.0,
        structural_strength=0.25,
    )
    dominant = _bounded_insert_edge(
        120.0,
        basin_start_seconds=119.5,
        basin_recovery_seconds=120.5,
        structural_strength=0.9,
    )

    region = _bounded_insert_region(
        reference_time_seconds=100.2,
        left=left,
        right=right,
        interior_hypotheses=(interior,),
    )

    outgoing = SimpleNamespace(
        floor_time_seconds=100.4,
    )
    incoming = SimpleNamespace(
        start_time_seconds=100.6,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=100.2,
        evidence=TransitionEvidenceState(),
        hypotheses=(left, interior, right, dominant),
        fade=SimpleNamespace(
            geometries=(
                SimpleNamespace(
                    outgoing=outgoing,
                    incoming=incoming,
                ),
            ),
        ),
        bounded_insert_regions=(region,),
    )

    assessment = DefaultTransitionRouter().assess_bounded_insert(
        context=context,
    )

    assert region.reference_inside_left_basin is True
    assert assessment.candidate is True


def test_bounded_insert_shadow_abstains_with_multiple_regions() -> None:
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.2,
    )
    middle = _bounded_insert_edge(
        105.0,
        basin_start_seconds=104.5,
        basin_recovery_seconds=105.5,
        structural_strength=0.3,
    )
    right = _bounded_insert_edge(
        110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.4,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=105.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(left, middle, right),
        bounded_insert_regions=(
            _bounded_insert_region(
                reference_time_seconds=105.0,
                left=left,
                right=middle,
            ),
            _bounded_insert_region(
                reference_time_seconds=105.0,
                left=middle,
                right=right,
            ),
        ),
    )

    result = DefaultTransitionRouter().assess_bounded_insert(
        context=context,
    )

    assert result.candidate is False
    assert result.unique_region is False
    assert result.related_region_count == 2


def test_bounded_insert_shadow_rejects_structurally_dominant_edge() -> None:
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=1.0,
    )
    right = _bounded_insert_edge(
        108.0,
        basin_start_seconds=107.5,
        basin_recovery_seconds=108.5,
        structural_strength=0.3,
    )

    region = _bounded_insert_region(
        reference_time_seconds=101.0,
        left=left,
        right=right,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=101.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(left, right),
        bounded_insert_regions=(region,),
    )

    result = DefaultTransitionRouter().assess_bounded_insert(
        context=context,
    )

    assert result.candidate is False
    assert result.left_is_context_structural_peak is True
    assert result.right_is_context_structural_peak is False


def test_bounded_insert_shadow_rejects_basin_entry_with_internal_structure() -> None:
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=98.0,
        basin_recovery_seconds=104.0,
        structural_strength=1.0,
    )
    right = _bounded_insert_edge(
        118.0,
        basin_start_seconds=117.5,
        basin_recovery_seconds=118.5,
        structural_strength=0.2,
    )
    interior_proposal = BoundaryProposal(
        time_seconds=108.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.4,
    )
    interior = BoundaryHypothesis(
        time_seconds=108.0,
        proposals=(interior_proposal,),
    )

    region = _bounded_insert_region(
        reference_time_seconds=99.5,
        left=left,
        right=right,
        interior_hypotheses=(interior,),
    )

    context = TransitionRoutingContext(
        reference_time_seconds=99.5,
        evidence=TransitionEvidenceState(),
        hypotheses=(left, interior, right),
        bounded_insert_regions=(region,),
    )

    result = DefaultTransitionRouter().assess_bounded_insert(
        context=context,
    )

    assert result.candidate is False
    assert result.left_is_context_structural_peak is True
    assert result.right_is_context_structural_peak is False
    assert result.reference_precedes_left_support_inside_basin is True
    assert result.structural_only_interior_count == 1


def _asymmetric_region(
    *,
    reference_time_seconds,
    primary,
    complementary,
    interior_hypotheses=(),
):
    from brynse.bounded_insert import (
        BoundedInsertAsymmetricRegionEvidence,
    )

    return BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=reference_time_seconds,
        primary=primary,
        complementary=complementary,
        interior_hypotheses=interior_hypotheses,
    )


def _structural_only_boundary(
    time_seconds,
    *,
    structural_strength,
):
    proposal = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=structural_strength,
    )
    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=(proposal,),
    )


def test_asymmetric_bounded_insert_accepts_structural_peak_primary() -> None:
    primary = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.8,
    )
    interior = _structural_only_boundary(
        108.0,
        structural_strength=0.2,
    )
    complementary = _structural_only_boundary(
        119.0,
        structural_strength=0.3,
    )

    region = _asymmetric_region(
        reference_time_seconds=118.5,
        primary=primary,
        complementary=complementary,
        interior_hypotheses=(interior,),
    )

    context = TransitionRoutingContext(
        reference_time_seconds=118.5,
        evidence=TransitionEvidenceState(),
        hypotheses=(primary, interior, complementary),
        bounded_insert_asymmetric_regions=(region,),
    )

    result = DefaultTransitionRouter().assess_asymmetric_bounded_insert(
        context=context,
    )

    assert result.candidate is True
    assert result.related_region_count == 1
    assert result.unique_region is True
    assert result.primary_is_context_structural_peak is True


def test_asymmetric_bounded_insert_rejects_non_peak_primary() -> None:
    primary = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.5,
    )
    complementary = _structural_only_boundary(
        119.0,
        structural_strength=0.3,
    )
    dominant = _structural_only_boundary(
        125.0,
        structural_strength=0.9,
    )

    region = _asymmetric_region(
        reference_time_seconds=118.5,
        primary=primary,
        complementary=complementary,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=118.5,
        evidence=TransitionEvidenceState(),
        hypotheses=(primary, complementary, dominant),
        bounded_insert_asymmetric_regions=(region,),
    )

    result = DefaultTransitionRouter().assess_asymmetric_bounded_insert(
        context=context,
    )

    assert result.candidate is False
    assert result.related_region_count == 1
    assert result.unique_region is True
    assert result.primary_is_context_structural_peak is False


def test_asymmetric_bounded_insert_abstains_with_multiple_regions() -> None:
    primary = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.8,
    )
    first = _structural_only_boundary(
        119.0,
        structural_strength=0.3,
    )
    second = _structural_only_boundary(
        121.0,
        structural_strength=0.2,
    )

    first_region = _asymmetric_region(
        reference_time_seconds=118.5,
        primary=primary,
        complementary=first,
    )
    second_region = _asymmetric_region(
        reference_time_seconds=118.5,
        primary=primary,
        complementary=second,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=118.5,
        evidence=TransitionEvidenceState(),
        hypotheses=(primary, first, second),
        bounded_insert_asymmetric_regions=(
            first_region,
            second_region,
        ),
    )

    result = DefaultTransitionRouter().assess_asymmetric_bounded_insert(
        context=context,
    )

    assert result.candidate is False
    assert result.related_region_count == 2
    assert result.unique_region is False


def test_asymmetric_ownership_routes_to_bounded_insert() -> None:
    primary = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.8,
    )
    complementary = _structural_only_boundary(
        119.0,
        structural_strength=0.3,
    )

    region = _asymmetric_region(
        reference_time_seconds=118.5,
        primary=primary,
        complementary=complementary,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=118.5,
        evidence=TransitionEvidenceState(),
        hypotheses=(primary, complementary),
        bounded_insert_asymmetric_regions=(region,),
    )

    router = DefaultTransitionRouter()

    assessment = router.assess_asymmetric_bounded_insert(
        context=context,
    )
    routed = router.classify(context=context)

    assert assessment.candidate is True
    assert routed is TransitionKind.BOUNDED_INSERT


@pytest.mark.parametrize(
    (
        "reference_time_seconds",
        "primary_time_seconds",
        "primary_start_seconds",
        "primary_recovery_seconds",
        "complementary_kind",
        "complementary_time_seconds",
    ),
    (
        (
            1363.958,
            1344.876,
            1344.251,
            1348.051,
            "structural",
            1364.451,
        ),
        (
            1780.376,
            1778.519,
            1773.069,
            1778.769,
            "basin",
            1781.919,
        ),
    ),
)
def test_live_asymmetric_failures_receive_router_ownership(
    reference_time_seconds,
    primary_time_seconds,
    primary_start_seconds,
    primary_recovery_seconds,
    complementary_kind,
    complementary_time_seconds,
) -> None:
    from brynse.bounded_insert import BoundedInsertRegionDiscovery

    primary = _bounded_insert_edge(
        primary_time_seconds,
        basin_start_seconds=primary_start_seconds,
        basin_recovery_seconds=primary_recovery_seconds,
        structural_strength=0.8,
    )

    if complementary_kind == "structural":
        complementary = _structural_only_boundary(
            complementary_time_seconds,
            structural_strength=0.3,
        )
    else:
        proposal = BoundaryProposal(
            time_seconds=complementary_time_seconds,
            source=BoundaryProposalSource.BASIN,
        )
        complementary = BoundaryHypothesis(
            time_seconds=complementary_time_seconds,
            proposals=(proposal,),
            evidence=(
                BoundaryEvidence(
                    proposal=proposal,
                    basin_start_seconds=1781.169,
                    basin_recovery_seconds=1782.069,
                ),
            ),
        )

    hypotheses = (
        primary,
        complementary,
    )

    regions = BoundedInsertRegionDiscovery().discover_asymmetric(
        reference_time_seconds=reference_time_seconds,
        hypotheses=hypotheses,
    )

    assert len(regions) == 1

    context = TransitionRoutingContext(
        reference_time_seconds=reference_time_seconds,
        evidence=TransitionEvidenceState(),
        hypotheses=hypotheses,
        bounded_insert_asymmetric_regions=regions,
    )

    router = DefaultTransitionRouter()

    assessment = router.assess_asymmetric_bounded_insert(
        context=context,
    )

    assert assessment.candidate is True
    assert assessment.unique_region is True
    assert assessment.primary_is_context_structural_peak is True

    assert (
        router.classify(context=context)
        is TransitionKind.BOUNDED_INSERT
    )


def test_bounded_insert_roles_preserve_symmetric_region_and_outer_edges() -> None:
    router = DefaultTransitionRouter()

    predecessor = _bounded_insert_edge(
        90.0,
        basin_start_seconds=89.5,
        basin_recovery_seconds=90.5,
        structural_strength=1.0,
    )
    bridge = _structural_only_hypothesis(
        95.0,
        structural_strength=0.8,
    )
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.3,
    )
    right = _bounded_insert_edge(
        110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.2,
    )

    region = _bounded_insert_region(
        reference_time_seconds=105.0,
        left=left,
        right=right,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=105.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(predecessor, bridge, left, right),
        bounded_insert_regions=(region,),
    )

    provenance = router.assess_bounded_insert_routing(
        context=context,
    )

    roles = router.assign_bounded_insert_roles(
        context=context,
        provenance=provenance,
        fade=router.assess_fade(context=context),
    )

    assert provenance.ownership is BoundedInsertOwnership.SYMMETRIC
    assert roles is not None
    assert roles.inserted_region is region
    assert roles.outgoing_handoff is predecessor
    assert roles.incoming_handoff is right


def test_bounded_insert_roles_prefer_adjacent_dominant_terminal_decay() -> None:
    from brynse.transition_evidence import (
        FadeRoutingEvidence,
        OutgoingFadeEvidence,
    )

    router = DefaultTransitionRouter()

    predecessor = _bounded_insert_edge(
        90.0,
        basin_start_seconds=89.5,
        basin_recovery_seconds=90.5,
        structural_strength=0.8,
    )
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.3,
    )
    right = _bounded_insert_edge(
        110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.2,
    )

    region = _bounded_insert_region(
        reference_time_seconds=105.0,
        left=left,
        right=right,
    )

    terminal = OutgoingFadeEvidence(
        start_time_seconds=84.0,
        floor_time_seconds=89.5,
        rise_time_seconds=None,
        start_rms=5000.0,
        floor_rms=80.0,
        descent_ratio=40.0,
        descent_consistency=0.75,
    )
    local = OutgoingFadeEvidence(
        start_time_seconds=96.0,
        floor_time_seconds=100.0,
        rise_time_seconds=None,
        start_rms=4000.0,
        floor_rms=200.0,
        descent_ratio=20.0,
        descent_consistency=0.8,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=105.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(
            predecessor,
            left,
            right,
        ),
        fade=FadeRoutingEvidence(
            outgoing_fades=(
                terminal,
                local,
            ),
        ),
        bounded_insert_regions=(region,),
    )

    fade = router.assess_fade(context=context)

    assert fade.outgoing_association is not None
    assert fade.outgoing_association.fade is local
    assert fade.outgoing_association.hypothesis is left

    provenance = router.assess_bounded_insert_routing(
        context=context,
    )

    roles = router.assign_bounded_insert_roles(
        context=context,
        provenance=provenance,
        fade=fade,
    )

    assert provenance.ownership is BoundedInsertOwnership.SYMMETRIC
    assert roles is not None
    assert roles.inserted_region is region
    assert roles.outgoing_handoff is predecessor
    assert roles.incoming_handoff is right


def test_bounded_insert_roles_keep_left_without_adjacent_fade_dominance() -> None:
    from brynse.transition_evidence import (
        FadeRoutingEvidence,
        OutgoingFadeEvidence,
    )

    router = DefaultTransitionRouter()

    predecessor = _bounded_insert_edge(
        90.0,
        basin_start_seconds=89.5,
        basin_recovery_seconds=90.5,
        structural_strength=0.8,
    )
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.3,
    )
    right = _bounded_insert_edge(
        110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.2,
    )

    region = _bounded_insert_region(
        reference_time_seconds=105.0,
        left=left,
        right=right,
    )

    predecessor_fade = OutgoingFadeEvidence(
        start_time_seconds=86.0,
        floor_time_seconds=89.5,
        rise_time_seconds=None,
        start_rms=3000.0,
        floor_rms=250.0,
        descent_ratio=12.0,
        descent_consistency=0.8,
    )
    local = OutgoingFadeEvidence(
        start_time_seconds=96.0,
        floor_time_seconds=100.0,
        rise_time_seconds=None,
        start_rms=4000.0,
        floor_rms=200.0,
        descent_ratio=20.0,
        descent_consistency=0.8,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=105.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(
            predecessor,
            left,
            right,
        ),
        fade=FadeRoutingEvidence(
            outgoing_fades=(
                predecessor_fade,
                local,
            ),
        ),
        bounded_insert_regions=(region,),
    )

    fade = router.assess_fade(context=context)

    assert fade.outgoing_association is not None
    assert fade.outgoing_association.fade is local
    assert fade.outgoing_association.hypothesis is left

    provenance = router.assess_bounded_insert_routing(
        context=context,
    )

    roles = router.assign_bounded_insert_roles(
        context=context,
        provenance=provenance,
        fade=fade,
    )

    assert provenance.ownership is BoundedInsertOwnership.SYMMETRIC
    assert roles is not None
    assert roles.inserted_region is region
    assert roles.outgoing_handoff is left
    assert roles.incoming_handoff is right


def test_bounded_insert_roles_allow_partial_symmetric_exterior_identity() -> None:
    router = DefaultTransitionRouter()

    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.3,
    )
    right = _bounded_insert_edge(
        110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.2,
    )
    dominant = _bounded_insert_edge(
        120.0,
        basin_start_seconds=119.5,
        basin_recovery_seconds=120.5,
        structural_strength=1.0,
    )

    region = _bounded_insert_region(
        reference_time_seconds=105.0,
        left=left,
        right=right,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=105.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(left, right, dominant),
        bounded_insert_regions=(region,),
    )

    provenance = router.assess_bounded_insert_routing(
        context=context,
    )

    roles = router.assign_bounded_insert_roles(
        context=context,
        provenance=provenance,
        fade=router.assess_fade(context=context),
    )

    assert provenance.ownership is BoundedInsertOwnership.SYMMETRIC
    assert roles is not None
    assert roles.inserted_region is region
    assert roles.outgoing_handoff is None
    assert roles.incoming_handoff is right


def test_bounded_insert_roles_assign_asymmetric_outer_identities() -> None:
    from brynse.bounded_insert import (
        BoundedInsertAsymmetricRegionEvidence,
    )

    router = DefaultTransitionRouter()

    predecessor = _bounded_insert_edge(
        90.0,
        basin_start_seconds=89.5,
        basin_recovery_seconds=90.5,
        structural_strength=0.5,
    )
    primary = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=1.0,
    )
    complementary = _structural_only_hypothesis(
        105.0,
        structural_strength=0.3,
    )
    outer_right = _bounded_insert_edge(
        110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.2,
    )

    asymmetric = BoundedInsertAsymmetricRegionEvidence(
        reference_time_seconds=105.0,
        primary=primary,
        complementary=complementary,
    )
    outer = _bounded_insert_region(
        reference_time_seconds=105.0,
        left=primary,
        right=outer_right,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=105.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(
            predecessor,
            primary,
            complementary,
            outer_right,
        ),
        bounded_insert_regions=(outer,),
        bounded_insert_asymmetric_regions=(asymmetric,),
    )

    provenance = router.assess_bounded_insert_routing(
        context=context,
    )

    roles = router.assign_bounded_insert_roles(
        context=context,
        provenance=provenance,
        fade=router.assess_fade(context=context),
    )

    assert provenance.ownership is BoundedInsertOwnership.ASYMMETRIC
    assert roles is not None
    assert roles.inserted_region is asymmetric
    assert roles.outgoing_handoff is predecessor
    assert roles.incoming_handoff is outer_right


def test_bounded_insert_route_carries_semantic_roles() -> None:
    router = DefaultTransitionRouter()

    predecessor = _bounded_insert_edge(
        90.0,
        basin_start_seconds=89.5,
        basin_recovery_seconds=90.5,
        structural_strength=1.0,
    )
    bridge = _structural_only_hypothesis(
        95.0,
        structural_strength=0.8,
    )
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.3,
    )
    right = _bounded_insert_edge(
        110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.2,
    )

    region = _bounded_insert_region(
        reference_time_seconds=105.0,
        left=left,
        right=right,
    )

    route = router.route(
        context=TransitionRoutingContext(
            reference_time_seconds=105.0,
            evidence=TransitionEvidenceState(),
            hypotheses=(predecessor, bridge, left, right),
            bounded_insert_regions=(region,),
        ),
    )

    assert route.kind is TransitionKind.BOUNDED_INSERT
    assert route.semantic_roles is not None
    assert route.semantic_roles.inserted_region is region
    assert route.semantic_roles.outgoing_handoff is predecessor
    assert route.semantic_roles.incoming_handoff is right


def test_bounded_insert_routing_provenance_preserves_symmetric_region() -> None:
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.2,
    )
    right = _bounded_insert_edge(
        110.0,
        basin_start_seconds=109.5,
        basin_recovery_seconds=110.5,
        structural_strength=0.3,
    )
    dominant = _structural_only_boundary(
        115.0,
        structural_strength=0.9,
    )

    region = _bounded_insert_region(
        reference_time_seconds=105.0,
        left=left,
        right=right,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=105.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(left, right, dominant),
        bounded_insert_regions=(region,),
    )

    router = DefaultTransitionRouter()

    provenance = router.assess_bounded_insert_routing(
        context=context,
    )

    assert provenance.ownership is BoundedInsertOwnership.SYMMETRIC
    assert provenance.symmetric_assessment.candidate is True
    assert provenance.symmetric_region is region
    assert provenance.asymmetric_region is None
    assert provenance.owned_region is region

    route = router.route(
        context=context,
    )

    assert route.kind is TransitionKind.BOUNDED_INSERT
    assert route.bounded_insert is not None
    assert route.bounded_insert is not provenance
    assert (
        route.bounded_insert.ownership
        is BoundedInsertOwnership.SYMMETRIC
    )
    assert route.bounded_insert.symmetric_region is region


def test_bounded_insert_routing_provenance_preserves_asymmetric_region() -> None:
    primary = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.8,
    )
    complementary = _structural_only_boundary(
        119.0,
        structural_strength=0.3,
    )

    region = _asymmetric_region(
        reference_time_seconds=118.5,
        primary=primary,
        complementary=complementary,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=118.5,
        evidence=TransitionEvidenceState(),
        hypotheses=(primary, complementary),
        bounded_insert_asymmetric_regions=(region,),
    )

    provenance = DefaultTransitionRouter().assess_bounded_insert_routing(
        context=context,
    )

    assert provenance.ownership is BoundedInsertOwnership.ASYMMETRIC
    assert provenance.symmetric_assessment.candidate is False
    assert provenance.asymmetric_assessment.candidate is True
    assert provenance.symmetric_region is None
    assert provenance.asymmetric_region is region
    assert provenance.owned_region is region


def test_bounded_insert_ownership_reports_symmetric_precedence() -> None:
    from types import SimpleNamespace

    from brynse.transition_router import (
        BoundedInsertOwnership,
        DefaultTransitionRouter,
    )

    class Router(DefaultTransitionRouter):
        def assess_bounded_insert(self, *, context):
            _ = context
            return SimpleNamespace(candidate=True)

        def assess_asymmetric_bounded_insert(self, *, context):
            _ = context
            return SimpleNamespace(candidate=True)

    assert (
        Router().bounded_insert_ownership(
            context=object(),
        )
        is BoundedInsertOwnership.SYMMETRIC
    )


def test_bounded_insert_ownership_reports_asymmetric_branch() -> None:
    from types import SimpleNamespace

    from brynse.transition_router import (
        BoundedInsertOwnership,
        DefaultTransitionRouter,
    )

    class Router(DefaultTransitionRouter):
        def assess_bounded_insert(self, *, context):
            _ = context
            return SimpleNamespace(candidate=False)

        def assess_asymmetric_bounded_insert(self, *, context):
            _ = context
            return SimpleNamespace(candidate=True)

    assert (
        Router().bounded_insert_ownership(
            context=object(),
        )
        is BoundedInsertOwnership.ASYMMETRIC
    )


def test_bounded_insert_ownership_reports_none_without_candidate() -> None:
    from types import SimpleNamespace

    from brynse.transition_router import DefaultTransitionRouter

    class Router(DefaultTransitionRouter):
        def assess_bounded_insert(self, *, context):
            _ = context
            return SimpleNamespace(candidate=False)

        def assess_asymmetric_bounded_insert(self, *, context):
            _ = context
            return SimpleNamespace(candidate=False)

    assert (
        Router().bounded_insert_ownership(
            context=object(),
        )
        is None
    )


def test_phase3_symmetric_bounded_insert_rejects_reference_after_right_support() -> None:
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource
    from brynse.transition_evidence import TransitionEvidenceState
    from brynse.transition_router import (
        DefaultTransitionRouter,
        TransitionRoutingContext,
    )

    def structural(strength):
        return SimpleNamespace(
            source=BoundaryProposalSource.STRUCTURAL,
            strength=strength,
        )

    left = SimpleNamespace(
        time_seconds=100.0,
        proposals=(structural(0.2),),
    )
    right = SimpleNamespace(
        time_seconds=110.0,
        proposals=(structural(0.3),),
    )

    region = SimpleNamespace(
        reference_time_seconds=111.0,
        left=left,
        right=right,
        left_edge=left,
        right_edge=right,
        reference_inside_left_basin=False,
        structural_only_interior_count=0,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=111.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(
            SimpleNamespace(
                proposals=(structural(0.9),),
            ),
        ),
        bounded_insert_regions=(region,),
    )

    assessment = DefaultTransitionRouter().assess_bounded_insert(
        context=context,
    )

    assert assessment.candidate is False
    assert assessment.unique_region is True
    assert assessment.left_is_context_structural_peak is False
    assert assessment.right_is_context_structural_peak is False




def test_phase4_symmetric_accepts_single_structural_interior_before_dominant_right_edge() -> None:
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.3,
    )
    right = _bounded_insert_edge(
        120.0,
        basin_start_seconds=119.5,
        basin_recovery_seconds=120.5,
        structural_strength=1.0,
    )

    interior_proposal = BoundaryProposal(
        time_seconds=108.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.4,
    )
    interior = BoundaryHypothesis(
        time_seconds=108.0,
        proposals=(interior_proposal,),
    )

    region = _bounded_insert_region(
        reference_time_seconds=118.0,
        left=left,
        right=right,
        interior_hypotheses=(interior,),
    )

    context = TransitionRoutingContext(
        reference_time_seconds=118.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(left, interior, right),
        bounded_insert_regions=(region,),
    )

    router = DefaultTransitionRouter()

    assessment = router.assess_bounded_insert(
        context=context,
    )

    assert assessment.candidate is True
    assert assessment.left_is_context_structural_peak is False
    assert assessment.right_is_context_structural_peak is True
    assert assessment.structural_only_interior_count == 1
    assert (
        router.classify(context=context)
        is TransitionKind.BOUNDED_INSERT
    )


def test_phase4_symmetric_rejects_multiple_structural_interiors_before_dominant_right_edge(
) -> None:
    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.3,
    )
    right = _bounded_insert_edge(
        120.0,
        basin_start_seconds=119.5,
        basin_recovery_seconds=120.5,
        structural_strength=1.0,
    )

    first_proposal = BoundaryProposal(
        time_seconds=106.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.4,
    )
    first = BoundaryHypothesis(
        time_seconds=106.0,
        proposals=(first_proposal,),
    )

    second_proposal = BoundaryProposal(
        time_seconds=112.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.5,
    )
    second = BoundaryHypothesis(
        time_seconds=112.0,
        proposals=(second_proposal,),
    )

    region = _bounded_insert_region(
        reference_time_seconds=118.0,
        left=left,
        right=right,
        interior_hypotheses=(first, second),
    )

    context = TransitionRoutingContext(
        reference_time_seconds=118.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(
            left,
            first,
            second,
            right,
        ),
        bounded_insert_regions=(region,),
    )

    router = DefaultTransitionRouter()

    assessment = router.assess_bounded_insert(
        context=context,
    )

    assert assessment.candidate is False
    assert assessment.left_is_context_structural_peak is False
    assert assessment.right_is_context_structural_peak is True
    assert assessment.structural_only_interior_count == 2
    assert (
        router.classify(context=context)
        is not TransitionKind.BOUNDED_INSERT
    )



def test_phase4_symmetric_rejects_mixed_interior_families() -> None:
    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_evidence import TransitionEvidenceState
    from brynse.transition_router import (
        DefaultTransitionRouter,
        TransitionRoutingContext,
    )

    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.3,
    )
    right = _bounded_insert_edge(
        120.0,
        basin_start_seconds=119.5,
        basin_recovery_seconds=120.5,
        structural_strength=0.2,
    )

    structural_proposal = BoundaryProposal(
        time_seconds=106.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.25,
    )
    structural_only = BoundaryHypothesis(
        time_seconds=106.0,
        proposals=(structural_proposal,),
    )

    basin_proposal = BoundaryProposal(
        time_seconds=112.0,
        source=BoundaryProposalSource.BASIN,
    )
    basin_only = BoundaryHypothesis(
        time_seconds=112.0,
        proposals=(basin_proposal,),
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=104.0,
        left=left,
        right=right,
        interior_hypotheses=(
            structural_only,
            basin_only,
        ),
    )

    context = TransitionRoutingContext(
        reference_time_seconds=104.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(
            left,
            structural_only,
            basin_only,
            right,
        ),
        bounded_insert_regions=(region,),
    )

    result = DefaultTransitionRouter().assess_bounded_insert(
        context=context,
    )

    assert result.candidate is False


def test_phase4_symmetric_accepts_structural_only_interior() -> None:
    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_evidence import TransitionEvidenceState
    from brynse.transition_router import (
        DefaultTransitionRouter,
        TransitionRoutingContext,
    )

    left = _bounded_insert_edge(
        100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.3,
    )
    right = _bounded_insert_edge(
        120.0,
        basin_start_seconds=119.5,
        basin_recovery_seconds=120.5,
        structural_strength=0.2,
    )

    proposal = BoundaryProposal(
        time_seconds=110.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.4,
    )
    interior = BoundaryHypothesis(
        time_seconds=110.0,
        proposals=(proposal,),
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=105.0,
        left=left,
        right=right,
        interior_hypotheses=(interior,),
    )

    context = TransitionRoutingContext(
        reference_time_seconds=105.0,
        evidence=TransitionEvidenceState(),
        hypotheses=(
            left,
            interior,
            right,
        ),
        bounded_insert_regions=(region,),
    )

    result = DefaultTransitionRouter().assess_bounded_insert(
        context=context,
    )

    assert result.candidate is True



def test_interstitial_roles_use_bracketing_reconciled_edge_geometry() -> None:
    from types import SimpleNamespace

    from brynse.transition_router import (
        DefaultTransitionRouter,
        FadeRoutingAssessment,
        InterstitialRoutingAssessment,
    )

    outgoing_handoff = _interstitial_hypothesis(
        basin_time=105.7,
        structural_time=105.8,
        basin_start=105.2,
        basin_recovery=106.2,
    )
    later_handoff = _interstitial_hypothesis(
        basin_time=123.5,
        structural_time=123.6,
        basin_start=123.0,
        basin_recovery=124.0,
    )

    left_outgoing = SimpleNamespace(
        floor_time_seconds=105.7,
    )
    left_incoming = SimpleNamespace(
        start_time_seconds=105.7,
        established_time_seconds=110.8,
    )
    right_outgoing = SimpleNamespace(
        floor_time_seconds=124.2,
    )
    right_incoming = SimpleNamespace(
        start_time_seconds=123.5,
        established_time_seconds=124.0,
    )

    context = SimpleNamespace(
        reference_time_seconds=121.5,
        bounded_insert_regions=(),
        fade=SimpleNamespace(
            geometries=(
                SimpleNamespace(
                    outgoing=left_outgoing,
                    incoming=left_incoming,
                ),
                SimpleNamespace(
                    outgoing=right_outgoing,
                    incoming=right_incoming,
                ),
            ),
        ),
    )

    roles = DefaultTransitionRouter.assign_interstitial_roles(
        fade=FadeRoutingAssessment(
            rise_present=False,
            basin_count=0,
            geometry_count=2,
        ),
        interstitial=InterstitialRoutingAssessment(
            routing_handoffs=(
                outgoing_handoff,
                later_handoff,
            ),
            supported_handoffs=(
                outgoing_handoff,
                later_handoff,
            ),
        ),
        context=context,
    )

    assert roles is not None
    assert roles.outgoing_handoff is outgoing_handoff
    assert roles.incoming_handoff is later_handoff
    assert roles.outgoing_edge_fade is left_outgoing
    assert roles.incoming_edge_fade is right_incoming



def test_interstitial_roles_reject_unowned_bracketing_fade_geometry() -> None:
    from types import SimpleNamespace

    from brynse.transition_router import (
        DefaultTransitionRouter,
        FadeRoutingAssessment,
        InterstitialRoutingAssessment,
    )

    outgoing_handoff = _interstitial_hypothesis(
        basin_time=1030.6,
        structural_time=1030.7,
        basin_start=1030.1,
        basin_recovery=1031.1,
    )
    unrelated_handoff = _interstitial_hypothesis(
        basin_time=1045.0,
        structural_time=1045.1,
        basin_start=1044.5,
        basin_recovery=1045.5,
    )

    owned_outgoing = SimpleNamespace(
        floor_time_seconds=1030.6,
    )
    owned_incoming = SimpleNamespace(
        start_time_seconds=1030.6,
        established_time_seconds=1030.9,
    )

    unowned_outgoing = SimpleNamespace(
        floor_time_seconds=1039.4,
    )
    unowned_incoming = SimpleNamespace(
        start_time_seconds=1039.0,
        established_time_seconds=1039.5,
    )

    context = SimpleNamespace(
        reference_time_seconds=1030.8,
        bounded_insert_regions=(),
        fade=SimpleNamespace(
            geometries=(
                SimpleNamespace(
                    outgoing=owned_outgoing,
                    incoming=owned_incoming,
                ),
                SimpleNamespace(
                    outgoing=unowned_outgoing,
                    incoming=unowned_incoming,
                ),
            ),
        ),
    )

    roles = DefaultTransitionRouter.assign_interstitial_roles(
        fade=FadeRoutingAssessment(
            rise_present=False,
            basin_count=0,
            geometry_count=2,
        ),
        interstitial=InterstitialRoutingAssessment(
            routing_handoffs=(
                outgoing_handoff,
                unrelated_handoff,
            ),
            supported_handoffs=(
                outgoing_handoff,
                unrelated_handoff,
            ),
        ),
        context=context,
    )

    # Temporal bracketing alone must not promote the unowned 1039s Fade edge.
    # With no proven two-sided edge topology, Router falls back only to
    # Interstitial-owned routing evidence.
    assert roles is not None
    assert roles.outgoing_handoff is outgoing_handoff
    assert roles.incoming_handoff is unrelated_handoff
    assert roles.outgoing_edge_fade is None
    assert roles.incoming_edge_fade is None



def test_interstitial_roles_use_unique_region_bracketing_reference() -> None:
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_evidence import TransitionEvidenceState
    from brynse.transition_router import (
        DefaultTransitionRouter,
        FadeRoutingAssessment,
        InterstitialRoutingAssessment,
        TransitionRoutingContext,
    )

    outgoing = _interstitial_hypothesis(
        basin_time=100.0,
        structural_time=100.2,
        basin_start=99.5,
        basin_recovery=100.5,
    )
    incoming = _interstitial_hypothesis(
        basin_time=108.0,
        structural_time=108.2,
        basin_start=107.5,
        basin_recovery=108.5,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=104.0,
        left=outgoing,
        right=incoming,
    )

    roles = DefaultTransitionRouter.assign_interstitial_roles(
        fade=FadeRoutingAssessment(
            rise_present=False,
            basin_count=0,
            geometry_count=0,
        ),
        interstitial=InterstitialRoutingAssessment(
            routing_handoffs=(
                outgoing,
                incoming,
                _interstitial_hypothesis(
                    basin_time=112.0,
                    structural_time=112.2,
                    basin_start=111.5,
                    basin_recovery=112.5,
                ),
            ),
            supported_handoffs=(
                outgoing,
                incoming,
            ),
        ),
        context=TransitionRoutingContext(
            reference_time_seconds=104.0,
            evidence=TransitionEvidenceState(),
            hypotheses=(outgoing, incoming),
            bounded_insert_regions=(region,),
        ),
    )

    assert roles is not None
    assert roles.outgoing_handoff is outgoing
    assert roles.incoming_handoff is incoming
    assert roles.inserted_region is region



def test_interstitial_roles_reject_region_after_reference() -> None:
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_evidence import TransitionEvidenceState
    from brynse.transition_router import (
        DefaultTransitionRouter,
        FadeRoutingAssessment,
        InterstitialRoutingAssessment,
        TransitionRoutingContext,
    )

    outgoing = _interstitial_hypothesis(
        basin_time=1031.8,
        structural_time=1032.0,
        basin_start=1030.6,
        basin_recovery=1034.5,
    )
    incoming = _interstitial_hypothesis(
        basin_time=1038.5,
        structural_time=1038.8,
        basin_start=1036.9,
        basin_recovery=1039.5,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=1030.8,
        left=outgoing,
        right=incoming,
    )

    roles = DefaultTransitionRouter.assign_interstitial_roles(
        fade=FadeRoutingAssessment(
            rise_present=False,
            basin_count=0,
            geometry_count=0,
        ),
        interstitial=InterstitialRoutingAssessment(
            routing_handoffs=(),
            supported_handoffs=(
                outgoing,
                incoming,
            ),
        ),
        context=TransitionRoutingContext(
            reference_time_seconds=1030.8,
            evidence=TransitionEvidenceState(),
            hypotheses=(outgoing, incoming),
            bounded_insert_regions=(region,),
        ),
    )

    assert roles is None



def test_interstitial_roles_use_mixed_completed_fade_and_handoff() -> None:
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource
    from brynse.transition_router import (
        DefaultTransitionRouter,
        FadeRoutingAssessment,
        InterstitialRoutingAssessment,
    )

    incoming_handoff = _interstitial_hypothesis(
        basin_time=123.2,
        structural_time=123.3,
        basin_start=123.0,
        basin_recovery=124.5,
    )
    later_handoff = _interstitial_hypothesis(
        basin_time=127.0,
        structural_time=127.2,
        basin_start=126.5,
        basin_recovery=127.5,
    )

    structural_witness = SimpleNamespace(
        time_seconds=108.5,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
        basin_geometry=None,
    )

    left_outgoing = SimpleNamespace(
        floor_time_seconds=105.7,
    )
    left_incoming = SimpleNamespace(
        start_time_seconds=105.7,
        established_time_seconds=110.8,
    )

    right_outgoing_1 = SimpleNamespace(
        floor_time_seconds=124.2,
    )
    right_incoming_1 = SimpleNamespace(
        start_time_seconds=123.5,
        established_time_seconds=124.0,
    )

    right_outgoing_2 = SimpleNamespace(
        floor_time_seconds=145.5,
    )
    right_incoming_2 = SimpleNamespace(
        start_time_seconds=124.2,
        established_time_seconds=125.0,
    )

    context = SimpleNamespace(
        reference_time_seconds=121.5,
        hypotheses=(
            structural_witness,
            incoming_handoff,
            later_handoff,
        ),
        bounded_insert_regions=(),
        fade=SimpleNamespace(
            geometries=(
                SimpleNamespace(
                    outgoing=left_outgoing,
                    incoming=left_incoming,
                ),
                SimpleNamespace(
                    outgoing=right_outgoing_1,
                    incoming=right_incoming_1,
                ),
                SimpleNamespace(
                    outgoing=right_outgoing_2,
                    incoming=right_incoming_2,
                ),
            ),
        ),
    )

    roles = DefaultTransitionRouter.assign_interstitial_roles(
        fade=FadeRoutingAssessment(
            rise_present=False,
            basin_count=0,
            geometry_count=3,
        ),
        interstitial=InterstitialRoutingAssessment(
            routing_handoffs=(
                incoming_handoff,
                later_handoff,
            ),
            supported_handoffs=(
                incoming_handoff,
                later_handoff,
            ),
        ),
        context=context,
    )

    assert roles is not None
    assert roles.outgoing_edge_fade is left_outgoing
    assert roles.incoming_edge_fade is None
    assert roles.outgoing_handoff is None
    assert roles.incoming_handoff is incoming_handoff



def test_interstitial_mixed_topology_yields_to_observed_region() -> None:
    from types import SimpleNamespace

    from brynse.boundaries import BoundaryProposalSource
    from brynse.bounded_insert import BoundedInsertRegionEvidence
    from brynse.transition_router import (
        DefaultTransitionRouter,
        FadeRoutingAssessment,
        InterstitialRoutingAssessment,
    )

    left = _interstitial_hypothesis(
        basin_time=100.0,
        structural_time=100.1,
        basin_start=99.0,
        basin_recovery=101.0,
    )
    right = _interstitial_hypothesis(
        basin_time=105.0,
        structural_time=105.1,
        basin_start=104.5,
        basin_recovery=105.5,
    )

    structural_witness = SimpleNamespace(
        time_seconds=82.0,
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
        basin_geometry=None,
    )

    old_outgoing = SimpleNamespace(
        floor_time_seconds=80.0,
    )
    old_incoming = SimpleNamespace(
        start_time_seconds=80.0,
        established_time_seconds=83.0,
    )

    upcoming_outgoing = SimpleNamespace(
        floor_time_seconds=100.5,
    )
    upcoming_incoming = SimpleNamespace(
        start_time_seconds=100.0,
        established_time_seconds=100.5,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=100.0,
        left=left,
        right=right,
    )

    context = SimpleNamespace(
        reference_time_seconds=100.0,
        hypotheses=(
            structural_witness,
            left,
            right,
        ),
        bounded_insert_regions=(region,),
        fade=SimpleNamespace(
            geometries=(
                SimpleNamespace(
                    outgoing=old_outgoing,
                    incoming=old_incoming,
                ),
                SimpleNamespace(
                    outgoing=upcoming_outgoing,
                    incoming=upcoming_incoming,
                ),
            ),
        ),
    )

    roles = DefaultTransitionRouter.assign_interstitial_roles(
        fade=FadeRoutingAssessment(
            rise_present=False,
            basin_count=0,
            geometry_count=2,
        ),
        interstitial=InterstitialRoutingAssessment(
            routing_handoffs=(
                left,
                right,
            ),
            supported_handoffs=(
                left,
                right,
            ),
        ),
        context=context,
    )

    assert roles is not None
    assert roles.outgoing_edge_fade is None



def test_transition_routing_context_preserves_acoustic_island_evidence() -> None:
    from types import SimpleNamespace

    from brynse.acoustic_island import AcousticIslandEvidence
    from brynse.transition_evidence import TransitionEvidenceState
    from brynse.transition_router import TransitionRoutingContext

    outgoing = SimpleNamespace(time_seconds=98.0)
    incoming = SimpleNamespace(time_seconds=108.0)
    region = SimpleNamespace(
        left=outgoing,
        right=incoming,
    )

    island = AcousticIslandEvidence(
        incoming_handoff=incoming,
        observed_region=region,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=103.0,
        evidence=TransitionEvidenceState(),
        acoustic_islands=(island,),
    )

    assert context.acoustic_islands == (island,)
    assert context.acoustic_islands[0] is island
    assert not hasattr(context, "incoming_start_seconds")
    assert not hasattr(context, "outgoing_end_seconds")


def test_region_backed_acoustic_island_precedes_direct_fade_to_floor() -> None:
    from types import SimpleNamespace

    from brynse.acoustic_island import AcousticIslandEvidence
    from brynse.fade_to_floor import FadeToFloorEvidence
    from brynse.transition_evidence import TransitionEvidenceState
    from brynse.transition_router import (
        DefaultTransitionRouter,
        TransitionKind,
        TransitionRoutingContext,
    )

    left = SimpleNamespace(time_seconds=100.0)
    right = SimpleNamespace(time_seconds=120.0)

    region = SimpleNamespace(
        left=left,
        right=right,
    )

    island = AcousticIslandEvidence(
        incoming_handoff=right,
        observed_region=region,
    )

    fade_to_floor = FadeToFloorEvidence(
        outgoing=SimpleNamespace(
            floor_time_seconds=90.0,
        ),
        incoming_handoff=SimpleNamespace(
            time_seconds=92.0,
        ),
    )

    route = DefaultTransitionRouter().route(
        context=TransitionRoutingContext(
            reference_time_seconds=105.0,
            evidence=TransitionEvidenceState(),
            acoustic_islands=(island,),
            fade_to_floor=(fade_to_floor,),
        )
    )

    assert route.kind is TransitionKind.INTERSTITIAL
    assert route.semantic_roles is not None
    assert route.semantic_roles.acoustic_island is island
    assert route.semantic_roles.inserted_region is region
    assert route.semantic_roles.outgoing_handoff is left
    assert route.semantic_roles.incoming_handoff is right


def test_acoustic_island_owns_richer_topology_before_fade() -> None:
    from types import SimpleNamespace

    from brynse.acoustic_island import AcousticIslandEvidence
    from brynse.transition_evidence import (
        CrossfadeBasinEvidence,
        CrossfadeRiseEvidence,
        FadeRoutingEvidence,
        TransitionEvidenceState,
    )
    from brynse.transition_router import (
        DefaultTransitionRouter,
        TransitionKind,
        TransitionRoutingContext,
    )

    outgoing = SimpleNamespace(time_seconds=98.0)
    incoming = SimpleNamespace(time_seconds=108.0)
    region = SimpleNamespace(
        left=outgoing,
        right=incoming,
    )

    island = AcousticIslandEvidence(
        incoming_handoff=incoming,
        observed_region=region,
    )

    fade = FadeRoutingEvidence(
        rise=CrossfadeRiseEvidence(
            time_seconds=104.0,
            jump_ratio=3.5,
            persistence_ratio=4.0,
            after_global_ratio=3.0,
        ),
        basins=(
            CrossfadeBasinEvidence(
                center_time_seconds=102.0,
                lead_seconds=0.8,
                width_seconds=0.7,
                prominence=1.5,
                quietness=1.8,
            ),
        ),
    )

    route = DefaultTransitionRouter().route(
        context=TransitionRoutingContext(
            reference_time_seconds=103.0,
            evidence=TransitionEvidenceState(),
            fade=fade,
            acoustic_islands=(island,),
        )
    )

    assert route.kind is TransitionKind.INTERSTITIAL
    assert route.fade_assessment is not None
    assert route.fade_assessment.ownership_candidate is True

    assert route.semantic_roles is not None
    assert route.semantic_roles.acoustic_island is island
    assert route.semantic_roles.outgoing_handoff is outgoing
    assert route.semantic_roles.incoming_handoff is incoming
    assert route.semantic_roles.inserted_region is region

    assert not hasattr(route.semantic_roles, "incoming_start_seconds")
    assert not hasattr(route.semantic_roles, "outgoing_end_seconds")
