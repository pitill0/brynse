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


def test_default_transition_router_keeps_conflicting_clear_patterns_ambiguous() -> None:
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

    assert result is TransitionKind.AMBIGUOUS


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


def test_hard_cut_route_carries_assessment_provenance() -> None:
    router = DefaultTransitionRouter()

    route = router.route(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=_complete_local_evidence(),
        )
    )

    assert route.kind is TransitionKind.HARD_CUT
    assert route.hard_cut is not None
    assert route.hard_cut.local_discontinuity_complete is True
    assert route.hard_cut.fade_rise_present is False
    assert route.hard_cut.fallback_eligible is True


def test_default_transition_router_routes_complete_local_boundary_as_hard_cut() -> None:
    from brynse.transition_router import DefaultTransitionRouter

    result = DefaultTransitionRouter().classify(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=_complete_local_evidence(),
        )
    )

    assert result is TransitionKind.HARD_CUT


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
        (112.0, TransitionKind.AMBIGUOUS),
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


def test_bounded_insert_shadow_accepts_basin_entry_with_internal_structure() -> None:
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

    assert result.candidate is True
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
