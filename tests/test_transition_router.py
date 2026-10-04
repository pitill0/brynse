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
    TransitionEvidenceState,
)
from brynse.transition_router import (
    DefaultTransitionRouter,
    TransitionKind,
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

    result = DefaultTransitionRouter().classify(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=TransitionEvidenceState(),
            hypotheses=hypotheses,
        )
    )

    assert result is TransitionKind.INTERSTITIAL


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
