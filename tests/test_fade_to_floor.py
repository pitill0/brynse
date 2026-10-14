from types import SimpleNamespace

from brynse.acoustic_island import AcousticIslandEvidence
from brynse.boundaries import BoundaryProposalSource
from brynse.fade_specialist import FadeSpecialist
from brynse.fade_to_floor import (
    FadeToFloorDetector,
    FadeToFloorEvidence,
)
from brynse.models import TemporalSplitKind
from brynse.transition_evidence import (
    FadeRoutingEvidence,
    IncomingFadeEvidence,
    OutgoingFadeEvidence,
    TransitionEvidenceState,
)
from brynse.transition_router import (
    DefaultTransitionRouter,
    TransitionKind,
    TransitionRoutingContext,
    TransitionSemanticRoles,
)


def supported_handoff(
    *,
    time_seconds: float,
    basin_start: float,
    basin_minimum: float,
    basin_recovery: float,
):
    return SimpleNamespace(
        time_seconds=time_seconds,
        basin_geometry=SimpleNamespace(
            start_seconds=basin_start,
            minimum_seconds=basin_minimum,
            recovery_seconds=basin_recovery,
        ),
        proposals=(
            SimpleNamespace(
                source=BoundaryProposalSource.BASIN,
            ),
            SimpleNamespace(
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
    )


def outgoing_fade(
    *,
    start: float = 92.0,
    floor: float = 100.0,
    rise=None,
):
    return OutgoingFadeEvidence(
        start_time_seconds=start,
        floor_time_seconds=floor,
        rise_time_seconds=rise,
        start_rms=2000.0,
        floor_rms=0.0,
        descent_ratio=2000.0,
        descent_consistency=0.85,
    )


def test_detector_observes_unique_fade_floor_inside_supported_handoff_basin():
    outgoing = outgoing_fade()
    handoff = supported_handoff(
        time_seconds=100.2,
        basin_start=98.0,
        basin_minimum=100.0,
        basin_recovery=103.0,
    )

    result = FadeToFloorDetector().observe(
        reference_time_seconds=101.0,
        fade=FadeRoutingEvidence(
            outgoing_fades=(outgoing,),
        ),
        hypotheses=(handoff,),
    )

    assert len(result) == 1
    assert result[0].outgoing is outgoing
    assert result[0].incoming_handoff is handoff


def test_detector_rejects_ambiguous_supported_handoffs():
    outgoing = outgoing_fade()

    first = supported_handoff(
        time_seconds=100.2,
        basin_start=98.0,
        basin_minimum=100.0,
        basin_recovery=103.0,
    )
    second = supported_handoff(
        time_seconds=101.2,
        basin_start=99.0,
        basin_minimum=100.5,
        basin_recovery=104.0,
    )

    result = FadeToFloorDetector().observe(
        reference_time_seconds=102.0,
        fade=FadeRoutingEvidence(
            outgoing_fades=(outgoing,),
        ),
        hypotheses=(
            first,
            second,
        ),
    )

    assert result == ()


def test_detector_observes_outgoing_only_floor_before_later_activity():
    terminal = OutgoingFadeEvidence(
        start_time_seconds=92.0,
        floor_time_seconds=100.0,
        rise_time_seconds=None,
        start_rms=2000.0,
        floor_rms=0.5,
        descent_ratio=4000.0,
        descent_consistency=0.85,
    )
    later = OutgoingFadeEvidence(
        start_time_seconds=103.0,
        floor_time_seconds=104.0,
        rise_time_seconds=None,
        start_rms=1800.0,
        floor_rms=600.0,
        descent_ratio=3.0,
        descent_consistency=0.8,
    )

    result = FadeToFloorDetector().observe(
        reference_time_seconds=105.0,
        fade=FadeRoutingEvidence(
            outgoing_fades=(terminal, later),
        ),
        hypotheses=(),
    )

    assert len(result) == 1
    assert result[0].outgoing is terminal
    assert result[0].incoming_handoff is None


def test_detector_rejects_when_deepest_floor_is_after_reference():
    earlier = OutgoingFadeEvidence(
        start_time_seconds=92.0,
        floor_time_seconds=94.0,
        rise_time_seconds=None,
        start_rms=2000.0,
        floor_rms=500.0,
        descent_ratio=4.0,
        descent_consistency=0.8,
    )
    terminal = OutgoingFadeEvidence(
        start_time_seconds=99.0,
        floor_time_seconds=101.0,
        rise_time_seconds=None,
        start_rms=2000.0,
        floor_rms=0.0,
        descent_ratio=2000.0,
        descent_consistency=0.9,
    )

    result = FadeToFloorDetector().observe(
        reference_time_seconds=100.0,
        fade=FadeRoutingEvidence(
            outgoing_fades=(earlier, terminal),
        ),
        hypotheses=(),
    )

    assert result == ()


def test_detector_rejects_shallower_terminal_when_deepest_floor_has_rise():
    earlier_terminal = OutgoingFadeEvidence(
        start_time_seconds=80.0,
        floor_time_seconds=82.0,
        rise_time_seconds=None,
        start_rms=2000.0,
        floor_rms=200.0,
        descent_ratio=10.0,
        descent_consistency=0.8,
    )
    deepest_with_rise = OutgoingFadeEvidence(
        start_time_seconds=98.0,
        floor_time_seconds=100.0,
        rise_time_seconds=101.0,
        start_rms=2000.0,
        floor_rms=0.0,
        descent_ratio=2000.0,
        descent_consistency=0.9,
    )

    result = FadeToFloorDetector().observe(
        reference_time_seconds=102.0,
        fade=FadeRoutingEvidence(
            outgoing_fades=(
                earlier_terminal,
                deepest_with_rise,
            ),
        ),
        hypotheses=(),
    )

    assert result == ()


def test_detector_rejects_isolated_unowned_terminal_floor():
    terminal = OutgoingFadeEvidence(
        start_time_seconds=92.0,
        floor_time_seconds=100.0,
        rise_time_seconds=None,
        start_rms=2000.0,
        floor_rms=0.5,
        descent_ratio=4000.0,
        descent_consistency=0.85,
    )

    result = FadeToFloorDetector().observe(
        reference_time_seconds=105.0,
        fade=FadeRoutingEvidence(
            outgoing_fades=(terminal,),
        ),
        hypotheses=(),
    )

    assert result == ()


def test_interstitial_roles_accept_outgoing_only_fade_to_floor():
    outgoing = outgoing_fade()
    incoming = supported_handoff(
        time_seconds=110.0,
        basin_start=109.0,
        basin_minimum=109.5,
        basin_recovery=111.0,
    )
    evidence = FadeToFloorEvidence(
        outgoing=outgoing,
    )
    context = TransitionRoutingContext(
        reference_time_seconds=105.0,
        evidence=TransitionEvidenceState(),
        fade_to_floor=(evidence,),
    )
    roles = TransitionSemanticRoles(
        incoming_handoff=incoming,
    )

    enriched = (
        DefaultTransitionRouter
        .enrich_interstitial_roles_with_fade_to_floor(
            roles=roles,
            context=context,
        )
    )

    assert enriched is not None
    assert enriched.incoming_handoff is incoming
    assert enriched.outgoing_edge_fade is outgoing
    assert enriched.fade_to_floor is evidence


def test_fade_specialist_resolves_outgoing_only_fade_to_floor():
    outgoing = outgoing_fade()
    evidence = FadeToFloorEvidence(
        outgoing=outgoing,
    )

    observation = SimpleNamespace(
        selected=SimpleNamespace(
            time_seconds=105.0,
        ),
        routing=SimpleNamespace(
            reference_time_seconds=105.0,
            fade=FadeRoutingEvidence(),
            hypotheses=(),
        ),
        fade_assessment=None,
        semantic_roles=TransitionSemanticRoles(
            outgoing_edge_fade=outgoing,
            fade_to_floor=evidence,
        ),
    )

    result = FadeSpecialist._resolve_geometry(
        observation=observation,
    )

    assert result.temporal is not None
    assert result.temporal.kind is TemporalSplitKind.EXCLUSION
    assert result.temporal.outgoing_end_seconds == 100.0
    assert result.temporal.incoming_start_seconds == 105.0


def test_detector_rejects_outgoing_with_observed_later_rise():
    outgoing = outgoing_fade(rise=100.1)
    handoff = supported_handoff(
        time_seconds=100.2,
        basin_start=98.0,
        basin_minimum=100.0,
        basin_recovery=103.0,
    )

    result = FadeToFloorDetector().observe(
        reference_time_seconds=101.0,
        fade=FadeRoutingEvidence(
            outgoing_fades=(outgoing,),
        ),
        hypotheses=(handoff,),
    )

    assert result == ()


def test_router_prefers_fade_to_floor_over_unrelated_acoustic_island():
    outgoing = outgoing_fade()
    handoff = supported_handoff(
        time_seconds=100.2,
        basin_start=98.0,
        basin_minimum=100.0,
        basin_recovery=103.0,
    )
    fade_to_floor = FadeToFloorEvidence(
        outgoing=outgoing,
        incoming_handoff=handoff,
    )

    old_terminal = outgoing_fade(
        start=70.0,
        floor=71.0,
    )
    old_reset = IncomingFadeEvidence(
        start_time_seconds=75.0,
        established_time_seconds=75.2,
        start_rms=100.0,
        established_rms=1000.0,
        ascent_ratio=10.0,
        ascent_consistency=0.8,
        persistence_ratio=5.0,
    )
    island = AcousticIslandEvidence(
        incoming_handoff=handoff,
        terminal_decay=old_terminal,
        source_reset=old_reset,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=101.0,
        evidence=TransitionEvidenceState(),
        fade=FadeRoutingEvidence(),
        fade_to_floor=(fade_to_floor,),
        acoustic_islands=(island,),
    )

    route = DefaultTransitionRouter().route(
        context=context,
    )

    assert route.kind is TransitionKind.FADE
    assert route.semantic_roles is not None
    assert route.semantic_roles.fade_to_floor is fade_to_floor
    assert route.semantic_roles.outgoing_edge_fade is outgoing
    assert route.semantic_roles.incoming_handoff is handoff
    assert route.semantic_roles.acoustic_island is None


def test_fade_specialist_resolves_fade_to_floor_as_exclusion():
    outgoing = outgoing_fade()
    handoff = supported_handoff(
        time_seconds=100.2,
        basin_start=98.0,
        basin_minimum=100.0,
        basin_recovery=103.0,
    )
    evidence = FadeToFloorEvidence(
        outgoing=outgoing,
        incoming_handoff=handoff,
    )

    observation = SimpleNamespace(
        selected=SimpleNamespace(
            time_seconds=101.0,
        ),
        routing=SimpleNamespace(
            reference_time_seconds=101.0,
            fade=FadeRoutingEvidence(),
            hypotheses=(),
        ),
        fade_assessment=None,
        semantic_roles=TransitionSemanticRoles(
            incoming_handoff=handoff,
            outgoing_edge_fade=outgoing,
            fade_to_floor=evidence,
        ),
    )

    result = FadeSpecialist._resolve_geometry(
        observation=observation,
    )

    assert result.temporal is not None
    assert result.temporal.kind is TemporalSplitKind.EXCLUSION
    assert result.temporal.outgoing_end_seconds == 100.0
    assert result.temporal.incoming_start_seconds == 100.2
    assert result.diagnostics is not None
    assert result.diagnostics.outgoing_selection is not None
    assert result.diagnostics.incoming_selection is None
