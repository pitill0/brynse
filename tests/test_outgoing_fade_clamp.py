from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
)
from brynse.fade_specialist import OutgoingFadeSelection
from brynse.models import TemporalSplitDecision, TemporalSplitKind
from brynse.orchestrator import _apply_outgoing_fade_clamp
from brynse.transition_evidence import OutgoingFadeEvidence


def _selection(*, floor: float) -> OutgoingFadeSelection:
    fade = OutgoingFadeEvidence(
        start_time_seconds=floor - 5.0,
        floor_time_seconds=floor,
        rise_time_seconds=None,
        start_rms=1000.0,
        floor_rms=100.0,
        descent_ratio=10.0,
        descent_consistency=0.9,
    )
    hypothesis = BoundaryHypothesis(
        time_seconds=floor,
        proposals=(
            BoundaryProposal(
                time_seconds=floor,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
        evidence=(),
        confidence=BoundaryConfidence.UNRESOLVED,
    )
    return OutgoingFadeSelection(
        fade=fade,
        hypothesis=hypothesis,
        support_distance_seconds=0.0,
    )


def test_outgoing_clamp_preserves_decision_without_selection() -> None:
    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        incoming_start_seconds=110.0,
        outgoing_end_seconds=105.0,
    )

    assert (
        _apply_outgoing_fade_clamp(
            decision=decision,
            selection=None,
        )
        is decision
    )


def test_outgoing_clamp_does_not_extend_existing_outgoing_edge() -> None:
    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        incoming_start_seconds=110.0,
        outgoing_end_seconds=105.0,
    )

    assert (
        _apply_outgoing_fade_clamp(
            decision=decision,
            selection=_selection(floor=106.0),
        )
        is decision
    )


def test_outgoing_clamp_shortens_exclusion_without_moving_incoming() -> None:
    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        incoming_start_seconds=110.0,
        outgoing_end_seconds=105.0,
    )

    result = _apply_outgoing_fade_clamp(
        decision=decision,
        selection=_selection(floor=102.0),
    )

    assert result.kind is TemporalSplitKind.EXCLUSION
    assert result.incoming_start_seconds == 110.0
    assert result.outgoing_end_seconds == 102.0


def test_outgoing_clamp_turns_hard_cut_into_exclusion() -> None:
    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=105.0,
        outgoing_end_seconds=105.0,
    )

    result = _apply_outgoing_fade_clamp(
        decision=decision,
        selection=_selection(floor=102.0),
    )

    assert result.kind is TemporalSplitKind.EXCLUSION
    assert result.incoming_start_seconds == 105.0
    assert result.outgoing_end_seconds == 102.0


def test_outgoing_clamp_shortens_valid_crossfade() -> None:
    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.CROSSFADE,
        incoming_start_seconds=100.0,
        outgoing_end_seconds=110.0,
    )

    result = _apply_outgoing_fade_clamp(
        decision=decision,
        selection=_selection(floor=105.0),
    )

    assert result.kind is TemporalSplitKind.CROSSFADE
    assert result.incoming_start_seconds == 100.0
    assert result.outgoing_end_seconds == 105.0


def test_outgoing_clamp_preserves_crossfade_if_floor_crosses_incoming() -> None:
    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.CROSSFADE,
        incoming_start_seconds=100.0,
        outgoing_end_seconds=110.0,
    )

    result = _apply_outgoing_fade_clamp(
        decision=decision,
        selection=_selection(floor=99.0),
    )

    assert result is decision
