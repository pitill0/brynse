import pytest

from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
)
from brynse.fade_specialist import FadeSpecialist
from brynse.transition_evidence import OutgoingFadeEvidence


def _fade(
    *,
    start: float,
    floor: float,
    ratio: float = 5.0,
    consistency: float = 0.9,
) -> OutgoingFadeEvidence:
    return OutgoingFadeEvidence(
        start_time_seconds=start,
        floor_time_seconds=floor,
        rise_time_seconds=None,
        start_rms=1000.0,
        floor_rms=200.0,
        descent_ratio=ratio,
        descent_consistency=consistency,
    )


def _hypothesis(
    *,
    time: float,
    confidence: BoundaryConfidence = BoundaryConfidence.UNRESOLVED,
) -> BoundaryHypothesis:
    return BoundaryHypothesis(
        time_seconds=time,
        proposals=(
            BoundaryProposal(
                time_seconds=time,
                source=BoundaryProposalSource.STRUCTURAL,
            ),
        ),
        evidence=(),
        confidence=confidence,
    )


def test_outgoing_selection_requires_fade_and_boundary_evidence() -> None:
    fade = _fade(start=100.0, floor=105.0)
    hypothesis = _hypothesis(time=105.1)

    assert (
        FadeSpecialist.select_outgoing(
            outgoing_fades=(),
            hypotheses=(hypothesis,),
        )
        is None
    )

    assert (
        FadeSpecialist.select_outgoing(
            outgoing_fades=(fade,),
            hypotheses=(),
        )
        is None
    )


def test_outgoing_selection_prefers_floor_nearest_boundary_hypothesis() -> None:
    early = _fade(start=100.0, floor=104.0)
    supported = _fade(start=101.0, floor=107.0)

    distant = _hypothesis(time=104.8)
    nearby = _hypothesis(time=107.2)

    selection = FadeSpecialist.select_outgoing(
        outgoing_fades=(early, supported),
        hypotheses=(distant, nearby),
    )

    assert selection is not None
    assert selection.fade is supported
    assert selection.hypothesis is nearby
    assert selection.support_distance_seconds == pytest.approx(0.2)


def test_outgoing_selection_uses_confidence_only_as_distance_tiebreaker() -> None:
    fade = _fade(start=100.0, floor=105.0)

    unresolved = _hypothesis(
        time=104.8,
        confidence=BoundaryConfidence.UNRESOLVED,
    )
    high = _hypothesis(
        time=105.2,
        confidence=BoundaryConfidence.HIGH,
    )

    selection = FadeSpecialist.select_outgoing(
        outgoing_fades=(fade,),
        hypotheses=(unresolved, high),
    )

    assert selection is not None
    assert selection.fade is fade
    assert selection.hypothesis is high


def test_outgoing_selection_does_not_replace_distance_with_confidence() -> None:
    nearby = _fade(start=100.0, floor=105.0)
    distant = _fade(start=100.0, floor=108.0)

    low_nearby = _hypothesis(
        time=105.1,
        confidence=BoundaryConfidence.LOW,
    )
    high_distant = _hypothesis(
        time=108.5,
        confidence=BoundaryConfidence.HIGH,
    )

    selection = FadeSpecialist.select_outgoing(
        outgoing_fades=(nearby, distant),
        hypotheses=(low_nearby, high_distant),
    )

    assert selection is not None
    assert selection.fade is nearby
    assert selection.hypothesis is low_nearby
