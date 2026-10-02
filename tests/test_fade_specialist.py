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
        floor_rms=1000.0 / ratio,
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
            reference_time_seconds=103.0,
            outgoing_fades=(),
            hypotheses=(hypothesis,),
        )
        is None
    )

    assert (
        FadeSpecialist.select_outgoing(
            reference_time_seconds=103.0,
            outgoing_fades=(fade,),
            hypotheses=(),
        )
        is None
    )


def test_outgoing_selection_rejects_negative_reference_time() -> None:
    with pytest.raises(ValueError, match="reference_time_seconds"):
        FadeSpecialist.select_outgoing(
            reference_time_seconds=-0.01,
            outgoing_fades=(),
            hypotheses=(),
        )


def test_outgoing_selection_ignores_fade_starting_after_reference() -> None:
    outgoing = _fade(
        start=100.0,
        floor=104.0,
        ratio=5.0,
    )
    later_stronger = _fade(
        start=106.0,
        floor=110.0,
        ratio=30.0,
    )

    selection = FadeSpecialist.select_outgoing(
        reference_time_seconds=105.0,
        outgoing_fades=(outgoing, later_stronger),
        hypotheses=(
            _hypothesis(time=104.1),
            _hypothesis(time=110.0),
        ),
    )

    assert selection is not None
    assert selection.fade is outgoing


def test_outgoing_selection_prefers_stronger_drawdown_over_nearer_support() -> None:
    weak_nearby = _fade(
        start=100.0,
        floor=104.0,
        ratio=2.0,
        consistency=0.95,
    )
    strong_supported = _fade(
        start=101.0,
        floor=107.0,
        ratio=12.0,
        consistency=0.9,
    )

    selection = FadeSpecialist.select_outgoing(
        reference_time_seconds=103.0,
        outgoing_fades=(weak_nearby, strong_supported),
        hypotheses=(
            _hypothesis(time=104.01),
            _hypothesis(time=107.4),
        ),
    )

    assert selection is not None
    assert selection.fade is strong_supported
    assert selection.hypothesis.time_seconds == pytest.approx(107.4)
    assert selection.support_distance_seconds == pytest.approx(0.4)


def test_outgoing_selection_uses_consistency_to_break_equal_drawdown() -> None:
    inconsistent = _fade(
        start=100.0,
        floor=104.0,
        ratio=10.0,
        consistency=0.7,
    )
    consistent = _fade(
        start=100.5,
        floor=106.0,
        ratio=10.0,
        consistency=0.95,
    )

    selection = FadeSpecialist.select_outgoing(
        reference_time_seconds=103.0,
        outgoing_fades=(inconsistent, consistent),
        hypotheses=(
            _hypothesis(time=104.0),
            _hypothesis(time=106.2),
        ),
    )

    assert selection is not None
    assert selection.fade is consistent


def test_outgoing_selection_uses_boundary_distance_after_fade_strength() -> None:
    first = _fade(
        start=100.0,
        floor=104.0,
        ratio=10.0,
        consistency=0.9,
    )
    second = _fade(
        start=100.5,
        floor=106.0,
        ratio=10.0,
        consistency=0.9,
    )

    selection = FadeSpecialist.select_outgoing(
        reference_time_seconds=103.0,
        outgoing_fades=(first, second),
        hypotheses=(
            _hypothesis(time=104.4),
            _hypothesis(time=106.1),
        ),
    )

    assert selection is not None
    assert selection.fade is second
    assert selection.support_distance_seconds == pytest.approx(0.1)


def test_outgoing_selection_uses_confidence_only_after_equal_distance() -> None:
    fade = _fade(
        start=100.0,
        floor=105.0,
        ratio=10.0,
    )

    unresolved = _hypothesis(
        time=104.8,
        confidence=BoundaryConfidence.UNRESOLVED,
    )
    high = _hypothesis(
        time=105.2,
        confidence=BoundaryConfidence.HIGH,
    )

    selection = FadeSpecialist.select_outgoing(
        reference_time_seconds=103.0,
        outgoing_fades=(fade,),
        hypotheses=(unresolved, high),
    )

    assert selection is not None
    assert selection.fade is fade
    assert selection.hypothesis is high
