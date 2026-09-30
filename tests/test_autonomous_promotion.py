from __future__ import annotations

import pytest

from brynse.autonomous_promotion import AutonomousBoundaryPromoter
from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
)
from brynse.confidence import (
    BoundaryConfidenceRoute,
    ShadowBoundaryAssessment,
)
from brynse.models import (
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
)
from brynse.shadow import ShadowBoundaryAnalysis


class _Timeline:
    pass


class _SplitAligner:
    def __init__(self, split: SplitDecision) -> None:
        self.split = split
        self.calls: list[dict[str, object]] = []

    def align(self, **kwargs: object) -> SplitDecision:
        self.calls.append(kwargs)
        return self.split


def _hypothesis(*, time_seconds: float = 100.25) -> BoundaryHypothesis:
    basin = BoundaryProposal(
        time_seconds=time_seconds - 0.25,
        source=BoundaryProposalSource.BASIN,
    )
    structural = BoundaryProposal(
        time_seconds=time_seconds + 0.25,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.5,
    )
    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=(basin, structural),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_depth=0.04,
            ),
            BoundaryEvidence(
                proposal=structural,
                structural_novelty=0.5,
            ),
        ),
    )


def _analysis(
    *,
    confidence: BoundaryConfidence,
    time_seconds: float = 100.25,
) -> ShadowBoundaryAnalysis:
    hypothesis = _hypothesis(time_seconds=time_seconds)
    return ShadowBoundaryAnalysis(
        hypothesis=hypothesis,
        spectral=None,
        assessment=ShadowBoundaryAssessment(
            hypothesis=hypothesis,
            spectral_between=0.3,
            candidate_confidence=confidence,
            route=(
                BoundaryConfidenceRoute.BOTH_DISTRIBUTION
                if confidence is BoundaryConfidence.HIGH
                else None
            ),
        ),
    )


@pytest.mark.parametrize(
    "confidence",
    [
        confidence
        for confidence in BoundaryConfidence
        if confidence is not BoundaryConfidence.HIGH
    ],
)
def test_autonomous_promoter_ignores_non_high_observations(
    confidence: BoundaryConfidence,
) -> None:
    aligner = _SplitAligner(
        SplitDecision(
            kind=SplitKind.HARD_CUT,
            incoming_start=4200,
            outgoing_end=4200,
        )
    )
    promoter = AutonomousBoundaryPromoter(split_aligner=aligner)

    result = promoter.promote(
        analysis=_analysis(confidence=confidence),
        timeline=_Timeline(),
    )

    assert result is None
    assert aligner.calls == []


def test_autonomous_promoter_promotes_high_observation_to_hard_cut() -> None:
    split = SplitDecision(
        kind=SplitKind.HARD_CUT,
        incoming_start=4200,
        outgoing_end=4200,
    )
    aligner = _SplitAligner(split)
    promoter = AutonomousBoundaryPromoter(split_aligner=aligner)
    analysis = _analysis(
        confidence=BoundaryConfidence.HIGH,
        time_seconds=100.25,
    )
    timeline = _Timeline()

    result = promoter.promote(
        analysis=analysis,
        timeline=timeline,
    )

    assert result is not None
    assert result.candidate.time_seconds == 100.25
    assert result.candidate.source == "autonomous"
    assert result.candidate.reference_offset is None

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=100.25,
        outgoing_end_seconds=100.25,
    )
    assert result.split == split

    assert aligner.calls == [
        {
            "decision": result.temporal,
            "timeline": timeline,
        }
    ]


def test_autonomous_promoter_preserves_hypothesis_time_exactly() -> None:
    split = SplitDecision(
        kind=SplitKind.HARD_CUT,
        incoming_start=9876,
        outgoing_end=9876,
    )
    aligner = _SplitAligner(split)
    promoter = AutonomousBoundaryPromoter(split_aligner=aligner)

    result = promoter.promote(
        analysis=_analysis(
            confidence=BoundaryConfidence.HIGH,
            time_seconds=345.217,
        ),
        timeline=_Timeline(),
    )

    assert result is not None
    assert result.candidate.time_seconds == 345.217
    assert result.temporal.incoming_start_seconds == 345.217
    assert result.temporal.outgoing_end_seconds == 345.217


def test_autonomous_promoter_does_not_require_radio_semantics() -> None:
    split = SplitDecision(
        kind=SplitKind.HARD_CUT,
        incoming_start=1234,
        outgoing_end=1234,
    )

    result = AutonomousBoundaryPromoter(
        split_aligner=_SplitAligner(split),
    ).promote(
        analysis=_analysis(
            confidence=BoundaryConfidence.HIGH,
            time_seconds=42.0,
        ),
        timeline=_Timeline(),
    )

    assert result is not None
    assert result.candidate.source == "autonomous"
    assert result.split == split
