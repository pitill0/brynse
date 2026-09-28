from __future__ import annotations

import pytest

from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
)
from brynse.confidence import (
    BoundaryConfidenceRoute,
    ShadowBoundaryConfidenceEvaluator,
)


def _proposal(
    time_seconds: float,
    *,
    source: BoundaryProposalSource,
    strength: float | None = None,
) -> BoundaryProposal:
    return BoundaryProposal(
        time_seconds=time_seconds,
        source=source,
        strength=strength,
    )


def _both_hypothesis(
    *,
    basin_depth: float = 0.04,
    structural_novelty: float = 0.50,
) -> BoundaryHypothesis:
    basin = _proposal(100.0, source=BoundaryProposalSource.BASIN)
    structural = _proposal(
        100.5,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=structural_novelty,
    )
    return BoundaryHypothesis(
        time_seconds=100.25,
        proposals=(basin, structural),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_depth=basin_depth,
            ),
            BoundaryEvidence(
                proposal=structural,
                structural_novelty=structural_novelty,
            ),
        ),
    )


def test_shadow_confidence_marks_supported_both_route_high() -> None:
    hypothesis = _both_hypothesis()

    assessment = ShadowBoundaryConfidenceEvaluator().assess(
        hypothesis=hypothesis,
        spectral_between=0.30,
    )

    assert assessment.candidate_confidence is BoundaryConfidence.HIGH
    assert assessment.route is BoundaryConfidenceRoute.BOTH_DISTRIBUTION
    assert assessment.hypothesis is hypothesis
    assert hypothesis.confidence is BoundaryConfidence.UNRESOLVED


@pytest.mark.parametrize(
    ("basin_depth", "novelty", "spectral_between"),
    [
        (0.07, 0.50, 0.30),
        (0.04, 0.34, 0.30),
        (0.04, 0.50, 0.14),
    ],
)
def test_shadow_confidence_leaves_unsupported_both_route_unresolved(
    basin_depth: float,
    novelty: float,
    spectral_between: float,
) -> None:
    assessment = ShadowBoundaryConfidenceEvaluator().assess(
        hypothesis=_both_hypothesis(
            basin_depth=basin_depth,
            structural_novelty=novelty,
        ),
        spectral_between=spectral_between,
    )

    assert assessment.candidate_confidence is BoundaryConfidence.UNRESOLVED
    assert assessment.route is None


def test_shadow_confidence_requires_both_audio_sources() -> None:
    basin = _proposal(100.0, source=BoundaryProposalSource.BASIN)
    hypothesis = BoundaryHypothesis(
        time_seconds=100.0,
        proposals=(basin,),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_depth=0.01,
            ),
        ),
    )

    assessment = ShadowBoundaryConfidenceEvaluator().assess(
        hypothesis=hypothesis,
        spectral_between=1.0,
    )

    assert assessment.candidate_confidence is BoundaryConfidence.UNRESOLVED
    assert assessment.route is None


def test_shadow_confidence_requires_descriptive_evidence() -> None:
    basin = _proposal(100.0, source=BoundaryProposalSource.BASIN)
    structural = _proposal(
        100.5,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.9,
    )
    hypothesis = BoundaryHypothesis(
        time_seconds=100.25,
        proposals=(basin, structural),
    )

    assessment = ShadowBoundaryConfidenceEvaluator().assess(
        hypothesis=hypothesis,
        spectral_between=1.0,
    )

    assert assessment.candidate_confidence is BoundaryConfidence.UNRESOLVED
    assert assessment.route is None


def test_shadow_confidence_treats_missing_spectral_context_as_unresolved() -> None:
    assessment = ShadowBoundaryConfidenceEvaluator().assess(
        hypothesis=_both_hypothesis(),
        spectral_between=None,
    )

    assert assessment.candidate_confidence is BoundaryConfidence.UNRESOLVED
    assert assessment.route is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_basin_depth": -0.01}, "max_basin_depth"),
        ({"min_structural_novelty": -0.01}, "min_structural_novelty"),
        ({"min_spectral_between": -0.01}, "min_spectral_between"),
    ],
)
def test_shadow_confidence_rejects_invalid_configuration(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ShadowBoundaryConfidenceEvaluator(**kwargs)


def test_shadow_confidence_rejects_negative_spectral_between() -> None:
    with pytest.raises(ValueError, match="spectral_between"):
        ShadowBoundaryConfidenceEvaluator().assess(
            hypothesis=_both_hypothesis(),
            spectral_between=-0.01,
        )
