"""Shadow confidence evaluation for reconciled boundary hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryHypothesis,
    BoundaryProposalSource,
)


class BoundaryConfidenceRoute(StrEnum):
    """Named experimental route that can support a shadow confidence decision."""

    BOTH_DISTRIBUTION = "both_distribution"


@dataclass(frozen=True)
class ShadowBoundaryAssessment:
    """Non-operative confidence assessment for one boundary hypothesis."""

    hypothesis: BoundaryHypothesis
    spectral_between: float | None
    candidate_confidence: BoundaryConfidence
    route: BoundaryConfidenceRoute | None = None


class ShadowBoundaryConfidenceEvaluator:
    """Evaluate candidate confidence without changing operational split behavior."""

    def __init__(
        self,
        *,
        max_basin_depth: float = 0.06,
        min_structural_novelty: float = 0.35,
        min_spectral_between: float = 0.15,
    ) -> None:
        if max_basin_depth < 0:
            raise ValueError("max_basin_depth must be non-negative")
        if min_structural_novelty < 0:
            raise ValueError("min_structural_novelty must be non-negative")
        if min_spectral_between < 0:
            raise ValueError("min_spectral_between must be non-negative")

        self._max_basin_depth = max_basin_depth
        self._min_structural_novelty = min_structural_novelty
        self._min_spectral_between = min_spectral_between

    def assess(
        self,
        *,
        hypothesis: BoundaryHypothesis,
        spectral_between: float | None,
    ) -> ShadowBoundaryAssessment:
        """Return a shadow-only confidence assessment for one hypothesis."""
        if spectral_between is not None and spectral_between < 0:
            raise ValueError("spectral_between must be non-negative")

        route = self._matching_route(
            hypothesis=hypothesis,
            spectral_between=spectral_between,
        )
        confidence = BoundaryConfidence.HIGH if route is not None else BoundaryConfidence.UNRESOLVED

        return ShadowBoundaryAssessment(
            hypothesis=hypothesis,
            spectral_between=spectral_between,
            candidate_confidence=confidence,
            route=route,
        )

    def _matching_route(
        self,
        *,
        hypothesis: BoundaryHypothesis,
        spectral_between: float | None,
    ) -> BoundaryConfidenceRoute | None:
        sources = {proposal.source for proposal in hypothesis.proposals}
        if sources != {
            BoundaryProposalSource.BASIN,
            BoundaryProposalSource.STRUCTURAL,
        }:
            return None

        basin_depths = [
            evidence.basin_depth
            for evidence in hypothesis.evidence
            if evidence.basin_depth is not None
        ]
        structural_novelties = [
            evidence.structural_novelty
            for evidence in hypothesis.evidence
            if evidence.structural_novelty is not None
        ]

        if not basin_depths or not structural_novelties or spectral_between is None:
            return None

        basin_depth = min(basin_depths)
        structural_novelty = max(structural_novelties)

        if basin_depth > self._max_basin_depth:
            return None
        if structural_novelty < self._min_structural_novelty:
            return None
        if spectral_between < self._min_spectral_between:
            return None

        return BoundaryConfidenceRoute.BOTH_DISTRIBUTION
