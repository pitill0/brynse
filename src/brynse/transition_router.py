"""Routing contract for selecting the specialist that owns a transition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from brynse.boundaries import BoundaryHypothesis, BoundaryProposalSource
from brynse.bounded_insert import BoundedInsertRegionEvidence
from brynse.transition_evidence import (
    FadeRoutingEvidence,
    TransitionEvidenceState,
)


class TransitionKind(StrEnum):
    """Acoustic transition classes understood by boundary specialists."""

    HARD_CUT = "hard_cut"
    FADE = "fade"
    INTERSTITIAL = "interstitial"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class TransitionRoutingContext:
    """Descriptive context available while choosing a transition specialist."""

    reference_time_seconds: float
    evidence: TransitionEvidenceState
    hypotheses: tuple[BoundaryHypothesis, ...] = ()
    fade: FadeRoutingEvidence = field(default_factory=FadeRoutingEvidence)
    bounded_insert_regions: tuple[BoundedInsertRegionEvidence, ...] = ()

    def __post_init__(self) -> None:
        if self.reference_time_seconds < 0:
            raise ValueError("reference_time_seconds must be non-negative")


class TransitionRouter(Protocol):
    """Classify transition context without choosing temporal boundaries."""

    def classify(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> TransitionKind:
        """Return which specialist owns the transition."""
        ...


@dataclass(frozen=True)
class BoundedInsertRoutingAssessment:
    """Shadow routing assessment for bounded inserted material."""

    candidate: bool
    related_region_count: int
    unique_region: bool
    left_is_context_structural_peak: bool = False
    right_is_context_structural_peak: bool = False
    reference_precedes_left_support_inside_basin: bool = False
    structural_only_interior_count: int = 0


class DefaultTransitionRouter:
    """Conservatively route transitions from descriptive acoustic evidence."""

    def assess_bounded_insert(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> BoundedInsertRoutingAssessment:
        """Assess bounded-insert ownership without changing routed kind."""
        regions = context.bounded_insert_regions
        region_count = len(regions)

        if region_count != 1:
            return BoundedInsertRoutingAssessment(
                candidate=False,
                related_region_count=region_count,
                unique_region=False,
            )

        region = regions[0]

        structural_strengths = tuple(
            proposal.strength
            for hypothesis in context.hypotheses
            for proposal in hypothesis.proposals
            if (
                proposal.source is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )

        context_peak = (
            max(structural_strengths)
            if structural_strengths
            else None
        )

        left_strengths = tuple(
            proposal.strength
            for proposal in region.left_edge.proposals
            if (
                proposal.source is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )
        right_strengths = tuple(
            proposal.strength
            for proposal in region.right_edge.proposals
            if (
                proposal.source is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )

        left_peak = (
            context_peak is not None
            and bool(left_strengths)
            and max(left_strengths) == context_peak
        )
        right_peak = (
            context_peak is not None
            and bool(right_strengths)
            and max(right_strengths) == context_peak
        )

        reference_precedes_left = (
            region.reference_time_seconds
            < region.left.time_seconds
            and region.reference_inside_left_basin
        )

        structurally_non_dominant_region = (
            not left_peak
            and not right_peak
        )

        basin_entry_with_internal_structure = (
            reference_precedes_left
            and region.structural_only_interior_count > 0
            and not right_peak
        )

        return BoundedInsertRoutingAssessment(
            candidate=(
                structurally_non_dominant_region
                or basin_entry_with_internal_structure
            ),
            related_region_count=region_count,
            unique_region=True,
            left_is_context_structural_peak=left_peak,
            right_is_context_structural_peak=right_peak,
            reference_precedes_left_support_inside_basin=(
                reference_precedes_left
            ),
            structural_only_interior_count=(
                region.structural_only_interior_count
            ),
        )

    def classify(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> TransitionKind:
        fade_like = context.fade.rise is not None and bool(context.fade.basins)
        interstitial_like = self._has_interstitial_pattern(context.hypotheses)

        if fade_like and interstitial_like:
            return TransitionKind.AMBIGUOUS

        if fade_like:
            return TransitionKind.FADE

        if interstitial_like:
            return TransitionKind.INTERSTITIAL

        if (
            context.fade.rise is None
            and self._has_complete_local_discontinuity(context.evidence)
        ):
            return TransitionKind.HARD_CUT

        return TransitionKind.AMBIGUOUS

    @staticmethod
    def _has_complete_local_discontinuity(
        evidence: TransitionEvidenceState,
    ) -> bool:
        return all(
            value is not None
            for value in (
                evidence.local_discontinuity_250ms,
                evidence.local_discontinuity_500ms,
                evidence.local_discontinuity_1s,
                evidence.local_discontinuity_2s,
            )
        )

    @staticmethod
    def _has_interstitial_pattern(
        hypotheses: tuple[BoundaryHypothesis, ...],
    ) -> bool:
        matching = 0

        for hypothesis in hypotheses:
            sources = {proposal.source for proposal in hypothesis.proposals}

            if sources != {
                BoundaryProposalSource.BASIN,
                BoundaryProposalSource.STRUCTURAL,
            }:
                continue

            if hypothesis.basin_geometry is None:
                continue

            matching += 1

            if matching >= 2:
                return True

        return False
