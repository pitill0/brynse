"""Routing contract for selecting the specialist that owns a transition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from brynse.boundaries import BoundaryHypothesis, BoundaryProposalSource
from brynse.bounded_insert import (
    BoundedInsertAsymmetricRegionEvidence,
    BoundedInsertRegionEvidence,
)
from brynse.transition_evidence import (
    FadeRoutingEvidence,
    TransitionEvidenceState,
)


class TransitionKind(StrEnum):
    """Acoustic transition classes understood by boundary specialists."""

    HARD_CUT = "hard_cut"
    FADE = "fade"
    INTERSTITIAL = "interstitial"
    BOUNDED_INSERT = "bounded_insert"
    AMBIGUOUS = "ambiguous"


class BoundedInsertOwnership(StrEnum):
    """Evidence family that gave bounded-insert routing ownership."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


@dataclass(frozen=True)
class TransitionRoutingContext:
    """Descriptive context available while choosing a transition specialist."""

    reference_time_seconds: float
    evidence: TransitionEvidenceState
    hypotheses: tuple[BoundaryHypothesis, ...] = ()
    fade: FadeRoutingEvidence = field(default_factory=FadeRoutingEvidence)
    bounded_insert_regions: tuple[BoundedInsertRegionEvidence, ...] = ()
    bounded_insert_asymmetric_regions: tuple[
        BoundedInsertAsymmetricRegionEvidence, ...
    ] = ()

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


@dataclass(frozen=True)
class BoundedInsertAsymmetricRoutingAssessment:
    """Routing ownership assessment for asymmetric bounded inserted material."""

    candidate: bool
    related_region_count: int
    unique_region: bool
    primary_is_context_structural_peak: bool = False


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

        reference_between_supports = (
            region.left.time_seconds
            <= region.reference_time_seconds
            <= region.right.time_seconds
        )

        structurally_non_dominant_region = (
            reference_between_supports
            and not left_peak
            and not right_peak
        )

        basin_entry_with_internal_structure = (
            reference_precedes_left
            and region.structural_only_interior_count > 0
            and not right_peak
        )

        interior_source_sets = tuple(
            {
                proposal.source
                for proposal in hypothesis.proposals
            }
            for hypothesis in getattr(
                region,
                "interior_hypotheses",
                (),
            )
        )

        has_structural_only_interior = any(
            sources == {BoundaryProposalSource.STRUCTURAL}
            for sources in interior_source_sets
        )
        has_basin_only_interior = any(
            sources == {BoundaryProposalSource.BASIN}
            for sources in interior_source_sets
        )

        mixed_interior_families = (
            has_structural_only_interior
            and has_basin_only_interior
        )

        single_structural_only_interior = (
            interior_source_sets
            == ({BoundaryProposalSource.STRUCTURAL},)
        )

        terminal_single_insert = (
            reference_between_supports
            and not left_peak
            and right_peak
            and single_structural_only_interior
        )

        return BoundedInsertRoutingAssessment(
            candidate=(
                not mixed_interior_families
                and (
                    structurally_non_dominant_region
                    or basin_entry_with_internal_structure
                    or terminal_single_insert
                )
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

    def assess_asymmetric_bounded_insert(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> BoundedInsertAsymmetricRoutingAssessment:
        """Assess asymmetric bounded-insert ownership without choosing times."""
        regions = context.bounded_insert_asymmetric_regions
        region_count = len(regions)

        if region_count != 1:
            return BoundedInsertAsymmetricRoutingAssessment(
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

        primary_strengths = tuple(
            proposal.strength
            for proposal in region.primary_edge.proposals
            if (
                proposal.source is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )

        primary_peak = (
            context_peak is not None
            and bool(primary_strengths)
            and max(primary_strengths) == context_peak
        )

        return BoundedInsertAsymmetricRoutingAssessment(
            candidate=primary_peak,
            related_region_count=region_count,
            unique_region=True,
            primary_is_context_structural_peak=primary_peak,
        )

    def bounded_insert_ownership(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> BoundedInsertOwnership | None:
        """Return which bounded-insert evidence family owns the transition."""
        symmetric = self.assess_bounded_insert(
            context=context,
        )

        if symmetric.candidate:
            return BoundedInsertOwnership.SYMMETRIC

        asymmetric = self.assess_asymmetric_bounded_insert(
            context=context,
        )

        if asymmetric.candidate:
            return BoundedInsertOwnership.ASYMMETRIC

        return None

    def classify(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> TransitionKind:
        bounded_insert_ownership = self.bounded_insert_ownership(
            context=context,
        )

        if bounded_insert_ownership is not None:
            return TransitionKind.BOUNDED_INSERT

        fade_like = context.fade.rise is not None and bool(context.fade.basins)
        interstitial_like = self._has_interstitial_pattern(context.hypotheses)

        if fade_like and interstitial_like:
            if self._has_interstitial_absorbed_right_edge_fade(
                context=context,
            ):
                return TransitionKind.INTERSTITIAL

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

    def _has_interstitial_absorbed_right_edge_fade(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> bool:
        """Return whether fade evidence belongs to the interstitial right edge.

        A narrow asymmetric-looking symmetric region can carry one structural
        event internally while its dominant left edge and weaker right edge
        still describe one interstitial handoff.  When an observed outgoing
        fade rises inside the right edge's own basin, that fade describes the
        same terminal edge instead of an independent fade phenomenon.
        """
        assessment = self.assess_bounded_insert(
            context=context,
        )

        if not (
            assessment.unique_region
            and not assessment.candidate
            and assessment.left_is_context_structural_peak
            and not assessment.right_is_context_structural_peak
            and assessment.structural_only_interior_count == 1
        ):
            return False

        if len(context.bounded_insert_regions) != 1:
            return False

        region = context.bounded_insert_regions[0]
        right_geometry = region.right_geometry

        if right_geometry is None:
            return False

        return any(
            fade.rise_time_seconds is not None
            and (
                right_geometry.start_seconds
                <= fade.rise_time_seconds
                <= right_geometry.recovery_seconds
            )
            for fade in context.fade.outgoing_fades
        )

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
