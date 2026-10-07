"""Routing contract for selecting the specialist that owns a transition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryHypothesis,
    BoundaryProposalSource,
)
from brynse.bounded_insert import (
    BoundedInsertAsymmetricRegionEvidence,
    BoundedInsertRegionEvidence,
)
from brynse.transition_evidence import (
    FadeRoutingEvidence,
    IncomingFadeEvidence,
    OutgoingFadeEvidence,
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


@dataclass(frozen=True)
class OutgoingFadeRoutingAssociation:
    """Descriptive association between outgoing fade and boundary evidence."""

    fade: OutgoingFadeEvidence
    hypothesis: BoundaryHypothesis
    support_distance_seconds: float


@dataclass(frozen=True)
class IncomingFadeRoutingAssociation:
    """Descriptive association between incoming fade and boundary evidence."""

    fade: IncomingFadeEvidence
    hypothesis: BoundaryHypothesis
    support_distance_seconds: float


@dataclass(frozen=True)
class FadeRoutingAssessment:
    """Router-owned interpretation of already observed fade evidence."""

    rise_present: bool
    basin_count: int
    geometry_count: int
    outgoing_association: OutgoingFadeRoutingAssociation | None = None
    incoming_association: IncomingFadeRoutingAssociation | None = None

    @property
    def legacy_candidate(self) -> bool:
        """Preserve the pre-Router-v2 fade routing condition."""
        return self.rise_present and self.basin_count > 0

    @property
    def has_two_sided_support(self) -> bool:
        """Return whether both fade sides have boundary support."""
        return (
            self.outgoing_association is not None
            and self.incoming_association is not None
        )


@dataclass(frozen=True)
class HardCutRoutingAssessment:
    """Non-temporal evidence describing direct-transition fallback ownership.

    Local-discontinuity values remain descriptive measurements: their mere
    availability does not prove an abrupt transition. The assessment records
    what evidence is available without inventing a threshold or choosing a
    temporal boundary.
    """

    local_discontinuity_complete: bool
    fade_rise_present: bool
    supported_handoffs: tuple[BoundaryHypothesis, ...] = ()

    @property
    def fallback_eligible(self) -> bool:
        """Return whether legacy direct-transition fallback remains available."""
        return (
            self.local_discontinuity_complete
            and not self.fade_rise_present
        )


@dataclass(frozen=True)
class TransitionRoute:
    """Non-temporal result describing transition ownership.

    A route selects the specialist that owns the observed phenomenon and may
    carry already-established routing provenance. Exact temporal boundaries
    remain exclusively specialist-owned.
    """

    kind: TransitionKind
    bounded_insert_ownership: BoundedInsertOwnership | None = None
    hard_cut: HardCutRoutingAssessment | None = None
    fade_assessment: FadeRoutingAssessment | None = None


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

    _fade_confidence_rank = {
        BoundaryConfidence.HIGH: 0,
        BoundaryConfidence.MEDIUM: 1,
        BoundaryConfidence.LOW: 2,
        BoundaryConfidence.UNRESOLVED: 3,
    }

    @classmethod
    def associate_incoming_fade(
        cls,
        *,
        reference_time_seconds: float,
        incoming_fades: tuple[IncomingFadeEvidence, ...],
        hypotheses: tuple[BoundaryHypothesis, ...],
    ) -> IncomingFadeRoutingAssociation | None:
        """Associate incoming fade geometry with coherent boundary support."""
        if reference_time_seconds < 0:
            raise ValueError(
                "reference_time_seconds must be non-negative"
            )

        if not incoming_fades or not hypotheses:
            return None

        candidates = []

        for fade in incoming_fades:
            if fade.established_time_seconds < reference_time_seconds:
                continue

            if fade.start_time_seconds >= reference_time_seconds:
                anchor_time = fade.start_time_seconds
            else:
                anchor_time = fade.established_time_seconds

            supported_hypotheses = [
                hypothesis
                for hypothesis in hypotheses
                if hypothesis.time_seconds >= fade.start_time_seconds
            ]

            if not supported_hypotheses:
                continue

            hypothesis = min(
                supported_hypotheses,
                key=lambda item: (
                    abs(anchor_time - item.time_seconds),
                    abs(anchor_time - reference_time_seconds),
                    cls._fade_confidence_rank[item.confidence],
                    item.time_seconds,
                ),
            )

            support_distance = abs(
                anchor_time - hypothesis.time_seconds
            )
            transition_distance = abs(
                anchor_time - reference_time_seconds
            )

            candidates.append(
                (
                    support_distance,
                    transition_distance,
                    -fade.ascent_consistency,
                    -fade.ascent_ratio,
                    -fade.persistence_ratio,
                    cls._fade_confidence_rank[hypothesis.confidence],
                    anchor_time,
                    fade,
                    hypothesis,
                )
            )

        if not candidates:
            return None

        (
            support_distance,
            _transition_distance,
            _consistency,
            _ratio,
            _persistence,
            _confidence,
            _anchor,
            fade,
            hypothesis,
        ) = min(candidates, key=lambda item: item[:7])

        return IncomingFadeRoutingAssociation(
            fade=fade,
            hypothesis=hypothesis,
            support_distance_seconds=support_distance,
        )

    @classmethod
    def associate_outgoing_fade(
        cls,
        *,
        reference_time_seconds: float,
        outgoing_fades: tuple[OutgoingFadeEvidence, ...],
        hypotheses: tuple[BoundaryHypothesis, ...],
    ) -> OutgoingFadeRoutingAssociation | None:
        """Associate outgoing fade geometry with coherent boundary support."""
        if reference_time_seconds < 0:
            raise ValueError(
                "reference_time_seconds must be non-negative"
            )

        if not outgoing_fades or not hypotheses:
            return None

        eligible_fades = [
            fade
            for fade in outgoing_fades
            if fade.start_time_seconds <= reference_time_seconds
        ]

        if not eligible_fades:
            return None

        pairs = [
            (
                abs(
                    fade.floor_time_seconds
                    - hypothesis.time_seconds
                ),
                cls._fade_confidence_rank[hypothesis.confidence],
                -fade.descent_consistency,
                -fade.descent_ratio,
                -fade.floor_time_seconds,
                fade,
                hypothesis,
            )
            for fade in eligible_fades
            for hypothesis in hypotheses
        ]

        (
            distance,
            _confidence,
            _consistency,
            _ratio,
            _floor,
            fade,
            hypothesis,
        ) = min(pairs, key=lambda item: item[:5])

        return OutgoingFadeRoutingAssociation(
            fade=fade,
            hypothesis=hypothesis,
            support_distance_seconds=distance,
        )

    def assess_fade(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> FadeRoutingAssessment:
        """Correlate fade evidence without selecting temporal boundaries."""
        return FadeRoutingAssessment(
            rise_present=context.fade.rise is not None,
            basin_count=len(context.fade.basins),
            geometry_count=len(context.fade.geometries),
            outgoing_association=self.associate_outgoing_fade(
                reference_time_seconds=context.reference_time_seconds,
                outgoing_fades=context.fade.outgoing_fades,
                hypotheses=context.hypotheses,
            ),
            incoming_association=self.associate_incoming_fade(
                reference_time_seconds=context.reference_time_seconds,
                incoming_fades=context.fade.incoming_fades,
                hypotheses=context.hypotheses,
            ),
        )

    def assess_hard_cut(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> HardCutRoutingAssessment:
        """Describe direct-transition evidence without choosing a boundary."""
        supported_handoffs = tuple(
            hypothesis
            for hypothesis in context.hypotheses
            if (
                hypothesis.basin_geometry is not None
                and {
                    proposal.source
                    for proposal in hypothesis.proposals
                }
                == {
                    BoundaryProposalSource.BASIN,
                    BoundaryProposalSource.STRUCTURAL,
                }
            )
        )

        return HardCutRoutingAssessment(
            local_discontinuity_complete=(
                self._has_complete_local_discontinuity(
                    context.evidence,
                )
            ),
            fade_rise_present=context.fade.rise is not None,
            supported_handoffs=supported_handoffs,
        )

    def classify(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> TransitionKind:
        """Return which specialist owns the transition.

        ``classify()`` remains the compatibility API. Router-v2-aware callers
        may consume ``route()`` to obtain the same kind together with routing
        provenance.
        """
        return self.route(context=context).kind

    def route(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> TransitionRoute:
        """Return specialist ownership without resolving temporal boundaries."""
        fade_assessment = self.assess_fade(
            context=context,
        )

        bounded_insert_ownership = self.bounded_insert_ownership(
            context=context,
        )

        if bounded_insert_ownership is not None:
            return TransitionRoute(
                kind=TransitionKind.BOUNDED_INSERT,
                bounded_insert_ownership=bounded_insert_ownership,
                fade_assessment=fade_assessment,
            )

        fade_like = fade_assessment.legacy_candidate
        interstitial_like = self._has_interstitial_pattern(context.hypotheses)

        if fade_like and interstitial_like:
            if self._has_interstitial_absorbed_right_edge_fade(
                context=context,
            ):
                return TransitionRoute(
                    kind=TransitionKind.INTERSTITIAL,
                    fade_assessment=fade_assessment,
                )

            return TransitionRoute(
                kind=TransitionKind.AMBIGUOUS,
                fade_assessment=fade_assessment,
            )

        if fade_like:
            return TransitionRoute(
                kind=TransitionKind.FADE,
                fade_assessment=fade_assessment,
            )

        if interstitial_like:
            return TransitionRoute(
                kind=TransitionKind.INTERSTITIAL,
                fade_assessment=fade_assessment,
            )

        hard_cut = self.assess_hard_cut(
            context=context,
        )

        if hard_cut.fallback_eligible:
            return TransitionRoute(
                kind=TransitionKind.HARD_CUT,
                hard_cut=hard_cut,
                fade_assessment=fade_assessment,
            )

        return TransitionRoute(
            kind=TransitionKind.AMBIGUOUS,
            fade_assessment=fade_assessment,
        )

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
