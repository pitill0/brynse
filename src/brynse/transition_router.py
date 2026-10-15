"""Routing contract for selecting the specialist that owns a transition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from brynse.acoustic_island import AcousticIslandEvidence
from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryHypothesis,
    BoundaryProposalSource,
)
from brynse.bounded_insert import (
    BoundedInsertAsymmetricRegionEvidence,
    BoundedInsertRegionEvidence,
)
from brynse.fade_to_floor import FadeToFloorEvidence
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
    acoustic_islands: tuple[AcousticIslandEvidence, ...] = ()
    fade_to_floor: tuple[FadeToFloorEvidence, ...] = ()
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
    stale_completed_geometry: bool = False
    reconciled_geometry_count: int = 0

    @property
    def legacy_candidate(self) -> bool:
        """Preserve the pre-Router-v2 fade routing condition."""
        return self.rise_present and self.basin_count > 0

    @property
    def ownership_candidate(self) -> bool:
        """Return whether fade evidence may own the current transition.

        Legacy rise+basin semantics remain available when rich correlation
        cannot say more. When correlated geometry proves that the selected
        outgoing fade already completed into an incoming fade before the
        current reference, that completed earlier phenomenon must not own
        the later transition.
        """
        return self.legacy_candidate and not self.stale_completed_geometry

    @property
    def has_reconciled_geometry(self) -> bool:
        """Return whether both selected fade sides form one exact geometry."""
        return self.reconciled_geometry_count == 1

    @property
    def has_two_sided_support(self) -> bool:
        """Return whether both fade sides have boundary support."""
        return (
            self.outgoing_association is not None
            and self.incoming_association is not None
        )


@dataclass(frozen=True)
class InterstitialRoutingAssessment:
    """Router-owned interpretation of interstitial acoustic evidence.

    Routing handoffs preserve the exact legacy Router criterion. Supported
    handoffs preserve the broader BASIN+STRUCTURAL ownership understood by the
    temporal specialist. Neither collection selects a final boundary.
    """

    routing_handoffs: tuple[BoundaryHypothesis, ...] = ()
    supported_handoffs: tuple[BoundaryHypothesis, ...] = ()
    absorbed_right_edge_fade: bool = False
    terminal_fade_region: BoundedInsertRegionEvidence | None = None
    terminal_fade_candidates: tuple[OutgoingFadeEvidence, ...] = ()

    @property
    def legacy_candidate(self) -> bool:
        """Preserve the pre-Router-v2 interstitial routing condition."""
        return len(self.routing_handoffs) >= 2

    @property
    def has_terminal_fade_support(self) -> bool:
        """Return whether a right-edge region owns terminal fade evidence."""
        return (
            self.terminal_fade_region is not None
            and bool(self.terminal_fade_candidates)
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
class TransitionSemanticRoles:
    """Router-owned semantic roles for already observed acoustic evidence.

    Roles identify which descriptive evidence belongs to each side of a
    transition without selecting the exact temporal geometry within that
    evidence. Specialists remain solely responsible for final split
    boundaries.
    """

    outgoing_handoff: BoundaryHypothesis | None = None
    incoming_handoff: BoundaryHypothesis | None = None
    outgoing_edge_fade: OutgoingFadeEvidence | None = None
    incoming_edge_fade: IncomingFadeEvidence | None = None
    acoustic_island: AcousticIslandEvidence | None = None
    fade_to_floor: FadeToFloorEvidence | None = None
    inserted_region: (
        BoundedInsertRegionEvidence
        | BoundedInsertAsymmetricRegionEvidence
        | None
    ) = None


@dataclass(frozen=True)
class TransitionRoute:
    """Non-temporal result describing transition ownership.

    A route selects the specialist that owns the observed phenomenon and may
    carry already-established routing provenance. Exact temporal boundaries
    remain exclusively specialist-owned.
    """

    kind: TransitionKind
    bounded_insert_ownership: BoundedInsertOwnership | None = None
    bounded_insert: BoundedInsertRoutingProvenance | None = None
    hard_cut: HardCutRoutingAssessment | None = None
    fade_assessment: FadeRoutingAssessment | None = None
    interstitial_assessment: InterstitialRoutingAssessment | None = None
    semantic_roles: TransitionSemanticRoles | None = None


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


@dataclass(frozen=True)
class BoundedInsertRoutingProvenance:
    """Router-owned bounded-insert classification provenance.

    Assessments explain why each evidence family did or did not claim
    ownership. The owned region identifies the acoustic phenomenon handed to
    the specialist without choosing any temporal boundary within that region.
    """

    ownership: BoundedInsertOwnership | None
    symmetric_assessment: BoundedInsertRoutingAssessment
    asymmetric_assessment: BoundedInsertAsymmetricRoutingAssessment
    symmetric_region: BoundedInsertRegionEvidence | None = None
    asymmetric_region: BoundedInsertAsymmetricRegionEvidence | None = None

    @property
    def candidate(self) -> bool:
        """Return whether either bounded-insert family owns the transition."""
        return self.ownership is not None

    @property
    def owned_region(
        self,
    ) -> (
        BoundedInsertRegionEvidence
        | BoundedInsertAsymmetricRegionEvidence
        | None
    ):
        """Return the region selected by existing ownership precedence."""
        if self.ownership is BoundedInsertOwnership.SYMMETRIC:
            return self.symmetric_region

        if self.ownership is BoundedInsertOwnership.ASYMMETRIC:
            return self.asymmetric_region

        return None


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

        left_geometry = getattr(
            region,
            "left_geometry",
            None,
        )

        left_basin_shared_handoff = (
            region.reference_inside_left_basin
            and left_geometry is not None
            and any(
                (
                    geometry.outgoing.floor_time_seconds
                    == geometry.incoming.start_time_seconds
                )
                and (
                    context.reference_time_seconds
                    <= geometry.outgoing.floor_time_seconds
                )
                and (
                    left_geometry.start_seconds
                    <= geometry.outgoing.floor_time_seconds
                    <= left_geometry.recovery_seconds
                )
                and (
                    geometry.outgoing.floor_time_seconds
                    < region.right.time_seconds
                )
                for geometry in context.fade.geometries
            )
        )

        return BoundedInsertRoutingAssessment(
            candidate=(
                not mixed_interior_families
                and not left_basin_shared_handoff
                and (
                    structurally_non_dominant_region
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

    def assess_bounded_insert_routing(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> BoundedInsertRoutingProvenance:
        """Return complete bounded-insert ownership provenance."""
        symmetric = self.assess_bounded_insert(
            context=context,
        )

        asymmetric = self.assess_asymmetric_bounded_insert(
            context=context,
        )

        ownership = None
        symmetric_region = None
        asymmetric_region = None

        if symmetric.candidate:
            ownership = BoundedInsertOwnership.SYMMETRIC

            symmetric_regions = getattr(
                context,
                "bounded_insert_regions",
                (),
            )

            if (
                getattr(symmetric, "unique_region", False)
                and len(symmetric_regions) == 1
            ):
                symmetric_region = symmetric_regions[0]

        elif asymmetric.candidate:
            ownership = BoundedInsertOwnership.ASYMMETRIC

            asymmetric_regions = getattr(
                context,
                "bounded_insert_asymmetric_regions",
                (),
            )

            if (
                getattr(asymmetric, "unique_region", False)
                and len(asymmetric_regions) == 1
            ):
                asymmetric_region = asymmetric_regions[0]

        return BoundedInsertRoutingProvenance(
            ownership=ownership,
            symmetric_assessment=symmetric,
            asymmetric_assessment=asymmetric,
            symmetric_region=symmetric_region,
            asymmetric_region=asymmetric_region,
        )

    def bounded_insert_ownership(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> BoundedInsertOwnership | None:
        """Return which bounded-insert evidence family owns the transition."""
        return self.assess_bounded_insert_routing(
            context=context,
        ).ownership

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
        outgoing_association = self.associate_outgoing_fade(
            reference_time_seconds=context.reference_time_seconds,
            outgoing_fades=context.fade.outgoing_fades,
            hypotheses=context.hypotheses,
        )
        incoming_association = self.associate_incoming_fade(
            reference_time_seconds=context.reference_time_seconds,
            incoming_fades=context.fade.incoming_fades,
            hypotheses=context.hypotheses,
        )

        if incoming_association is not None:
            coherent_outgoing_fades = tuple(
                geometry.outgoing
                for geometry in context.fade.geometries
                if geometry.incoming is incoming_association.fade
            )

            if coherent_outgoing_fades:
                coherent_outgoing_association = (
                    self.associate_outgoing_fade(
                        reference_time_seconds=(
                            context.reference_time_seconds
                        ),
                        outgoing_fades=coherent_outgoing_fades,
                        hypotheses=context.hypotheses,
                    )
                )

                if coherent_outgoing_association is not None:
                    outgoing_association = coherent_outgoing_association

        reconciled_geometry_count = 0

        if (
            outgoing_association is not None
            and incoming_association is not None
        ):
            reconciled_geometry_count = sum(
                1
                for geometry in context.fade.geometries
                if (
                    geometry.outgoing is outgoing_association.fade
                    and geometry.incoming is incoming_association.fade
                )
            )

        stale_completed_geometry = False

        if (
            outgoing_association is not None
            and incoming_association is None
        ):
            associated_geometries = tuple(
                geometry
                for geometry in context.fade.geometries
                if geometry.outgoing is outgoing_association.fade
            )

            stale_completed_geometry = (
                bool(associated_geometries)
                and all(
                    geometry.incoming.established_time_seconds
                    < context.reference_time_seconds
                    for geometry in associated_geometries
                )
            )

        return FadeRoutingAssessment(
            rise_present=context.fade.rise is not None,
            basin_count=len(context.fade.basins),
            geometry_count=len(context.fade.geometries),
            outgoing_association=outgoing_association,
            incoming_association=incoming_association,
            stale_completed_geometry=stale_completed_geometry,
            reconciled_geometry_count=reconciled_geometry_count,
        )

    @staticmethod
    def _interstitial_supported_handoff(
        hypothesis: BoundaryHypothesis,
    ) -> bool:
        """Return whether one hypothesis has basin and structural ownership."""
        sources = {
            proposal.source
            for proposal in hypothesis.proposals
        }

        return (
            hypothesis.basin_geometry is not None
            and BoundaryProposalSource.BASIN in sources
            and BoundaryProposalSource.STRUCTURAL in sources
        )

    @classmethod
    def _interstitial_routing_handoff(
        cls,
        hypothesis: BoundaryHypothesis,
    ) -> bool:
        """Return whether one hypothesis matches the exact legacy criterion."""
        if not cls._interstitial_supported_handoff(hypothesis):
            return False

        sources = {
            proposal.source
            for proposal in hypothesis.proposals
        }

        return sources == {
            BoundaryProposalSource.BASIN,
            BoundaryProposalSource.STRUCTURAL,
        }

    def _interstitial_terminal_fade_ownership(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> tuple[
        BoundedInsertRegionEvidence | None,
        tuple[OutgoingFadeEvidence, ...],
    ]:
        """Correlate terminal fade evidence with one interstitial right edge.

        This method deliberately stops before temporal decision state is
        considered. The specialist still decides whether its current resolved
        outgoing/incoming boundaries make one of these fades usable.
        """
        if context.fade.rise is not None:
            return None, ()

        regions = context.bounded_insert_regions

        if len(regions) != 1:
            return None, ()

        region = regions[0]

        if not region.reference_inside_right_basin:
            return None, ()

        if region.reference_inside_left_basin:
            return None, ()

        if region.structural_only_interior_count != 0:
            return None, ()

        right_geometry = region.right_edge.basin_geometry

        if right_geometry is None:
            return None, ()

        structural_strengths = tuple(
            proposal.strength
            for hypothesis in context.hypotheses
            for proposal in hypothesis.proposals
            if (
                proposal.source is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )

        if not structural_strengths:
            return None, ()

        context_peak = max(structural_strengths)

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
            bool(left_strengths)
            and max(left_strengths) == context_peak
        )

        right_peak = (
            bool(right_strengths)
            and max(right_strengths) == context_peak
        )

        if not left_peak or right_peak:
            return None, ()

        candidates = tuple(
            outgoing_fade
            for outgoing_fade in context.fade.outgoing_fades
            if (
                right_geometry.start_seconds
                <= outgoing_fade.floor_time_seconds
                <= right_geometry.recovery_seconds
            )
        )

        if not candidates:
            return None, ()

        return region, candidates

    def assess_interstitial(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> InterstitialRoutingAssessment:
        """Correlate interstitial ownership without choosing temporal cuts."""
        ordered = tuple(
            sorted(
                context.hypotheses,
                key=lambda hypothesis: hypothesis.time_seconds,
            )
        )

        supported_handoffs = tuple(
            hypothesis
            for hypothesis in ordered
            if self._interstitial_supported_handoff(hypothesis)
        )

        routing_handoffs = tuple(
            hypothesis
            for hypothesis in supported_handoffs
            if self._interstitial_routing_handoff(hypothesis)
        )

        terminal_region, terminal_fades = (
            self._interstitial_terminal_fade_ownership(
                context=context,
            )
        )

        return InterstitialRoutingAssessment(
            routing_handoffs=routing_handoffs,
            supported_handoffs=supported_handoffs,
            absorbed_right_edge_fade=(
                self._has_interstitial_absorbed_right_edge_fade(
                    context=context,
                )
            ),
            terminal_fade_region=terminal_region,
            terminal_fade_candidates=terminal_fades,
        )

    @staticmethod
    def assign_bounded_insert_roles(
        *,
        context: TransitionRoutingContext,
        provenance: BoundedInsertRoutingProvenance,
        fade: FadeRoutingAssessment,
    ) -> TransitionSemanticRoles | None:
        """Assign bounded-insert evidence identities without choosing cuts.

        The owned region identifies the inserted phenomenon itself.  Exterior
        handoffs are published only when existing acoustic relationships make
        their identity defensible.  Missing exterior identity remains unknown.
        """

        region = provenance.owned_region

        if region is None:
            return None

        def proposal_sources(hypothesis):
            return {
                proposal.source
                for proposal in hypothesis.proposals
            }

        def supported(hypothesis) -> bool:
            sources = proposal_sources(hypothesis)

            return (
                BoundaryProposalSource.BASIN in sources
                and BoundaryProposalSource.STRUCTURAL in sources
                and hypothesis.basin_geometry is not None
            )

        def structural_only(hypothesis) -> bool:
            return proposal_sources(hypothesis) == {
                BoundaryProposalSource.STRUCTURAL
            }

        ordered = tuple(
            sorted(
                context.hypotheses,
                key=lambda hypothesis: hypothesis.time_seconds,
            )
        )

        if provenance.ownership is BoundedInsertOwnership.SYMMETRIC:
            outgoing = None
            incoming = region.right

            association = fade.outgoing_association

            if association is not None:
                candidate = association.hypothesis

                if candidate is region.left:
                    outgoing = candidate

                elif (
                    candidate.time_seconds < region.left.time_seconds
                    and supported(candidate)
                ):
                    intervening = tuple(
                        hypothesis
                        for hypothesis in ordered
                        if (
                            candidate.time_seconds
                            < hypothesis.time_seconds
                            < region.left.time_seconds
                        )
                    )

                    if all(
                        structural_only(hypothesis)
                        for hypothesis in intervening
                    ):
                        outgoing = candidate

            if outgoing is None:
                previous_supported = tuple(
                    hypothesis
                    for hypothesis in ordered
                    if (
                        hypothesis.time_seconds < region.left.time_seconds
                        and supported(hypothesis)
                    )
                )

                if previous_supported:
                    predecessor = previous_supported[-1]

                    intervening = tuple(
                        hypothesis
                        for hypothesis in ordered
                        if (
                            predecessor.time_seconds
                            < hypothesis.time_seconds
                            < region.left.time_seconds
                        )
                    )

                    if (
                        intervening
                        and all(
                            structural_only(hypothesis)
                            for hypothesis in intervening
                        )
                    ):
                        outgoing = predecessor

            return TransitionSemanticRoles(
                outgoing_handoff=outgoing,
                inserted_region=region,
                incoming_handoff=incoming,
            )

        if provenance.ownership is BoundedInsertOwnership.ASYMMETRIC:
            outgoing = None
            incoming = None

            previous_supported = tuple(
                hypothesis
                for hypothesis in ordered
                if (
                    hypothesis.time_seconds < region.primary.time_seconds
                    and supported(hypothesis)
                )
            )

            if previous_supported:
                predecessor = previous_supported[-1]

                intervening = tuple(
                    hypothesis
                    for hypothesis in ordered
                    if (
                        predecessor.time_seconds
                        < hypothesis.time_seconds
                        < region.primary.time_seconds
                    )
                )

                if all(
                    structural_only(hypothesis)
                    for hypothesis in intervening
                ):
                    outgoing = predecessor

            matching_outer = tuple(
                candidate
                for candidate in context.bounded_insert_regions
                if candidate.left is region.primary
            )

            if len(matching_outer) == 1:
                incoming = matching_outer[0].right

            return TransitionSemanticRoles(
                outgoing_handoff=outgoing,
                inserted_region=region,
                incoming_handoff=incoming,
            )

        return TransitionSemanticRoles(
            inserted_region=region,
        )

    @staticmethod
    def assign_interstitial_roles(
        *,
        fade: FadeRoutingAssessment,
        interstitial: InterstitialRoutingAssessment,
        context: TransitionRoutingContext | None = None,
    ) -> TransitionSemanticRoles | None:
        """Assign Interstitial-owned semantic topology.

        Interstitial ownership and Interstitial topology are deliberately
        separate from Fade ownership.

        With routing context available, Router derives topology only from
        Interstitial-owned identity:

        * two distinct reconciled fade geometries may provide descriptive edge
          evidence only when each edge is uniquely owned by a different
          Interstitial supported handoff;
        * otherwise exactly two Interstitial routing handoffs may establish
          two identities;
        * a bounded relationship may complete one missing side only from an
          already-established outgoing identity.  A region alone never
          manufactures both Interstitial identities.

        Selected Fade associations may corroborate one missing side, but they
        must never manufacture both Interstitial roles.  In particular, two
        associations landing on the same late hypothesis cannot collapse a
        richer Interstitial transition into a false shared handoff.

        Calls without routing context remain conservative: only topology
        already owned by Interstitial routing handoffs may be published.
        Fade associations never manufacture Interstitial identities.
        """
        supported = interstitial.supported_handoffs

        if context is not None:
            reference = context.reference_time_seconds
            regions = context.bounded_insert_regions

            if len(regions) == 1:
                region = regions[0]

                if (
                    region.left in supported
                    and region.right in supported
                    and not region.reference_inside_left_basin
                    and not region.reference_inside_right_basin
                    and region.left.time_seconds < reference
                    and reference < region.right.time_seconds
                ):
                    return TransitionSemanticRoles(
                        outgoing_handoff=region.left,
                        incoming_handoff=region.right,
                        inserted_region=region,
                    )

            geometries = tuple(context.fade.geometries)

            def unique_supported_owner(time_seconds: float):
                owners = tuple(
                    hypothesis
                    for hypothesis in supported
                    if (
                        hypothesis.basin_geometry is not None
                        and (
                            hypothesis.basin_geometry.start_seconds
                            <= time_seconds
                            <= hypothesis.basin_geometry.recovery_seconds
                        )
                    )
                )

                if len(owners) != 1:
                    return None

                return owners[0]

            completed_geometries = tuple(
                (
                    geometry,
                    unique_supported_owner(
                        geometry.outgoing.floor_time_seconds
                    ),
                )
                for geometry in geometries
                if (
                    geometry.incoming.established_time_seconds
                    < reference
                )
            )
            completed_geometries = tuple(
                item
                for item in completed_geometries
                if item[1] is not None
            )

            upcoming_geometries = tuple(
                (
                    geometry,
                    unique_supported_owner(
                        geometry.incoming.start_time_seconds
                    ),
                )
                for geometry in geometries
                if geometry.incoming.start_time_seconds >= reference
            )
            upcoming_geometries = tuple(
                item
                for item in upcoming_geometries
                if item[1] is not None
            )

            if completed_geometries and upcoming_geometries:
                outgoing_geometry, outgoing_owner = max(
                    completed_geometries,
                    key=lambda item: (
                        item[0].incoming.established_time_seconds,
                        item[0].outgoing.floor_time_seconds,
                    ),
                )
                incoming_geometry, incoming_owner = min(
                    upcoming_geometries,
                    key=lambda item: (
                        item[0].incoming.start_time_seconds,
                        item[0].outgoing.floor_time_seconds,
                    ),
                )

                if (
                    outgoing_geometry is not incoming_geometry
                    and outgoing_owner is not incoming_owner
                ):
                    return TransitionSemanticRoles(
                        outgoing_handoff=outgoing_owner,
                        incoming_handoff=incoming_owner,
                        outgoing_edge_fade=outgoing_geometry.outgoing,
                        incoming_edge_fade=incoming_geometry.incoming,
                    )

            if not context.bounded_insert_regions:
                structural_only = tuple(
                    hypothesis
                    for hypothesis in getattr(
                        context,
                        "hypotheses",
                        (),
                    )
                    if {
                        proposal.source
                        for proposal in hypothesis.proposals
                    } == {
                        BoundaryProposalSource.STRUCTURAL
                    }
                )

                mixed_left = []

                for geometry in geometries:
                    incoming = geometry.incoming
                    outgoing = geometry.outgoing

                    if (
                        incoming.established_time_seconds
                        >= reference
                    ):
                        continue

                    witnesses = tuple(
                        hypothesis
                        for hypothesis in structural_only
                        if (
                            max(
                                incoming.start_time_seconds,
                                outgoing.floor_time_seconds,
                            )
                            <= hypothesis.time_seconds
                            <= incoming.established_time_seconds
                        )
                    )

                    if len(witnesses) == 1:
                        mixed_left.append(geometry)

                mixed_right_owners = []

                for geometry in geometries:
                    incoming_start = (
                        geometry.incoming.start_time_seconds
                    )

                    if incoming_start < reference:
                        continue

                    owners = tuple(
                        hypothesis
                        for hypothesis in supported
                        if (
                            hypothesis.basin_geometry is not None
                            and hypothesis.basin_geometry.start_seconds
                            <= incoming_start
                            <= hypothesis.basin_geometry.recovery_seconds
                        )
                    )

                    if len(owners) != 1:
                        continue

                    owner = owners[0]

                    if not any(
                        owner is existing
                        for existing in mixed_right_owners
                    ):
                        mixed_right_owners.append(owner)

                if (
                    len(mixed_left) == 1
                    and len(mixed_right_owners) == 1
                ):
                    return TransitionSemanticRoles(
                        outgoing_edge_fade=(
                            mixed_left[0].outgoing
                        ),
                        incoming_handoff=(
                            mixed_right_owners[0]
                        ),
                    )

            routing_handoffs = interstitial.routing_handoffs

            if len(routing_handoffs) == 2:
                return TransitionSemanticRoles(
                    outgoing_handoff=routing_handoffs[0],
                    incoming_handoff=routing_handoffs[1],
                )

            outgoing = (
                fade.outgoing_association.hypothesis
                if (
                    fade.outgoing_association is not None
                    and fade.outgoing_association.hypothesis in supported
                )
                else None
            )
            incoming = (
                fade.incoming_association.hypothesis
                if (
                    fade.incoming_association is not None
                    and fade.incoming_association.hypothesis in supported
                )
                else None
            )

            if outgoing is not None and incoming is None:
                if len(context.bounded_insert_regions) == 1:
                    region = context.bounded_insert_regions[0]

                    if (
                        region.left is outgoing
                        and region.right in supported
                    ):
                        return TransitionSemanticRoles(
                            outgoing_handoff=outgoing,
                            incoming_handoff=region.right,
                        )

                return TransitionSemanticRoles(
                    outgoing_handoff=outgoing,
                )

            if incoming is not None and outgoing is None:
                return TransitionSemanticRoles(
                    incoming_handoff=incoming,
                )

            return None

        routing_handoffs = interstitial.routing_handoffs

        if len(routing_handoffs) != 2:
            return None

        return TransitionSemanticRoles(
            outgoing_handoff=routing_handoffs[0],
            incoming_handoff=routing_handoffs[1],
        )

    @staticmethod
    def assign_acoustic_island_roles(
        *,
        context: TransitionRoutingContext,
    ) -> TransitionSemanticRoles | None:
        """Publish uniquely observed acoustic-island topology without cutting."""
        if len(context.acoustic_islands) != 1:
            return None

        island = context.acoustic_islands[0]
        region = island.observed_region

        return TransitionSemanticRoles(
            outgoing_handoff=(
                None
                if region is None
                else region.left
            ),
            incoming_handoff=island.incoming_handoff,
            acoustic_island=island,
            inserted_region=region,
        )

    @staticmethod
    def assign_fade_to_floor_roles(
        *,
        context: TransitionRoutingContext,
    ) -> TransitionSemanticRoles | None:
        """Publish one direct fade-to-floor topology without choosing cuts."""

        if len(context.fade_to_floor) != 1:
            return None

        evidence = context.fade_to_floor[0]

        if evidence.incoming_handoff is None:
            return None

        return TransitionSemanticRoles(
            incoming_handoff=evidence.incoming_handoff,
            outgoing_edge_fade=evidence.outgoing,
            fade_to_floor=evidence,
        )

    @staticmethod
    def enrich_interstitial_roles_with_fade_to_floor(
        *,
        roles: TransitionSemanticRoles | None,
        context: TransitionRoutingContext,
    ) -> TransitionSemanticRoles | None:
        """Add outgoing-only FadeToFloor evidence to owned Interstitial roles."""

        if roles is None or len(context.fade_to_floor) != 1:
            return roles

        evidence = context.fade_to_floor[0]

        if (
            evidence.incoming_handoff is not None
            or roles.outgoing_edge_fade is not None
        ):
            return roles

        return TransitionSemanticRoles(
            outgoing_handoff=roles.outgoing_handoff,
            incoming_handoff=roles.incoming_handoff,
            outgoing_edge_fade=evidence.outgoing,
            incoming_edge_fade=roles.incoming_edge_fade,
            acoustic_island=roles.acoustic_island,
            fade_to_floor=evidence,
            inserted_region=roles.inserted_region,
        )

    @staticmethod
    def assign_outgoing_fade_to_floor_roles(
        *,
        context: TransitionRoutingContext,
    ) -> TransitionSemanticRoles | None:
        """Publish outgoing-only FadeToFloor authority."""

        if len(context.fade_to_floor) != 1:
            return None

        evidence = context.fade_to_floor[0]

        if evidence.incoming_handoff is not None:
            return None

        return TransitionSemanticRoles(
            outgoing_edge_fade=evidence.outgoing,
            fade_to_floor=evidence,
        )

    @staticmethod
    def resolve_fade_interstitial_conflict(
        *,
        fade: FadeRoutingAssessment,
        interstitial: InterstitialRoutingAssessment,
    ) -> TransitionKind:
        """Resolve overlapping Fade/Interstitial ownership semantically.

        This is deliberately not a score or confidence vote.

        Interstitial owns a fade already proven to belong to its terminal
        acoustic edge. Otherwise, a fade with boundary support on both sides
        is a complete phenomenon and owns the transition. A legacy fade
        candidate without two-sided association does not override an already
        established interstitial phenomenon.
        """
        if interstitial.absorbed_right_edge_fade:
            return TransitionKind.INTERSTITIAL

        if fade.has_two_sided_support:
            return TransitionKind.FADE

        return TransitionKind.INTERSTITIAL

    @staticmethod
    def assign_hard_cut_roles(
        *,
        fade: FadeRoutingAssessment,
        hard_cut: HardCutRoutingAssessment,
    ) -> TransitionSemanticRoles | None:
        """Return a shared handoff only when both acoustic sides agree.

        A genuine HARD_CUT has one semantic event shared by outgoing and
        incoming material.  Local-discontinuity availability and a lone
        supported hypothesis are not sufficient proof.

        The Router therefore requires both independently correlated fade sides
        to identify the same unique supported handoff.  No temporal geometry
        inside that hypothesis is selected here.
        """
        if len(hard_cut.supported_handoffs) != 1:
            return None

        shared = hard_cut.supported_handoffs[0]

        outgoing = fade.outgoing_association
        incoming = fade.incoming_association

        if outgoing is None or incoming is None:
            return None

        if outgoing.hypothesis is not shared:
            return None

        if incoming.hypothesis is not shared:
            return None

        return TransitionSemanticRoles(
            outgoing_handoff=shared,
            incoming_handoff=shared,
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

        interstitial_assessment = self.assess_interstitial(
            context=context,
        )

        bounded_insert = self.assess_bounded_insert_routing(
            context=context,
        )

        bounded_insert_ownership = self.bounded_insert_ownership(
            context=context,
        )

        if bounded_insert_ownership is not bounded_insert.ownership:
            bounded_insert = BoundedInsertRoutingProvenance(
                ownership=bounded_insert_ownership,
                symmetric_assessment=(
                    bounded_insert.symmetric_assessment
                ),
                asymmetric_assessment=(
                    bounded_insert.asymmetric_assessment
                ),
                symmetric_region=(
                    bounded_insert.symmetric_region
                    if (
                        bounded_insert_ownership
                        is BoundedInsertOwnership.SYMMETRIC
                    )
                    else None
                ),
                asymmetric_region=(
                    bounded_insert.asymmetric_region
                    if (
                        bounded_insert_ownership
                        is BoundedInsertOwnership.ASYMMETRIC
                    )
                    else None
                ),
            )

        if bounded_insert_ownership is not None:
            return TransitionRoute(
                kind=TransitionKind.BOUNDED_INSERT,
                bounded_insert_ownership=bounded_insert_ownership,
                bounded_insert=bounded_insert,
                fade_assessment=fade_assessment,
                interstitial_assessment=interstitial_assessment,
                semantic_roles=self.assign_bounded_insert_roles(
                    context=context,
                    provenance=bounded_insert,
                    fade=fade_assessment,
                ),
            )

        acoustic_island_roles = self.assign_acoustic_island_roles(
            context=context,
        )

        region_backed_acoustic_island = (
            acoustic_island_roles is not None
            and acoustic_island_roles.acoustic_island is not None
            and (
                acoustic_island_roles
                .acoustic_island
                .observed_region
                is not None
            )
        )

        if region_backed_acoustic_island:
            return TransitionRoute(
                kind=TransitionKind.INTERSTITIAL,
                fade_assessment=fade_assessment,
                interstitial_assessment=interstitial_assessment,
                semantic_roles=acoustic_island_roles,
            )

        fade_to_floor_roles = self.assign_fade_to_floor_roles(
            context=context,
        )

        if fade_to_floor_roles is not None:
            return TransitionRoute(
                kind=TransitionKind.FADE,
                fade_assessment=fade_assessment,
                interstitial_assessment=interstitial_assessment,
                semantic_roles=fade_to_floor_roles,
            )

        if acoustic_island_roles is not None:
            return TransitionRoute(
                kind=TransitionKind.INTERSTITIAL,
                fade_assessment=fade_assessment,
                interstitial_assessment=interstitial_assessment,
                semantic_roles=acoustic_island_roles,
            )

        fade_like = fade_assessment.ownership_candidate
        interstitial_like = interstitial_assessment.legacy_candidate

        if fade_like and interstitial_like:
            # Independent support on both fade sides is not sufficient to
            # prove that Fade owns the whole transition.  If Router cannot
            # correlate those selected sides through exactly one shared
            # FadeGeometryEvidence, the richer interstitial phenomenon owns
            # the conflict instead of letting Fade compose unrelated edges.
            if not fade_assessment.has_reconciled_geometry:
                conflict_kind = TransitionKind.INTERSTITIAL
            else:
                conflict_kind = self.resolve_fade_interstitial_conflict(
                    fade=fade_assessment,
                    interstitial=interstitial_assessment,
                )

            semantic_roles = (
                self.enrich_interstitial_roles_with_fade_to_floor(
                    roles=self.assign_interstitial_roles(
                        fade=fade_assessment,
                        interstitial=interstitial_assessment,
                        context=context,
                    ),
                    context=context,
                )
                if conflict_kind is TransitionKind.INTERSTITIAL
                else None
            )

            return TransitionRoute(
                kind=conflict_kind,
                fade_assessment=fade_assessment,
                interstitial_assessment=interstitial_assessment,
                semantic_roles=semantic_roles,
            )

        if fade_like:
            return TransitionRoute(
                kind=TransitionKind.FADE,
                fade_assessment=fade_assessment,
                interstitial_assessment=interstitial_assessment,
            )

        if interstitial_like:
            return TransitionRoute(
                kind=TransitionKind.INTERSTITIAL,
                fade_assessment=fade_assessment,
                interstitial_assessment=interstitial_assessment,
                semantic_roles=(
                    self.enrich_interstitial_roles_with_fade_to_floor(
                        roles=self.assign_interstitial_roles(
                            fade=fade_assessment,
                            interstitial=interstitial_assessment,
                            context=context,
                        ),
                        context=context,
                    )
                ),
            )

        outgoing_fade_to_floor_roles = (
            self.assign_outgoing_fade_to_floor_roles(
                context=context,
            )
        )

        if outgoing_fade_to_floor_roles is not None:
            return TransitionRoute(
                kind=TransitionKind.FADE,
                fade_assessment=fade_assessment,
                interstitial_assessment=interstitial_assessment,
                semantic_roles=outgoing_fade_to_floor_roles,
            )

        hard_cut = self.assess_hard_cut(
            context=context,
        )

        hard_cut_roles = self.assign_hard_cut_roles(
            fade=fade_assessment,
            hard_cut=hard_cut,
        )

        if (
            hard_cut.fallback_eligible
            and hard_cut_roles is not None
        ):
            return TransitionRoute(
                kind=TransitionKind.HARD_CUT,
                hard_cut=hard_cut,
                fade_assessment=fade_assessment,
                interstitial_assessment=interstitial_assessment,
                semantic_roles=hard_cut_roles,
            )

        return TransitionRoute(
            kind=TransitionKind.AMBIGUOUS,
            hard_cut=hard_cut,
            fade_assessment=fade_assessment,
            interstitial_assessment=interstitial_assessment,
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
