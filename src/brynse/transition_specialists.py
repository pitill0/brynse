"""Temporal specialists for routed acoustic transitions.

The router decides WHO owns a transition.
A specialist decides WHERE its temporal boundaries are.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from brynse.boundaries import BoundaryConfidence, BoundaryProposalSource
from brynse.models import TemporalSplitDecision, TemporalSplitKind
from brynse.transition_router import (
    BoundedInsertOwnership,
    TransitionKind,
    TransitionRouter,
    TransitionRoutingContext,
)


@dataclass(frozen=True)
class TransitionSpecialistInput:
    """Neutral observation material handed to one transition specialist."""

    routing: TransitionRoutingContext
    selected: Any
    profile: Any
    window: Any
    pcm8: Any
    pcm16: Any
    bounded_insert_ownership: BoundedInsertOwnership | None = None


@dataclass(frozen=True)
class TransitionSpecialistResult:
    """Temporal decision plus optional specialist-owned diagnostics."""

    temporal: TemporalSplitDecision | None
    diagnostics: Any = None


class TransitionSpecialist(Protocol):
    """Resolve temporal boundaries for one owned transition phenomenon."""

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        """Return the specialist's complete resolution."""
        ...


class HardCutSpecialist:
    """Resolve the acoustic boundary of a genuine abrupt transition."""

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        boundary = self._resolve_boundary(
            selected_time_seconds=observation.selected.time_seconds,
            reference_time_seconds=(
                observation.routing.reference_time_seconds
            ),
            hypotheses=observation.routing.hypotheses,
        )

        return TransitionSpecialistResult(
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                incoming_start_seconds=boundary,
                outgoing_end_seconds=boundary,
            )
        )

    @classmethod
    def _resolve_boundary(
        cls,
        *,
        selected_time_seconds: float,
        reference_time_seconds: float,
        hypotheses,
    ) -> float:
        geometries = tuple(
            (
                hypothesis,
                hypothesis.basin_geometry,
            )
            for hypothesis in hypotheses
            if hypothesis.basin_geometry is not None
        )

        selected_geometry = cls._containing_geometry(
            time_seconds=selected_time_seconds,
            geometries=geometries,
        )

        if selected_geometry is not None:
            if (
                selected_time_seconds
                >= selected_geometry.minimum_seconds
            ):
                return selected_geometry.recovery_seconds

            selected_hypothesis = cls._hypothesis_for_geometry(
                geometry=selected_geometry,
                geometries=geometries,
            )
            selected_sources = cls._proposal_sources(
                selected_hypothesis,
            )

            if (
                selected_sources is not None
                and BoundaryProposalSource.STRUCTURAL
                in selected_sources
                and reference_time_seconds
                >= selected_geometry.recovery_seconds
            ):
                return selected_geometry.recovery_seconds

            if (
                selected_sources is not None
                and BoundaryProposalSource.BASIN
                in selected_sources
                and BoundaryProposalSource.STRUCTURAL
                not in selected_sources
                and selected_time_seconds
                < selected_geometry.minimum_seconds
                and selected_geometry.start_seconds
                <= reference_time_seconds
                <= selected_geometry.minimum_seconds
            ):
                return reference_time_seconds

            if (
                selected_geometry.start_seconds
                <= reference_time_seconds
                <= selected_geometry.minimum_seconds
            ):
                return selected_geometry.minimum_seconds

            return selected_time_seconds

        reference_geometry = cls._containing_geometry(
            time_seconds=reference_time_seconds,
            geometries=geometries,
        )

        if reference_geometry is not None:
            if (
                reference_time_seconds
                <= reference_geometry.minimum_seconds
            ):
                reference_hypothesis = cls._hypothesis_for_geometry(
                    geometry=reference_geometry,
                    geometries=geometries,
                )
                reference_sources = cls._proposal_sources(
                    reference_hypothesis,
                )

                if (
                    reference_sources is not None
                    and BoundaryProposalSource.BASIN
                    in reference_sources
                    and BoundaryProposalSource.STRUCTURAL
                    not in reference_sources
                ):
                    return reference_time_seconds

                return reference_geometry.minimum_seconds

            return reference_geometry.recovery_seconds

        enclosed = tuple(
            geometry
            for _, geometry in geometries
            if (
                selected_time_seconds
                < geometry.start_seconds
                and geometry.recovery_seconds
                < reference_time_seconds
            )
        )

        if len(enclosed) == 1:
            return enclosed[0].recovery_seconds

        if geometries:
            return selected_time_seconds

        if hypotheses:
            hypothesis = min(
                hypotheses,
                key=lambda item: (
                    abs(
                        item.time_seconds
                        - reference_time_seconds
                    ),
                    item.time_seconds,
                ),
            )
            return hypothesis.time_seconds

        return selected_time_seconds

    @staticmethod
    def _hypothesis_for_geometry(
        *,
        geometry,
        geometries,
    ):
        for hypothesis, candidate in geometries:
            if candidate is geometry:
                return hypothesis

        return None

    @staticmethod
    def _proposal_sources(hypothesis):
        proposals = getattr(
            hypothesis,
            "proposals",
            None,
        )

        if proposals is None:
            return None

        return {
            proposal.source
            for proposal in proposals
            if hasattr(proposal, "source")
        }

    @staticmethod
    def _containing_geometry(
        *,
        time_seconds: float,
        geometries,
    ):
        containing = tuple(
            geometry
            for _, geometry in geometries
            if (
                geometry.start_seconds
                <= time_seconds
                <= geometry.recovery_seconds
            )
        )

        if not containing:
            return None

        return min(
            containing,
            key=lambda geometry: (
                abs(
                    geometry.minimum_seconds
                    - time_seconds
                ),
                geometry.minimum_seconds,
            ),
        )


class InterstitialSpecialist:
    """Resolve track -> interstitial -> incoming-track geometry.

    Incoming refinement establishes the right edge of an interstitial region.
    Once that region exists, this specialist independently analyzes acoustic
    hypotheses to refine its left edge.
    """

    def __init__(self, *, incoming_pipeline) -> None:
        self._incoming_pipeline = incoming_pipeline

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        boundary = observation.selected.time_seconds

        initial = TemporalSplitDecision(
            kind=TemporalSplitKind.HARD_CUT,
            incoming_start_seconds=boundary,
            outgoing_end_seconds=boundary,
        )

        if self._incoming_pipeline is None:
            return TransitionSpecialistResult(
                temporal=initial,
            )

        refined = self._incoming_pipeline.refine(
            decision=initial,
            window=observation.window,
            pcm8=observation.pcm8,
            pcm16=observation.pcm16,
            hypotheses=observation.routing.hypotheses,
        )

        if refined.kind is not TemporalSplitKind.EXCLUSION:
            recovered = self._recover_supported_exclusion(
                selected_time_seconds=boundary,
                hypotheses=observation.routing.hypotheses,
            )

            if recovered is not None:
                return TransitionSpecialistResult(
                    temporal=recovered,
                )

            return TransitionSpecialistResult(
                temporal=refined,
            )

        phase_refined = self._refine_dominant_incoming_basin(
            decision=refined,
            reference_time_seconds=(
                observation.routing.reference_time_seconds
            ),
            hypotheses=observation.routing.hypotheses,
        )

        if phase_refined is not None:
            return TransitionSpecialistResult(
                temporal=phase_refined,
            )

        analyze_hypotheses = getattr(
            self._incoming_pipeline,
            "analyze_hypotheses",
            None,
        )
        if analyze_hypotheses is None:
            return TransitionSpecialistResult(
                temporal=refined,
            )

        analyses = analyze_hypotheses(
            hypotheses=observation.routing.hypotheses,
            window=observation.window,
            pcm16=observation.pcm16,
        )

        outgoing = self._discover_outgoing(
            selected_time_seconds=boundary,
            incoming_start_seconds=refined.incoming_start_seconds,
            analyses=analyses,
        )

        if outgoing is None:
            return TransitionSpecialistResult(
                temporal=refined,
            )

        return TransitionSpecialistResult(
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=outgoing,
                incoming_start_seconds=refined.incoming_start_seconds,
            )
        )

    @staticmethod
    def _refine_dominant_incoming_basin(
        *,
        decision: TemporalSplitDecision,
        reference_time_seconds: float,
        hypotheses,
    ) -> TemporalSplitDecision | None:
        from brynse.boundaries import BoundaryProposalSource

        structural_strengths = tuple(
            proposal.strength
            for hypothesis in hypotheses
            for proposal in getattr(
                hypothesis,
                "proposals",
                (),
            )
            if (
                proposal.source is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )

        if not structural_strengths:
            return None

        context_peak = max(structural_strengths)

        matching = []

        for hypothesis in hypotheses:
            geometry = hypothesis.basin_geometry

            if geometry is None:
                continue

            if (
                geometry.recovery_seconds
                != decision.incoming_start_seconds
            ):
                continue

            strengths = tuple(
                proposal.strength
                for proposal in getattr(
                    hypothesis,
                    "proposals",
                    (),
                )
                if (
                    proposal.source
                    is BoundaryProposalSource.STRUCTURAL
                    and proposal.strength is not None
                )
            )

            if not strengths:
                continue

            if max(strengths) != context_peak:
                continue

            matching.append(
                (
                    hypothesis,
                    geometry,
                )
            )

        if len(matching) != 1:
            return None

        hypothesis, geometry = matching[0]

        basin_proposal_times = tuple(
            proposal.time_seconds
            for proposal in getattr(
                hypothesis,
                "proposals",
                (),
            )
            if (
                proposal.source
                is BoundaryProposalSource.BASIN
            )
        )

        if len(basin_proposal_times) != 1:
            return None

        if (
            reference_time_seconds
            < basin_proposal_times[0]
        ):
            return None

        if not (
            decision.outgoing_end_seconds
            < geometry.start_seconds
            < decision.incoming_start_seconds
        ):
            return None

        dominant_structural_times = tuple(
            proposal.time_seconds
            for proposal in getattr(
                hypothesis,
                "proposals",
                (),
            )
            if (
                proposal.source
                is BoundaryProposalSource.STRUCTURAL
                and proposal.strength == context_peak
            )
        )

        outgoing = geometry.start_seconds

        if geometry.recovery_seconds in dominant_structural_times:
            outgoing = geometry.minimum_seconds

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=outgoing,
            incoming_start_seconds=decision.incoming_start_seconds,
        )

    @staticmethod
    def _recover_supported_exclusion(
        *,
        selected_time_seconds: float,
        hypotheses,
    ) -> TemporalSplitDecision | None:
        from brynse.boundaries import BoundaryProposalSource

        ordered = tuple(
            sorted(
                hypotheses,
                key=lambda hypothesis: hypothesis.time_seconds,
            )
        )

        before = tuple(
            hypothesis
            for hypothesis in ordered
            if hypothesis.time_seconds < selected_time_seconds
        )
        after = tuple(
            hypothesis
            for hypothesis in ordered
            if hypothesis.time_seconds > selected_time_seconds
        )

        if not before or not after:
            return None

        left = before[-1]
        right = after[0]

        def supported(hypothesis) -> bool:
            sources = {
                proposal.source
                for proposal in hypothesis.proposals
            }

            return (
                hypothesis.basin_geometry is not None
                and BoundaryProposalSource.BASIN in sources
                and BoundaryProposalSource.STRUCTURAL in sources
            )

        if not supported(left) or not supported(right):
            return None

        left_distance = (
            selected_time_seconds
            - left.time_seconds
        )
        right_distance = (
            right.time_seconds
            - selected_time_seconds
        )

        if left_distance <= right_distance:
            outgoing = min(
                left.time_seconds,
                left.basin_geometry.recovery_seconds,
            )
        else:
            outgoing = selected_time_seconds

        incoming = max(
            right.time_seconds,
            right.basin_geometry.recovery_seconds,
        )

        if outgoing >= incoming:
            return None

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=outgoing,
            incoming_start_seconds=incoming,
        )

    @staticmethod
    def _discover_outgoing(
        *,
        selected_time_seconds: float,
        incoming_start_seconds: float,
        analyses,
    ) -> float | None:
        supported = tuple(
            analysis.hypothesis.time_seconds
            for analysis in analyses
            if (
                analysis.assessment.candidate_confidence
                is BoundaryConfidence.HIGH
                and analysis.hypothesis.time_seconds
                < incoming_start_seconds
            )
        )

        if not supported:
            return None

        return min(
            supported,
            key=lambda time_seconds: (
                abs(time_seconds - selected_time_seconds),
                time_seconds,
            ),
        )


class BoundedInsertSpecialist:
    """Resolve temporal geometry for acoustically bounded inserted material."""

    @staticmethod
    def _proposal_sources(hypothesis):
        return {
            proposal.source
            for proposal in hypothesis.proposals
        }

    @classmethod
    def _is_supported_edge(cls, hypothesis) -> bool:
        from brynse.boundaries import BoundaryProposalSource

        sources = cls._proposal_sources(hypothesis)

        return (
            BoundaryProposalSource.BASIN in sources
            and BoundaryProposalSource.STRUCTURAL in sources
            and hypothesis.basin_geometry is not None
        )

    @classmethod
    def _is_structural_only(cls, hypothesis) -> bool:
        from brynse.boundaries import BoundaryProposalSource

        return cls._proposal_sources(hypothesis) == {
            BoundaryProposalSource.STRUCTURAL
        }

    @classmethod
    def propose_structural_bridge_cut(
        cls,
        region,
    ):
        """Collapse one structural bridge that owns the semantic handoff."""
        interior = region.interior_hypotheses

        if len(interior) != 1:
            return None

        bridge = interior[0]

        if not cls._is_structural_only(bridge):
            return None

        nearest = min(
            (
                region.left,
                bridge,
                region.right,
            ),
            key=lambda hypothesis: (
                abs(
                    hypothesis.time_seconds
                    - region.reference_time_seconds
                ),
                hypothesis.time_seconds,
            ),
        )

        if nearest is not bridge:
            return None

        boundary = bridge.time_seconds

        return TemporalSplitDecision(
            kind=TemporalSplitKind.HARD_CUT,
            outgoing_end_seconds=boundary,
            incoming_start_seconds=boundary,
        )

    @classmethod
    def propose_symmetric_outer_extent(
        cls,
        *,
        region,
        hypotheses,
    ):
        """Recover one supported predecessor through structural-only interior."""
        anchor = region.left

        previous_supported = tuple(
            hypothesis
            for hypothesis in hypotheses
            if (
                hypothesis.time_seconds < anchor.time_seconds
                and cls._is_supported_edge(hypothesis)
            )
        )

        if not previous_supported:
            return None

        predecessor = max(
            previous_supported,
            key=lambda hypothesis: hypothesis.time_seconds,
        )

        interior = tuple(
            hypothesis
            for hypothesis in hypotheses
            if (
                predecessor.time_seconds
                < hypothesis.time_seconds
                < anchor.time_seconds
            )
        )

        if not interior:
            return None

        if not all(
            cls._is_structural_only(hypothesis)
            for hypothesis in interior
        ):
            return None

        outgoing = predecessor.time_seconds
        incoming = region.right.time_seconds

        if outgoing >= incoming:
            return None

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=outgoing,
            incoming_start_seconds=incoming,
        )

    @staticmethod
    def propose_symmetric_phase_aware(
        region,
        *,
        reference_time_seconds=None,
    ):
        """Propose a phase-aware symmetric exclusion without applying it."""
        if reference_time_seconds is None:
            reference_time_seconds = region.reference_time_seconds

        outgoing = region.left.time_seconds

        geometry = region.left_edge.basin_geometry

        if geometry is not None:
            outgoing = min(
                outgoing,
                geometry.recovery_seconds,
            )

        later_reference_basins = tuple(
            evidence
            for evidence in region.left.evidence
            if (
                evidence.basin_start_seconds is not None
                and evidence.basin_recovery_seconds is not None
                and evidence.proposal.time_seconds
                > region.left.time_seconds
                and evidence.basin_start_seconds
                <= reference_time_seconds
                <= evidence.basin_recovery_seconds
            )
        )

        if len(later_reference_basins) == 1:
            outgoing = (
                later_reference_basins[0]
                .basin_recovery_seconds
            )

        incoming = region.right.time_seconds

        if outgoing >= incoming:
            return None

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=outgoing,
            incoming_start_seconds=incoming,
        )

    @classmethod
    def propose_asymmetric_outer_extent(
        cls,
        *,
        asymmetric_region,
        symmetric_regions,
        hypotheses,
    ):
        """Propose the exterior extent around one asymmetric anchor.

        The asymmetric primary remains the ownership anchor.  A preceding
        supported edge may extend the left side only when all intervening
        hypotheses are structural-only.  A symmetric region beginning at the
        anchor may provide the exterior right edge.
        """
        anchor = asymmetric_region.primary

        ordered = tuple(
            sorted(
                hypotheses,
                key=lambda hypothesis: hypothesis.time_seconds,
            )
        )

        previous_supported = tuple(
            hypothesis
            for hypothesis in ordered
            if (
                hypothesis.time_seconds < anchor.time_seconds
                and cls._is_supported_edge(hypothesis)
            )
        )

        if not previous_supported:
            return None

        predecessor = max(
            previous_supported,
            key=lambda hypothesis: hypothesis.time_seconds,
        )

        interior = tuple(
            hypothesis
            for hypothesis in ordered
            if (
                predecessor.time_seconds
                < hypothesis.time_seconds
                < anchor.time_seconds
            )
        )

        if not all(
            cls._is_structural_only(hypothesis)
            for hypothesis in interior
        ):
            return None

        matching_symmetric = tuple(
            region
            for region in symmetric_regions
            if region.left.time_seconds == anchor.time_seconds
        )

        if len(matching_symmetric) != 1:
            return None

        outer = matching_symmetric[0]

        outgoing = predecessor.time_seconds
        incoming = outer.right.time_seconds

        if outgoing >= incoming:
            return None

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=outgoing,
            incoming_start_seconds=incoming,
        )

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        """Resolve the complete bounded-insert temporal geometry."""
        symmetric = observation.routing.bounded_insert_regions
        asymmetric = (
            observation.routing.bounded_insert_asymmetric_regions
        )
        ownership = observation.bounded_insert_ownership

        if ownership is BoundedInsertOwnership.SYMMETRIC:
            if len(symmetric) != 1:
                return TransitionSpecialistResult(
                    temporal=None,
                )

            region = symmetric[0]

            proposal = self.propose_structural_bridge_cut(
                region,
            )

            if proposal is not None:
                return TransitionSpecialistResult(
                    temporal=proposal,
                )

            if observation.routing.fade.rise is None:
                proposal = self.propose_symmetric_outer_extent(
                    region=region,
                    hypotheses=observation.routing.hypotheses,
                )

                if proposal is not None:
                    return TransitionSpecialistResult(
                        temporal=proposal,
                    )

            proposal = self.propose_symmetric_phase_aware(
                region,
                reference_time_seconds=(
                    observation.routing.reference_time_seconds
                ),
            )

            if proposal is not None:
                return TransitionSpecialistResult(
                    temporal=proposal,
                )

            return self._resolve_symmetric_fallback(
                region
            )

        if ownership is BoundedInsertOwnership.ASYMMETRIC:
            if len(asymmetric) != 1:
                return TransitionSpecialistResult(
                    temporal=None,
                )

            proposal = None

            if observation.routing.fade.rise is None:
                proposal = self.propose_asymmetric_outer_extent(
                    asymmetric_region=asymmetric[0],
                    symmetric_regions=symmetric,
                    hypotheses=observation.routing.hypotheses,
                )

            if proposal is not None:
                return TransitionSpecialistResult(
                    temporal=proposal,
                )

            return self._resolve_asymmetric_fallback(
                asymmetric[0],
                fade=observation.routing.fade,
            )

        # Compatibility path for direct specialist use without
        # Router provenance.  Preserve the pre-Phase-2 behavior.
        if len(symmetric) == 1:
            return self._resolve_symmetric_fallback(
                symmetric[0]
            )

        if symmetric:
            return TransitionSpecialistResult(
                temporal=None,
            )

        if len(asymmetric) != 1:
            return TransitionSpecialistResult(
                temporal=None,
            )

        return self._resolve_asymmetric_fallback(
            asymmetric[0]
        )

    @staticmethod
    def _resolve_symmetric_fallback(
        region,
    ) -> TransitionSpecialistResult:
        outgoing = region.left.time_seconds
        incoming = region.right.time_seconds

        if outgoing >= incoming:
            return TransitionSpecialistResult(
                temporal=None,
            )

        return TransitionSpecialistResult(
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=outgoing,
                incoming_start_seconds=incoming,
            )
        )

    @staticmethod
    def _resolve_asymmetric_fallback(
        region,
        *,
        fade=None,
    ) -> TransitionSpecialistResult:
        primary_geometry = region.primary_edge.basin_geometry

        if primary_geometry is None:
            return TransitionSpecialistResult(
                temporal=None,
            )

        outgoing = primary_geometry.recovery_seconds

        from brynse.boundaries import BoundaryProposalSource

        basin_proposal_times = tuple(
            proposal.time_seconds
            for proposal in region.primary.proposals
            if proposal.source is BoundaryProposalSource.BASIN
        )

        structural_proposals = tuple(
            proposal
            for proposal in region.primary.proposals
            if (
                proposal.source
                is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )

        if (
            len(basin_proposal_times) == 1
            and structural_proposals
        ):
            strongest_structural = max(
                structural_proposals,
                key=lambda proposal: proposal.strength,
            )
            basin_time = basin_proposal_times[0]

            if (
                strongest_structural.time_seconds
                < region.primary.time_seconds
                < basin_time
                < primary_geometry.recovery_seconds
            ):
                outgoing = basin_time

        # Phase 4f:
        # A single structural proposal immediately preceding the
        # primary basin can describe the true outer-left extent of
        # an asymmetric insert when independent incoming-phase
        # evidence becomes established between that structural
        # onset and the basin entry.
        #
        # This refines temporal geometry only.  Router ownership
        # remains unchanged, and direct/legacy specialist use
        # receives fade=None and therefore preserves its existing
        # fallback behavior.
        if (
            fade is not None
            and fade.rise is None
            and len(structural_proposals) == 1
        ):
            structural = structural_proposals[0]

            if structural.time_seconds < primary_geometry.start_seconds:
                incoming_phase_reaches_basin = any(
                    (
                        structural.time_seconds
                        <= incoming.established_time_seconds
                        <= primary_geometry.start_seconds
                    )
                    for incoming in fade.incoming_fades
                )

                if incoming_phase_reaches_basin:
                    outgoing = structural.time_seconds

        complementary_geometry = (
            region.complementary_edge.basin_geometry
        )

        incoming = (
            region.complementary.time_seconds
            if complementary_geometry is None
            else complementary_geometry.recovery_seconds
        )

        if outgoing >= incoming:
            return TransitionSpecialistResult(
                temporal=None,
            )

        return TransitionSpecialistResult(
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=outgoing,
                incoming_start_seconds=incoming,
            )
        )


class LegacyAmbiguousSpecialist:
    """Preserve the current safe resolver while ownership is ambiguous.

    AMBIGUOUS is deliberately not another acoustic specialist.  It is a
    compatibility fallback used while the router cannot assign ownership.
    """

    def __init__(
        self,
        *,
        temporal_resolver,
        incoming_pipeline=None,
    ) -> None:
        self._temporal_resolver = temporal_resolver
        self._incoming_pipeline = incoming_pipeline

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        decision = self._temporal_resolver.resolve(
            selected=observation.selected,
            profile=observation.profile,
            window=observation.window,
            semantic_time_seconds=(
                observation.routing.reference_time_seconds
            ),
        )

        if decision is None:
            return TransitionSpecialistResult(
                temporal=None,
            )

        if self._incoming_pipeline is None:
            return TransitionSpecialistResult(
                temporal=decision,
            )

        refined = self._incoming_pipeline.refine(
            decision=decision,
            window=observation.window,
            pcm8=observation.pcm8,
            pcm16=observation.pcm16,
            hypotheses=observation.routing.hypotheses,
        )

        refined = self._refine_preceding_multistructural_basin(
            decision=refined,
            observation=observation,
        )

        return TransitionSpecialistResult(
            temporal=refined,
        )

    @staticmethod
    def _refine_preceding_multistructural_basin(
        *,
        decision: TemporalSplitDecision,
        observation: TransitionSpecialistInput,
    ) -> TemporalSplitDecision:
        if decision.kind is not TemporalSplitKind.EXCLUSION:
            return decision

        reference = observation.routing.reference_time_seconds
        outgoing = decision.outgoing_end_seconds
        incoming = decision.incoming_start_seconds

        candidates = []

        for hypothesis in observation.routing.hypotheses:
            geometry = getattr(
                hypothesis,
                "basin_geometry",
                None,
            )

            if geometry is None:
                continue

            if not (
                geometry.recovery_seconds
                < outgoing
                < reference
            ):
                continue

            if geometry.recovery_seconds >= incoming:
                continue

            proposals = getattr(
                hypothesis,
                "proposals",
                (),
            )

            basin_proposals = tuple(
                proposal
                for proposal in proposals
                if proposal.source
                is BoundaryProposalSource.BASIN
            )

            if len(basin_proposals) != 1:
                continue

            structural_times = tuple(
                proposal.time_seconds
                for proposal in proposals
                if proposal.source
                is BoundaryProposalSource.STRUCTURAL
            )

            has_pre_minimum_structure = any(
                time_seconds < geometry.minimum_seconds
                for time_seconds in structural_times
            )
            has_post_minimum_structure = any(
                geometry.minimum_seconds
                < time_seconds
                < geometry.recovery_seconds
                for time_seconds in structural_times
            )

            if not (
                has_pre_minimum_structure
                and has_post_minimum_structure
            ):
                continue

            candidates.append(geometry)

        if len(candidates) != 1:
            return decision

        geometry = candidates[0]

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=geometry.recovery_seconds,
            incoming_start_seconds=incoming,
        )


class RoutedTransitionResolver:
    """Route one neutral observation to exactly one temporal specialist."""

    def __init__(
        self,
        *,
        router: TransitionRouter,
        hard_cut,
        fade,
        interstitial,
        bounded_insert,
        ambiguous,
    ) -> None:
        self._router = router
        self._specialists = {
            TransitionKind.HARD_CUT: hard_cut,
            TransitionKind.FADE: fade,
            TransitionKind.INTERSTITIAL: interstitial,
            TransitionKind.BOUNDED_INSERT: bounded_insert,
            TransitionKind.AMBIGUOUS: ambiguous,
        }

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> tuple[TransitionKind, TransitionSpecialistResult]:
        """Classify once, then delegate to exactly one specialist."""

        kind = self._router.classify(
            context=observation.routing,
        )

        routed_observation = observation

        if kind is TransitionKind.BOUNDED_INSERT:
            ownership_resolver = getattr(
                self._router,
                "bounded_insert_ownership",
                None,
            )

            if ownership_resolver is not None:
                ownership = ownership_resolver(
                    context=observation.routing,
                )

                routed_observation = replace(
                    observation,
                    bounded_insert_ownership=ownership,
                )

        specialist = self._specialists[kind]

        result = specialist.resolve(
            observation=routed_observation,
        )

        return kind, result
