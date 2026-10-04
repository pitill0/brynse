"""Temporal specialists for routed acoustic transitions.

The router decides WHO owns a transition.
A specialist decides WHERE its temporal boundaries are.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from brynse.boundaries import BoundaryConfidence
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
    """Resolve a genuine hard cut at the selected acoustic boundary."""

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        boundary = observation.selected.time_seconds

        return TransitionSpecialistResult(
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                incoming_start_seconds=boundary,
                outgoing_end_seconds=boundary,
            )
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
            return TransitionSpecialistResult(
                temporal=refined,
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

    @staticmethod
    def propose_symmetric_phase_aware(region):
        """Propose a phase-aware symmetric exclusion without applying it."""
        outgoing = region.left.time_seconds

        geometry = region.left_edge.basin_geometry

        if geometry is not None:
            outgoing = min(
                outgoing,
                geometry.recovery_seconds,
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

            proposal = self.propose_symmetric_phase_aware(
                region
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
                asymmetric[0]
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
    ) -> TransitionSpecialistResult:
        primary_geometry = region.primary_edge.basin_geometry

        if primary_geometry is None:
            return TransitionSpecialistResult(
                temporal=None,
            )

        outgoing = primary_geometry.recovery_seconds

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

        return TransitionSpecialistResult(
            temporal=self._incoming_pipeline.refine(
                decision=decision,
                window=observation.window,
                pcm8=observation.pcm8,
                pcm16=observation.pcm16,
                hypotheses=observation.routing.hypotheses,
            )
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
