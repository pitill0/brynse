"""Temporal specialists for routed acoustic transitions.

The router decides WHO owns a transition.
A specialist decides WHERE its temporal boundaries are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from brynse.boundaries import BoundaryConfidence
from brynse.models import TemporalSplitDecision, TemporalSplitKind
from brynse.transition_router import (
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

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        symmetric = observation.routing.bounded_insert_regions

        if len(symmetric) == 1:
            region = symmetric[0]
            outgoing = region.left.time_seconds
            incoming = region.right.time_seconds

            if outgoing >= incoming:
                return TransitionSpecialistResult(temporal=None)

            return TransitionSpecialistResult(
                temporal=TemporalSplitDecision(
                    kind=TemporalSplitKind.EXCLUSION,
                    outgoing_end_seconds=outgoing,
                    incoming_start_seconds=incoming,
                )
            )

        if symmetric:
            return TransitionSpecialistResult(temporal=None)

        asymmetric = (
            observation.routing.bounded_insert_asymmetric_regions
        )

        if len(asymmetric) != 1:
            return TransitionSpecialistResult(temporal=None)

        region = asymmetric[0]

        primary_geometry = region.primary_edge.basin_geometry

        if primary_geometry is None:
            return TransitionSpecialistResult(temporal=None)

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
            return TransitionSpecialistResult(temporal=None)

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

        specialist = self._specialists[kind]

        result = specialist.resolve(
            observation=observation,
        )

        return kind, result
