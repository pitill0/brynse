"""Temporal specialists for routed acoustic transitions.

The router decides WHO owns a transition.
A specialist decides WHERE its temporal boundaries are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

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

    The existing IncomingBoundaryPipeline already contains the acoustic
    understanding needed for the incoming side.  This specialist owns that
    behaviour instead of leaving it as a global post-resolution refinement.
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

        return TransitionSpecialistResult(
            temporal=self._incoming_pipeline.refine(
                decision=initial,
                window=observation.window,
                pcm8=observation.pcm8,
                pcm16=observation.pcm16,
                hypotheses=observation.routing.hypotheses,
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
        ambiguous,
    ) -> None:
        self._router = router
        self._specialists = {
            TransitionKind.HARD_CUT: hard_cut,
            TransitionKind.FADE: fade,
            TransitionKind.INTERSTITIAL: interstitial,
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
