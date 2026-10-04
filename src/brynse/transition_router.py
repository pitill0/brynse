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


class DefaultTransitionRouter:
    """Conservatively route transitions from descriptive acoustic evidence."""

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
