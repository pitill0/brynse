"""Routing contract for selecting the specialist that owns a transition."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from brynse.transition_evidence import TransitionEvidenceState


class TransitionKind(StrEnum):
    """Acoustic transition classes understood by boundary specialists."""

    HARD_CUT = "hard_cut"
    FADE = "fade"
    INTERSTITIAL = "interstitial"
    AMBIGUOUS = "ambiguous"


class TransitionRouter(Protocol):
    """Classify transition context without choosing temporal boundaries."""

    def classify(
        self,
        *,
        evidence: TransitionEvidenceState,
    ) -> TransitionKind:
        """Return which specialist owns the transition."""
        ...
