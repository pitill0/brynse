"""Shadow-only assessment domain for composed transition evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fluxtuner_ripper.transition_evidence import TransitionEvidenceState


class TransitionAssessmentLabel(StrEnum):
    """Conservative shadow interpretation of a transition evidence state."""

    UNRESOLVED = "unresolved"
    REAL_LIKE = "real_like"
    INTERNAL_LIKE = "internal_like"


@dataclass(frozen=True)
class ShadowTransitionAssessment:
    """Non-operative interpretation of one composed transition evidence state."""

    evidence: TransitionEvidenceState
    label: TransitionAssessmentLabel = TransitionAssessmentLabel.UNRESOLVED


class ShadowTransitionEvaluator:
    """Return a conservative shadow assessment without applying a decision policy."""

    def assess(self, *, evidence: TransitionEvidenceState) -> ShadowTransitionAssessment:
        """Leave evidence unresolved until a validated policy is introduced."""
        return ShadowTransitionAssessment(evidence=evidence)
