from __future__ import annotations

from dataclasses import dataclass

from brynse.boundaries import BoundaryConfidence, BoundaryHypothesis
from brynse.transition_evidence import OutgoingFadeEvidence


@dataclass(frozen=True)
class OutgoingFadeSelection:
    """Descriptive association between an outgoing fade and boundary evidence."""

    fade: OutgoingFadeEvidence
    hypothesis: BoundaryHypothesis
    support_distance_seconds: float


class FadeSpecialist:
    """Understand fade transition geometry without applying split decisions."""

    _confidence_rank = {
        BoundaryConfidence.HIGH: 0,
        BoundaryConfidence.MEDIUM: 1,
        BoundaryConfidence.LOW: 2,
        BoundaryConfidence.UNRESOLVED: 3,
    }

    @classmethod
    def select_outgoing(
        cls,
        *,
        outgoing_fades: tuple[OutgoingFadeEvidence, ...],
        hypotheses: tuple[BoundaryHypothesis, ...],
    ) -> OutgoingFadeSelection | None:
        """Associate an outgoing fade floor with its nearest boundary hypothesis."""

        if not outgoing_fades or not hypotheses:
            return None

        pairs = [
            (
                abs(
                    fade.floor_time_seconds
                    - hypothesis.time_seconds
                ),
                cls._confidence_rank[hypothesis.confidence],
                -fade.descent_consistency,
                -fade.descent_ratio,
                -fade.floor_time_seconds,
                fade,
                hypothesis,
            )
            for fade in outgoing_fades
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

        return OutgoingFadeSelection(
            fade=fade,
            hypothesis=hypothesis,
            support_distance_seconds=distance,
        )
