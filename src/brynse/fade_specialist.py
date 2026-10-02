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
        reference_time_seconds: float,
        outgoing_fades: tuple[OutgoingFadeEvidence, ...],
        hypotheses: tuple[BoundaryHypothesis, ...],
    ) -> OutgoingFadeSelection | None:
        """Select the strongest outgoing fade with boundary support."""

        if reference_time_seconds < 0:
            raise ValueError("reference_time_seconds must be non-negative")

        if not outgoing_fades or not hypotheses:
            return None

        eligible_fades = [
            fade
            for fade in outgoing_fades
            if fade.start_time_seconds <= reference_time_seconds
        ]
        if not eligible_fades:
            return None

        candidates = []

        for fade in eligible_fades:
            hypothesis = min(
                hypotheses,
                key=lambda item: (
                    abs(
                        fade.floor_time_seconds
                        - item.time_seconds
                    ),
                    cls._confidence_rank[item.confidence],
                    item.time_seconds,
                ),
            )
            distance = abs(
                fade.floor_time_seconds
                - hypothesis.time_seconds
            )

            candidates.append(
                (
                    -fade.descent_ratio,
                    -fade.descent_consistency,
                    distance,
                    cls._confidence_rank[hypothesis.confidence],
                    -fade.floor_time_seconds,
                    fade,
                    hypothesis,
                )
            )

        (
            _ratio,
            _consistency,
            distance,
            _confidence,
            _floor,
            fade,
            hypothesis,
        ) = min(candidates, key=lambda item: item[:5])

        return OutgoingFadeSelection(
            fade=fade,
            hypothesis=hypothesis,
            support_distance_seconds=distance,
        )
