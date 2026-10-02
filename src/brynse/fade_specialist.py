from __future__ import annotations

from dataclasses import dataclass

from brynse.boundaries import BoundaryConfidence, BoundaryHypothesis
from brynse.transition_evidence import (
    IncomingFadeEvidence,
    OutgoingFadeEvidence,
)


@dataclass(frozen=True)
class OutgoingFadeSelection:
    """Descriptive association between an outgoing fade and boundary evidence."""

    fade: OutgoingFadeEvidence
    hypothesis: BoundaryHypothesis
    support_distance_seconds: float


@dataclass(frozen=True)
class IncomingFadeSelection:
    """Descriptive association between an incoming fade and boundary evidence."""

    fade: IncomingFadeEvidence
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
    def select_incoming(
        cls,
        *,
        reference_time_seconds: float,
        incoming_fades: tuple[IncomingFadeEvidence, ...],
        hypotheses: tuple[BoundaryHypothesis, ...],
    ) -> IncomingFadeSelection | None:
        """Select incoming geometry with coherent boundary support."""

        if reference_time_seconds < 0:
            raise ValueError("reference_time_seconds must be non-negative")

        if not incoming_fades or not hypotheses:
            return None

        candidates = []

        for fade in incoming_fades:
            # A region already fully established before the transition cannot
            # describe the incoming side of this transition.
            if fade.established_time_seconds < reference_time_seconds:
                continue

            # If the transition precedes the region, its onset is the natural
            # incoming anchor. If the transition falls inside the region,
            # establishment is the meaningful later edge.
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
                    cls._confidence_rank[item.confidence],
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
                    cls._confidence_rank[hypothesis.confidence],
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

        return IncomingFadeSelection(
            fade=fade,
            hypothesis=hypothesis,
            support_distance_seconds=support_distance,
        )

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
