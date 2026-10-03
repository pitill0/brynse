from __future__ import annotations

from dataclasses import dataclass

from brynse.boundaries import BoundaryConfidence, BoundaryHypothesis
from brynse.models import TemporalSplitDecision, TemporalSplitKind
from brynse.transition_evidence import (
    IncomingFadeEvidence,
    OutgoingFadeEvidence,
)
from brynse.transition_specialists import TransitionSpecialistResult


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


@dataclass(frozen=True)
class FadeSpecialistDiagnostics:
    """Fade-specific evidence selected while resolving one transition."""

    outgoing_selection: OutgoingFadeSelection | None
    incoming_selection: IncomingFadeSelection | None


class FadeSpecialist:
    """Understand fade transition geometry and resolve fade boundaries."""

    _confidence_rank = {
        BoundaryConfidence.HIGH: 0,
        BoundaryConfidence.MEDIUM: 1,
        BoundaryConfidence.LOW: 2,
        BoundaryConfidence.UNRESOLVED: 3,
    }

    @classmethod
    def resolve(
        cls,
        *,
        observation,
    ) -> TransitionSpecialistResult:
        """Resolve one fade transition from already observed fade evidence."""

        boundary = observation.selected.time_seconds
        reference = observation.routing.reference_time_seconds
        fade_evidence = observation.routing.fade

        outgoing_selection = cls.select_outgoing(
            reference_time_seconds=reference,
            outgoing_fades=fade_evidence.outgoing_fades,
            hypotheses=observation.routing.hypotheses,
        )
        incoming_selection = cls.select_incoming(
            reference_time_seconds=reference,
            incoming_fades=fade_evidence.incoming_fades,
            hypotheses=observation.routing.hypotheses,
        )

        diagnostics = FadeSpecialistDiagnostics(
            outgoing_selection=outgoing_selection,
            incoming_selection=incoming_selection,
        )

        outgoing = fade_evidence.rise

        if outgoing is None:
            return TransitionSpecialistResult(
                temporal=TemporalSplitDecision(
                    kind=TemporalSplitKind.HARD_CUT,
                    incoming_start_seconds=boundary,
                    outgoing_end_seconds=boundary,
                ),
                diagnostics=diagnostics,
            )

        if boundary >= outgoing.time_seconds:
            return TransitionSpecialistResult(
                temporal=None,
                diagnostics=diagnostics,
            )

        incoming_lead = reference - boundary
        outgoing_delay = outgoing.time_seconds - reference

        if (
            incoming_lead >= 0.0
            and outgoing_delay > incoming_lead
        ):
            return TransitionSpecialistResult(
                temporal=TemporalSplitDecision(
                    kind=TemporalSplitKind.HARD_CUT,
                    incoming_start_seconds=boundary,
                    outgoing_end_seconds=boundary,
                ),
                diagnostics=diagnostics,
            )

        # When both sides of the fade have independently selected boundary
        # support, the specialist owns their temporal geometry.  Do not
        # reproduce the old unilateral outgoing clamp: only replace the
        # conservative temporal pair when the two specialist-supported
        # boundaries form a valid overlap.
        if (
            outgoing_selection is not None
            and incoming_selection is not None
        ):
            specialist_incoming = (
                incoming_selection.hypothesis.time_seconds
            )
            specialist_outgoing = (
                outgoing_selection.hypothesis.time_seconds
            )

            if specialist_incoming < specialist_outgoing:
                return TransitionSpecialistResult(
                    temporal=TemporalSplitDecision(
                        kind=TemporalSplitKind.CROSSFADE,
                        incoming_start_seconds=specialist_incoming,
                        outgoing_end_seconds=specialist_outgoing,
                    ),
                    diagnostics=diagnostics,
                )

        return TransitionSpecialistResult(
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.CROSSFADE,
                incoming_start_seconds=boundary,
                outgoing_end_seconds=outgoing.time_seconds,
            ),
            diagnostics=diagnostics,
        )

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
        """Select outgoing fade geometry with the best boundary support."""

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
            for fade in eligible_fades
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
