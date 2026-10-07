from __future__ import annotations

from dataclasses import dataclass

from brynse.boundaries import BoundaryHypothesis
from brynse.models import TemporalSplitDecision, TemporalSplitKind
from brynse.transition_evidence import (
    IncomingFadeEvidence,
    OutgoingFadeEvidence,
)
from brynse.transition_router import (
    DefaultTransitionRouter,
    IncomingFadeRoutingAssociation,
    OutgoingFadeRoutingAssociation,
)
from brynse.transition_specialists import TransitionSpecialistResult

OutgoingFadeSelection = OutgoingFadeRoutingAssociation
IncomingFadeSelection = IncomingFadeRoutingAssociation


@dataclass(frozen=True)
class FadeSpecialistDiagnostics:
    """Fade-specific evidence selected while resolving one transition."""

    outgoing_selection: OutgoingFadeSelection | None
    incoming_selection: IncomingFadeSelection | None


class FadeSpecialist:
    """Understand fade transition geometry and resolve fade boundaries."""

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

        fade_assessment = getattr(
            observation,
            "fade_assessment",
            None,
        )

        if fade_assessment is not None:
            outgoing_selection = fade_assessment.outgoing_association
            incoming_selection = fade_assessment.incoming_association
        else:
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
        """Compatibility wrapper for Router-owned incoming association."""
        return DefaultTransitionRouter.associate_incoming_fade(
            reference_time_seconds=reference_time_seconds,
            incoming_fades=incoming_fades,
            hypotheses=hypotheses,
        )

    @classmethod
    def select_outgoing(
        cls,
        *,
        reference_time_seconds: float,
        outgoing_fades: tuple[OutgoingFadeEvidence, ...],
        hypotheses: tuple[BoundaryHypothesis, ...],
    ) -> OutgoingFadeSelection | None:
        """Compatibility wrapper for Router-owned outgoing association."""
        return DefaultTransitionRouter.associate_outgoing_fade(
            reference_time_seconds=reference_time_seconds,
            outgoing_fades=outgoing_fades,
            hypotheses=hypotheses,
        )
