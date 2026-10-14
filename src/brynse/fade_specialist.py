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
from brynse.transition_specialists import (
    TransitionSpecialistResult,
    _verify_owned_temporal_decision,
)

OutgoingFadeSelection = OutgoingFadeRoutingAssociation
IncomingFadeSelection = IncomingFadeRoutingAssociation


@dataclass(frozen=True)
class FadeSpecialistDiagnostics:
    """Fade-specific evidence selected while resolving one transition."""

    outgoing_selection: OutgoingFadeSelection | None
    incoming_selection: IncomingFadeSelection | None


class FadeSpecialist:
    """Understand fade transition geometry and resolve fade boundaries."""

    def __init__(
        self,
        *,
        temporal_edge_verifier=None,
    ) -> None:
        self._temporal_edge_verifier = temporal_edge_verifier

    def resolve(
        self=None,
        *,
        observation,
    ) -> TransitionSpecialistResult:
        """Resolve one fade transition, then verify owned edges against PCM."""

        cls = (
            type(self)
            if isinstance(self, FadeSpecialist)
            else FadeSpecialist
        )

        result = cls._resolve_geometry(
            observation=observation,
        )

        verifier = (
            self._temporal_edge_verifier
            if isinstance(self, FadeSpecialist)
            else None
        )

        if result.temporal is None or verifier is None:
            return result

        outgoing_extra = []
        incoming_extra = []

        diagnostics = result.diagnostics

        if diagnostics is not None:
            outgoing_selection = diagnostics.outgoing_selection
            incoming_selection = diagnostics.incoming_selection

            if outgoing_selection is not None:
                fade = outgoing_selection.fade

                outgoing_extra.extend(
                    (
                        fade.start_time_seconds,
                        fade.floor_time_seconds,
                        outgoing_selection.hypothesis.time_seconds,
                    )
                )

            if incoming_selection is not None:
                fade = incoming_selection.fade

                incoming_extra.extend(
                    (
                        fade.start_time_seconds,
                        fade.established_time_seconds,
                        incoming_selection.hypothesis.time_seconds,
                    )
                )

        fade_to_floor = (
            None
            if observation.semantic_roles is None
            else observation.semantic_roles.fade_to_floor
        )

        if (
            fade_to_floor is not None
            and fade_to_floor.incoming_handoff is None
        ):
            outgoing_extra.extend(
                (
                    fade_to_floor.outgoing.start_time_seconds,
                    fade_to_floor.outgoing.floor_time_seconds,
                )
            )
            incoming_extra.append(
                observation.routing.reference_time_seconds
            )

        if not outgoing_extra and not incoming_extra:
            shared = (
                observation.selected.time_seconds,
                observation.routing.reference_time_seconds,
            )
            outgoing_extra.extend(shared)
            incoming_extra.extend(shared)

        outgoing_only_fade_to_floor = (
            fade_to_floor is not None
            and fade_to_floor.incoming_handoff is None
        )

        verified = _verify_owned_temporal_decision(
            verifier=verifier,
            observation=observation,
            decision=result.temporal,
            outgoing_extra_times=outgoing_extra,
            incoming_extra_times=incoming_extra,
            verify_outgoing=True,
            verify_incoming=not outgoing_only_fade_to_floor,
        )

        if verified is result.temporal:
            return result

        return TransitionSpecialistResult(
            temporal=verified,
            diagnostics=result.diagnostics,
        )

    @classmethod
    def _resolve_geometry(
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

        semantic_roles = getattr(
            observation,
            "semantic_roles",
            None,
        )

        fade_to_floor = (
            None
            if semantic_roles is None
            else getattr(
                semantic_roles,
                "fade_to_floor",
                None,
            )
        )

        if fade_to_floor is not None:
            outgoing_fade = fade_to_floor.outgoing
            incoming_handoff = fade_to_floor.incoming_handoff

            specialist_outgoing = (
                outgoing_fade.floor_time_seconds
            )
            specialist_incoming = (
                reference
                if incoming_handoff is None
                else incoming_handoff.time_seconds
            )

            if specialist_outgoing < specialist_incoming:
                if incoming_handoff is not None:
                    diagnostics = FadeSpecialistDiagnostics(
                        outgoing_selection=OutgoingFadeSelection(
                            fade=outgoing_fade,
                            hypothesis=incoming_handoff,
                            support_distance_seconds=abs(
                                specialist_outgoing
                                - specialist_incoming
                            ),
                        ),
                        incoming_selection=None,
                    )

                return TransitionSpecialistResult(
                    temporal=TemporalSplitDecision(
                        kind=TemporalSplitKind.EXCLUSION,
                        outgoing_end_seconds=(
                            specialist_outgoing
                        ),
                        incoming_start_seconds=(
                            specialist_incoming
                        ),
                    ),
                    diagnostics=diagnostics,
                )

        if (
            fade_assessment is not None
            and outgoing_selection is not None
            and incoming_selection is not None
        ):
            reconciled_geometries = tuple(
                geometry
                for geometry in fade_evidence.geometries
                if (
                    geometry.outgoing is outgoing_selection.fade
                    and geometry.incoming is incoming_selection.fade
                )
            )

            if len(reconciled_geometries) == 1:
                geometry = reconciled_geometries[0]
                specialist_outgoing = (
                    geometry.outgoing.floor_time_seconds
                )
                specialist_incoming = (
                    geometry.incoming.start_time_seconds
                )

                if specialist_incoming == specialist_outgoing:
                    kind = TemporalSplitKind.HARD_CUT
                elif specialist_incoming < specialist_outgoing:
                    kind = TemporalSplitKind.CROSSFADE
                else:
                    kind = TemporalSplitKind.EXCLUSION

                return TransitionSpecialistResult(
                    temporal=TemporalSplitDecision(
                        kind=kind,
                        incoming_start_seconds=specialist_incoming,
                        outgoing_end_seconds=specialist_outgoing,
                    ),
                    diagnostics=diagnostics,
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
