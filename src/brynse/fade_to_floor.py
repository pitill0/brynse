"""Descriptive evidence for an outgoing fade reaching a supported floor."""

from __future__ import annotations

from dataclasses import dataclass

from brynse.boundaries import (
    BoundaryHypothesis,
    BoundaryProposalSource,
)
from brynse.transition_evidence import (
    FadeRoutingEvidence,
    OutgoingFadeEvidence,
)


@dataclass(frozen=True)
class FadeToFloorEvidence:
    """Terminal outgoing fade, optionally paired with a direct handoff."""

    outgoing: OutgoingFadeEvidence
    incoming_handoff: BoundaryHypothesis | None = None

    def __post_init__(self) -> None:
        if (
            self.incoming_handoff is not None
            and self.outgoing.floor_time_seconds
            > self.incoming_handoff.time_seconds
        ):
            raise ValueError(
                "fade-to-floor outgoing floor must not follow "
                "incoming handoff"
            )


class FadeToFloorDetector:
    """Observe direct fade-to-floor topology without choosing final cuts."""

    @staticmethod
    def _is_supported_handoff(
        hypothesis: BoundaryHypothesis,
    ) -> bool:
        sources = {
            proposal.source
            for proposal in hypothesis.proposals
        }

        return (
            hypothesis.basin_geometry is not None
            and BoundaryProposalSource.BASIN in sources
            and BoundaryProposalSource.STRUCTURAL in sources
        )

    def observe(
        self,
        *,
        reference_time_seconds: float,
        fade: FadeRoutingEvidence,
        hypotheses: tuple[BoundaryHypothesis, ...],
    ) -> tuple[FadeToFloorEvidence, ...]:
        """Return one conservative fade-to-floor observation, or none."""

        supported = tuple(
            hypothesis
            for hypothesis in hypotheses
            if self._is_supported_handoff(hypothesis)
        )

        observed = tuple(fade.outgoing_fades)

        if not observed:
            return ()

        deepest_rms = min(
            outgoing.floor_rms
            for outgoing in observed
        )
        deepest = tuple(
            outgoing
            for outgoing in observed
            if outgoing.floor_rms == deepest_rms
        )

        # Equal acoustic floors do not establish unique outgoing authority.
        if len(deepest) != 1:
            return ()

        outgoing = deepest[0]
        floor = outgoing.floor_time_seconds

        # FadeToFloor authority belongs only to the globally deepest outgoing
        # floor. If that deepest fade later rises, a shallower earlier fade
        # must not be promoted as an artificial terminal substitute.
        if outgoing.rise_time_seconds is not None:
            return ()

        # A deeper floor after the reference means the observed track had not
        # yet reached its terminal floor at the semantic boundary.
        if floor > reference_time_seconds:
            return ()

        owners = tuple(
            hypothesis
            for hypothesis in supported
            if (
                hypothesis.time_seconds >= floor
                and hypothesis.basin_geometry is not None
                and (
                    hypothesis.basin_geometry.start_seconds
                    <= floor
                    <= hypothesis.basin_geometry.recovery_seconds
                )
            )
        )

        if len(owners) == 1:
            return (
                FadeToFloorEvidence(
                    outgoing=outgoing,
                    incoming_handoff=owners[0],
                ),
            )

        # The outgoing edge may still be authoritative even when the next
        # semantic identity is not. Require independently observed later
        # acoustic activity before the reference so an isolated ordinary fade
        # does not manufacture a transition by itself.
        later_activity = any(
            candidate.start_time_seconds > floor
            and candidate.start_time_seconds <= reference_time_seconds
            for candidate in fade.outgoing_fades
        )

        if not later_activity:
            return ()

        return (
            FadeToFloorEvidence(
                outgoing=outgoing,
            ),
        )
