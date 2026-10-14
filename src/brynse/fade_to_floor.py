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
    """Outgoing fade whose floor belongs to one supported later handoff."""

    outgoing: OutgoingFadeEvidence
    incoming_handoff: BoundaryHypothesis

    def __post_init__(self) -> None:
        if (
            self.outgoing.floor_time_seconds
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

        # FadeToFloor describes a direct transition into one acoustically
        # supported destination. Multiple supported handoffs make ownership
        # ambiguous, so the Observer must abstain rather than choose one.
        if len(supported) != 1:
            return ()

        candidates: list[FadeToFloorEvidence] = []

        for outgoing in fade.outgoing_fades:
            # A later observed rise means ordinary fade geometry has more
            # specific evidence available. FadeToFloor describes the case
            # where the outgoing envelope simply reaches its floor.
            if outgoing.rise_time_seconds is not None:
                continue

            floor = outgoing.floor_time_seconds

            if floor > reference_time_seconds:
                continue

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

            if len(owners) != 1:
                continue

            candidates.append(
                FadeToFloorEvidence(
                    outgoing=outgoing,
                    incoming_handoff=owners[0],
                )
            )

        if len(candidates) != 1:
            return ()

        return (candidates[0],)
