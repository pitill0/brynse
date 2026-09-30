"""Radio integration for promoted autonomous segment boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from brynse.models import Segment
from brynse.orchestrator import CandidateResolution
from brynse.integrations.radio.session import SegmentTransition


def untracked_segment_label(time_seconds: float) -> str:
    """Return a deterministic provisional identity for an unknown segment."""
    milliseconds = round(time_seconds * 1000)
    return f"untracked_{milliseconds:012d}"


@dataclass
class AutonomousSegmentState:
    """Track the currently open logical segment independently of radio identity."""

    current: Segment | None = None

    def observe_known_segment(self, segment: Segment) -> None:
        """Seed segment state when no autonomous segment currently owns it."""
        if self.current is None:
            self.current = segment

    def observe_materialized_segment(self, segment: Segment) -> None:
        """Synchronize state with the segment actually materialized by output."""
        self.current = segment

    def transition(
        self,
        resolution: CandidateResolution,
        *,
        minimum_open_seconds: float = 0.0,
    ) -> SegmentTransition | None:
        """Turn one promoted generic boundary into a logical segment transition."""
        if minimum_open_seconds < 0:
            raise ValueError("minimum_open_seconds must be non-negative")

        outgoing = self.current
        if outgoing is None:
            return None

        boundary_time = resolution.temporal.incoming_start_seconds
        if boundary_time < outgoing.start_time_seconds + minimum_open_seconds:
            return None

        split = resolution.split
        if split.incoming_start is None:
            return None

        incoming = Segment(
            start_offset=split.incoming_start,
            start_time_seconds=resolution.temporal.incoming_start_seconds,
            label=untracked_segment_label(
                resolution.temporal.incoming_start_seconds,
            ),
        )

        transition = SegmentTransition(
            outgoing=outgoing,
            incoming=incoming,
            boundary=resolution,
        )
        self.current = incoming
        return transition
