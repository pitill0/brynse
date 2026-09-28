"""Source-agnostic operational boundary providers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from brynse.models import BoundaryCandidate


class BoundaryProvider(Protocol):
    """Produce logical boundary candidates for one stream-time interval."""

    def propose(
        self,
        *,
        start_time_seconds: float,
        end_time_seconds: float,
    ) -> tuple[BoundaryCandidate, ...]:
        """Return candidates in the half-open interval (start, end]."""


class FixedIntervalBoundaryProvider:
    """Propose deterministic boundaries at a fixed interval from stream time zero."""

    def __init__(self, *, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")
        self._interval_seconds = interval_seconds

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    def propose(
        self,
        *,
        start_time_seconds: float,
        end_time_seconds: float,
    ) -> tuple[BoundaryCandidate, ...]:
        if start_time_seconds < 0:
            raise ValueError("start_time_seconds must be non-negative")
        if end_time_seconds < start_time_seconds:
            raise ValueError("end_time_seconds must be greater than or equal to start_time_seconds")

        first_index = math.floor(start_time_seconds / self._interval_seconds) + 1
        last_index = math.floor(end_time_seconds / self._interval_seconds)

        if last_index < first_index:
            return ()

        return tuple(
            BoundaryCandidate(
                time_seconds=index * self._interval_seconds,
                source="fixed_interval",
            )
            for index in range(first_index, last_index + 1)
        )


class ManualBoundaryProvider:
    """Propose explicitly supplied boundary times."""

    def __init__(self, boundary_times_seconds: tuple[float, ...]) -> None:
        if any(value < 0 for value in boundary_times_seconds):
            raise ValueError("boundary times must be non-negative")

        self._boundary_times_seconds = tuple(sorted(set(boundary_times_seconds)))

    def propose(
        self,
        *,
        start_time_seconds: float,
        end_time_seconds: float,
    ) -> tuple[BoundaryCandidate, ...]:
        if start_time_seconds < 0:
            raise ValueError("start_time_seconds must be non-negative")
        if end_time_seconds < start_time_seconds:
            raise ValueError("end_time_seconds must be greater than or equal to start_time_seconds")

        return tuple(
            BoundaryCandidate(
                time_seconds=time_seconds,
                source="manual",
            )
            for time_seconds in self._boundary_times_seconds
            if start_time_seconds < time_seconds <= end_time_seconds
        )


@dataclass(frozen=True)
class ExternalBoundary:
    """Boundary supplied by an external producer."""

    time_seconds: float
    source: str = "external"
    reference_offset: int | None = None

    def __post_init__(self) -> None:
        if self.time_seconds < 0:
            raise ValueError("time_seconds must be non-negative")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.reference_offset is not None and self.reference_offset < 0:
            raise ValueError("reference_offset must be non-negative")


class ExternalBoundaryProvider:
    """Expose externally supplied boundaries through the generic provider contract."""

    def __init__(self, boundaries: tuple[ExternalBoundary, ...]) -> None:
        self._boundaries = tuple(
            sorted(
                boundaries,
                key=lambda boundary: (
                    boundary.time_seconds,
                    boundary.source,
                    boundary.reference_offset if boundary.reference_offset is not None else -1,
                ),
            )
        )

    def propose(
        self,
        *,
        start_time_seconds: float,
        end_time_seconds: float,
    ) -> tuple[BoundaryCandidate, ...]:
        if start_time_seconds < 0:
            raise ValueError("start_time_seconds must be non-negative")
        if end_time_seconds < start_time_seconds:
            raise ValueError("end_time_seconds must be greater than or equal to start_time_seconds")

        return tuple(
            BoundaryCandidate(
                time_seconds=boundary.time_seconds,
                source=boundary.source,
                reference_offset=boundary.reference_offset,
            )
            for boundary in self._boundaries
            if start_time_seconds < boundary.time_seconds <= end_time_seconds
        )
