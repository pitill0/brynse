"""Source-agnostic operational boundary providers."""

from __future__ import annotations

import math
from typing import Protocol

from fluxtuner_ripper.models import BoundaryCandidate


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
