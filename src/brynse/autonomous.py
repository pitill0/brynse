"""Autonomous acoustic boundary detection."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from brynse.boundaries import BoundaryProposal, BoundaryProposalSource
from brynse.models import AcousticProfile


@dataclass(frozen=True)
class AutonomousBoundaryCandidate:
    """One high-confidence boundary inferred from audio alone."""

    time_seconds: float
    basin_start_seconds: float
    attack_seconds: float
    basin_rms: float
    attack_rms: float
    persistence_rms: float

    @property
    def transition_span_seconds(self) -> float:
        return self.attack_seconds - self.basin_start_seconds

    def as_proposal(self) -> BoundaryProposal:
        """Represent this audio-only candidate in the boundary domain model."""
        return BoundaryProposal(
            time_seconds=self.time_seconds,
            source=BoundaryProposalSource.BASIN,
        )


class AutonomousBoundaryDetector:
    """Detect strong basin-to-attack transitions without metadata hints."""

    def __init__(
        self,
        *,
        baseline_span_levels: int = 8,
        basin_span_levels: int = 2,
        attack_span_levels: int = 2,
        persistence_span_levels: int = 6,
        quiet_ratio: float = 0.30,
        baseline_stability_ratio: float = 0.50,
        min_jump_ratio: float = 3.0,
        min_persistence_ratio: float = 2.0,
        min_track_seconds: float = 30.0,
    ) -> None:
        if baseline_span_levels <= 0:
            raise ValueError("baseline_span_levels must be greater than zero")
        if basin_span_levels <= 0:
            raise ValueError("basin_span_levels must be greater than zero")
        if attack_span_levels <= 0:
            raise ValueError("attack_span_levels must be greater than zero")
        if persistence_span_levels <= 0:
            raise ValueError("persistence_span_levels must be greater than zero")
        if not 0 < quiet_ratio < 1:
            raise ValueError("quiet_ratio must be between zero and one")
        if not 0 < baseline_stability_ratio <= 1:
            raise ValueError("baseline_stability_ratio must be between zero and one")
        if min_jump_ratio <= 1:
            raise ValueError("min_jump_ratio must be greater than one")
        if min_persistence_ratio <= 1:
            raise ValueError("min_persistence_ratio must be greater than one")
        if min_track_seconds <= 0:
            raise ValueError("min_track_seconds must be greater than zero")

        self._baseline_span = baseline_span_levels
        self._basin_span = basin_span_levels
        self._attack_span = attack_span_levels
        self._persistence_span = persistence_span_levels
        self._quiet_ratio = quiet_ratio
        self._baseline_stability = baseline_stability_ratio
        self._min_jump = min_jump_ratio
        self._min_persistence = min_persistence_ratio
        self._min_track_seconds = min_track_seconds

    @staticmethod
    def _mean(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    def detect(
        self,
        *,
        profile: AcousticProfile,
        absolute_start_time_seconds: float = 0.0,
    ) -> tuple[AutonomousBoundaryCandidate, ...]:
        """Return conservative audio-only boundary candidates."""
        if absolute_start_time_seconds < 0:
            raise ValueError("absolute_start_time_seconds must be non-negative")

        levels = profile.levels
        minimum_before = self._baseline_span + self._basin_span
        minimum_after = max(self._attack_span, self._persistence_span)
        if len(levels) <= minimum_before + minimum_after:
            return ()

        candidates: list[AutonomousBoundaryCandidate] = []
        last_time: float | None = None

        for attack_index in range(
            minimum_before,
            len(levels) - minimum_after + 1,
        ):
            baseline_start = attack_index - self._basin_span - self._baseline_span
            baseline_end = attack_index - self._basin_span

            baseline_values = [level.rms for level in levels[baseline_start:baseline_end]]
            basin_values = [
                level.rms for level in levels[attack_index - self._basin_span : attack_index]
            ]
            attack_values = [
                level.rms for level in levels[attack_index : attack_index + self._attack_span]
            ]
            persistence_start = attack_index + self._attack_span
            persistence_end = persistence_start + self._persistence_span
            if persistence_end > len(levels):
                continue

            persistence_values = [level.rms for level in levels[persistence_start:persistence_end]]

            baseline = statistics.median(baseline_values)
            basin = self._mean(basin_values)
            attack = self._mean(attack_values)
            persistence = statistics.median(persistence_values)

            if baseline <= 0:
                continue

            baseline_low = min(baseline_values)
            baseline_high = max(baseline_values)
            if baseline_high <= 0:
                continue
            if baseline_low / baseline_high < self._baseline_stability:
                continue

            safe_basin = max(basin, 1e-9)
            if basin / baseline > self._quiet_ratio:
                continue
            if attack / safe_basin < self._min_jump:
                continue
            if persistence / safe_basin < self._min_persistence:
                continue

            basin_start_level = levels[attack_index - self._basin_span]
            attack_level = levels[attack_index]

            basin_start = absolute_start_time_seconds + basin_start_level.start_time_seconds
            attack_time = absolute_start_time_seconds + attack_level.start_time_seconds

            if last_time is not None and attack_time - last_time < self._min_track_seconds:
                continue

            candidates.append(
                AutonomousBoundaryCandidate(
                    time_seconds=attack_time,
                    basin_start_seconds=basin_start,
                    attack_seconds=attack_time,
                    basin_rms=basin,
                    attack_rms=attack,
                    persistence_rms=persistence,
                )
            )
            last_time = attack_time

        return tuple(candidates)

    def detect_proposals(
        self,
        *,
        profile: AcousticProfile,
        absolute_start_time_seconds: float = 0.0,
    ) -> tuple[BoundaryProposal, ...]:
        """Detect audio-only candidates and expose them as boundary proposals."""
        return tuple(
            candidate.as_proposal()
            for candidate in self.detect(
                profile=profile,
                absolute_start_time_seconds=absolute_start_time_seconds,
            )
        )
