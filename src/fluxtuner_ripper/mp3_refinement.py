from __future__ import annotations

import statistics
from dataclasses import dataclass

from fluxtuner_ripper.models import (
    AcousticProfile,
    AcousticWindow,
    TemporalSplitDecision,
    TemporalSplitKind,
)


@dataclass(frozen=True)
class Mp3TransitionGeometry:
    """Fine RMS geometry around one coarse MP3 hard-cut boundary."""

    basin_start_seconds: float
    basin_mid_seconds: float
    attack_seconds: float

    @property
    def transition_span_seconds(self) -> float:
        return self.attack_seconds - self.basin_start_seconds


class Mp3BoundaryRefiner:
    """Refine a coarse MP3 hard cut using fine RMS transition geometry."""

    def __init__(
        self,
        *,
        local_radius_seconds: float = 1.25,
        candidate_radius_seconds: float = 1.50,
        quiet_ratio: float = 0.35,
        basin_span_levels: int = 2,
        rise_span_levels: int = 2,
        persistence_span_levels: int = 6,
        min_jump_ratio: float = 2.5,
        min_persistence_ratio: float = 2.0,
        max_transition_span_seconds: float = 3.0,
        near_attack_seconds: float = 0.10,
        pre_basin_lead_seconds: float = 0.20,
        late_basin_start_seconds: float = 0.0,
        late_basin_mid_seconds: float = 0.30,
    ) -> None:
        if local_radius_seconds <= 0:
            raise ValueError("local_radius_seconds must be greater than zero")
        if candidate_radius_seconds <= 0:
            raise ValueError("candidate_radius_seconds must be greater than zero")
        if not 0 < quiet_ratio < 1:
            raise ValueError("quiet_ratio must be between zero and one")
        if basin_span_levels <= 0:
            raise ValueError("basin_span_levels must be greater than zero")
        if rise_span_levels <= 0:
            raise ValueError("rise_span_levels must be greater than zero")
        if persistence_span_levels <= 0:
            raise ValueError("persistence_span_levels must be greater than zero")
        if min_jump_ratio <= 1:
            raise ValueError("min_jump_ratio must be greater than one")
        if min_persistence_ratio <= 1:
            raise ValueError("min_persistence_ratio must be greater than one")
        if max_transition_span_seconds <= 0:
            raise ValueError("max_transition_span_seconds must be greater than zero")
        if near_attack_seconds < 0:
            raise ValueError("near_attack_seconds must be non-negative")
        if pre_basin_lead_seconds < 0:
            raise ValueError("pre_basin_lead_seconds must be non-negative")
        if late_basin_mid_seconds < late_basin_start_seconds:
            raise ValueError("late_basin_mid_seconds must not precede late_basin_start_seconds")

        self._local_radius = local_radius_seconds
        self._candidate_radius = candidate_radius_seconds
        self._quiet_ratio = quiet_ratio
        self._basin_span = basin_span_levels
        self._rise_span = rise_span_levels
        self._persistence_span = persistence_span_levels
        self._min_jump = min_jump_ratio
        self._min_persistence = min_persistence_ratio
        self._max_transition_span = max_transition_span_seconds
        self._near_attack = near_attack_seconds
        self._pre_basin_lead = pre_basin_lead_seconds
        self._late_basin_start = late_basin_start_seconds
        self._late_basin_mid = late_basin_mid_seconds

    @staticmethod
    def _mean_rms(profile: AcousticProfile, start: int, count: int) -> float:
        levels = profile.levels[start : start + count]
        if not levels:
            return 0.0
        return statistics.mean(level.rms for level in levels)

    def detect_geometry(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        current_time_seconds: float,
    ) -> Mp3TransitionGeometry | None:
        if not profile.levels:
            return None

        current_relative = current_time_seconds - window.start_time_seconds
        local_values = [
            level.rms
            for level in profile.levels
            if abs(level.start_time_seconds - current_relative) <= self._local_radius
        ]
        if not local_values:
            return None

        local_median = statistics.median(local_values)
        if local_median <= 0:
            local_median = max(local_values)
        if local_median <= 0:
            return None

        candidates: list[tuple[float, float, float, int]] = []
        for index in range(
            self._basin_span,
            len(profile.levels) - self._persistence_span,
        ):
            level = profile.levels[index]
            delta = level.start_time_seconds - current_relative
            if abs(delta) > self._candidate_radius:
                continue

            basin = self._mean_rms(profile, index - self._basin_span, self._basin_span)
            immediate = self._mean_rms(profile, index, self._rise_span)
            persistent = self._mean_rms(profile, index, self._persistence_span)
            safe_basin = max(basin, 1e-9)
            basin_ratio = basin / local_median
            jump_ratio = immediate / safe_basin
            persistence_ratio = persistent / safe_basin

            if (
                basin_ratio <= self._quiet_ratio
                and jump_ratio >= self._min_jump
                and persistence_ratio >= self._min_persistence
            ):
                candidates.append((abs(delta), -jump_ratio, -persistence_ratio, index))

        if not candidates:
            return None

        _, _, _, attack_index = min(candidates)
        quiet_limit = local_median * self._quiet_ratio
        basin_start_index = attack_index
        while basin_start_index > 0 and profile.levels[basin_start_index - 1].rms <= quiet_limit:
            basin_start_index -= 1

        if basin_start_index == attack_index:
            basin_start_index = max(0, attack_index - self._basin_span)

        basin_start_relative = profile.levels[basin_start_index].start_time_seconds
        attack_relative = profile.levels[attack_index].start_time_seconds
        basin_mid_relative = (basin_start_relative + attack_relative) / 2.0

        return Mp3TransitionGeometry(
            basin_start_seconds=window.start_time_seconds + basin_start_relative,
            basin_mid_seconds=window.start_time_seconds + basin_mid_relative,
            attack_seconds=window.start_time_seconds + attack_relative,
        )

    def refine(
        self,
        *,
        decision: TemporalSplitDecision,
        profile: AcousticProfile,
        window: AcousticWindow,
    ) -> TemporalSplitDecision:
        if decision.kind is not TemporalSplitKind.HARD_CUT:
            return decision

        current = decision.incoming_start_seconds
        geometry = self.detect_geometry(
            profile=profile,
            window=window,
            current_time_seconds=current,
        )
        if geometry is None:
            return decision

        attack_delta = geometry.attack_seconds - current
        basin_start_delta = geometry.basin_start_seconds - current
        basin_mid_delta = geometry.basin_mid_seconds - current

        if geometry.transition_span_seconds > self._max_transition_span:
            boundary = current
        elif attack_delta <= self._near_attack:
            boundary = max(
                window.start_time_seconds,
                geometry.basin_start_seconds - self._pre_basin_lead,
            )
        elif basin_start_delta > self._late_basin_start and basin_mid_delta > self._late_basin_mid:
            boundary = current
        else:
            boundary = geometry.basin_mid_seconds

        return TemporalSplitDecision(
            kind=TemporalSplitKind.HARD_CUT,
            incoming_start_seconds=boundary,
            outgoing_end_seconds=boundary,
        )
