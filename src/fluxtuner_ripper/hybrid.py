from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from fluxtuner_ripper.models import (
    AcousticProfile,
    AcousticWindow,
    TemporalSplitDecision,
    TemporalSplitKind,
)


@dataclass(frozen=True)
class _QuietBasin:
    start_time_seconds: float
    end_time_seconds: float
    center_time_seconds: float
    width_seconds: float
    minimum_rms: float
    prominence: float
    quietness: float


@dataclass(frozen=True)
class _EnergyRise:
    time_seconds: float
    jump_ratio: float
    persistence_ratio: float
    after_global_ratio: float


class HybridAcousticSplitResolver:
    """Resolve one semantic transition from fine-grained RMS evidence.

    Normal transitions use a pre-semantic quiet basin as a hard-cut boundary.
    Strong persistent post-semantic energy rises enable an independently
    resolved crossfade with separate incoming and outgoing edges.
    """

    def __init__(
        self,
        *,
        basin_ratio: float = 1.40,
        max_basin_width_seconds: float = 1.50,
        boundary_distance_penalty: float = 0.18,
        edge_span_levels: int = 3,
        persistence_span_levels: int = 10,
        crossfade_min_jump_ratio: float = 2.5,
        crossfade_min_persistence_ratio: float = 3.0,
        crossfade_min_after_global_ratio: float = 2.5,
        crossfade_min_delay_seconds: float = 1.0,
        crossfade_incoming_min_lead_seconds: float = 0.50,
        crossfade_incoming_max_lead_seconds: float = 2.50,
    ) -> None:
        if basin_ratio <= 1.0:
            raise ValueError("basin_ratio must be greater than one")
        if max_basin_width_seconds <= 0:
            raise ValueError("max_basin_width_seconds must be greater than zero")
        if boundary_distance_penalty < 0:
            raise ValueError("boundary_distance_penalty must be non-negative")
        if edge_span_levels <= 0:
            raise ValueError("edge_span_levels must be greater than zero")
        if persistence_span_levels <= 0:
            raise ValueError("persistence_span_levels must be greater than zero")
        if crossfade_min_jump_ratio <= 1.0:
            raise ValueError("crossfade_min_jump_ratio must be greater than one")
        if crossfade_min_persistence_ratio <= 1.0:
            raise ValueError("crossfade_min_persistence_ratio must be greater than one")
        if crossfade_min_after_global_ratio <= 0:
            raise ValueError("crossfade_min_after_global_ratio must be greater than zero")
        if crossfade_min_delay_seconds < 0:
            raise ValueError("crossfade_min_delay_seconds must be non-negative")
        if crossfade_incoming_min_lead_seconds < 0:
            raise ValueError("crossfade_incoming_min_lead_seconds must be non-negative")
        if crossfade_incoming_max_lead_seconds <= crossfade_incoming_min_lead_seconds:
            raise ValueError("crossfade_incoming_max_lead_seconds must exceed minimum lead")

        self._basin_ratio = basin_ratio
        self._max_basin_width = max_basin_width_seconds
        self._boundary_distance_penalty = boundary_distance_penalty
        self._edge_span_levels = edge_span_levels
        self._persistence_span_levels = persistence_span_levels
        self._crossfade_min_jump = crossfade_min_jump_ratio
        self._crossfade_min_persistence = crossfade_min_persistence_ratio
        self._crossfade_min_after_global = crossfade_min_after_global_ratio
        self._crossfade_min_delay = crossfade_min_delay_seconds
        self._crossfade_incoming_min_lead = crossfade_incoming_min_lead_seconds
        self._crossfade_incoming_max_lead = crossfade_incoming_max_lead_seconds

    def resolve(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        semantic_time_seconds: float,
    ) -> TemporalSplitDecision | None:
        if semantic_time_seconds < 0:
            raise ValueError("semantic_time_seconds must be non-negative")
        if not profile.levels:
            return None

        outgoing = self._select_crossfade_outgoing(
            profile=profile,
            window=window,
            semantic_time_seconds=semantic_time_seconds,
        )

        if outgoing is None:
            incoming = self._select_boundary_basin(
                profile=profile,
                window=window,
                semantic_time_seconds=semantic_time_seconds,
            )
            if incoming is None:
                return None

            boundary = incoming.center_time_seconds
            return TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                incoming_start_seconds=boundary,
                outgoing_end_seconds=boundary,
            )

        incoming = self._select_crossfade_incoming_basin(
            profile=profile,
            window=window,
            semantic_time_seconds=semantic_time_seconds,
        )
        if incoming is None:
            return None
        if incoming.center_time_seconds >= outgoing.time_seconds:
            return None

        return TemporalSplitDecision(
            kind=TemporalSplitKind.CROSSFADE,
            incoming_start_seconds=incoming.center_time_seconds,
            outgoing_end_seconds=outgoing.time_seconds,
        )

    def _basins(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        semantic_time_seconds: float,
    ) -> tuple[_QuietBasin, ...]:
        levels = profile.levels
        global_median = statistics.median(level.rms for level in levels)
        if global_median <= 0:
            return ()

        unique: dict[tuple[int, int], _QuietBasin] = {}

        for index, level in enumerate(levels):
            left_rms = levels[index - 1].rms if index > 0 else None
            right_rms = levels[index + 1].rms if index + 1 < len(levels) else None

            if left_rms is not None and level.rms > left_rms:
                continue
            if right_rms is not None and level.rms > right_rms:
                continue

            minimum = level.rms
            threshold = minimum * self._basin_ratio
            left = index
            right = index

            while left > 0 and levels[left - 1].rms <= threshold:
                left -= 1
            while right + 1 < len(levels) and levels[right + 1].rms <= threshold:
                right += 1

            start = window.start_time_seconds + levels[left].start_time_seconds
            end = window.start_time_seconds + levels[right].end_time_seconds
            center = (start + end) / 2.0
            width = end - start

            if center >= semantic_time_seconds:
                continue
            if width > self._max_basin_width:
                continue

            local_start = max(0, index - 6)
            local_end = min(len(levels), index + 7)
            local_median = statistics.median(item.rms for item in levels[local_start:local_end])

            basin = _QuietBasin(
                start_time_seconds=start,
                end_time_seconds=end,
                center_time_seconds=center,
                width_seconds=width,
                minimum_rms=minimum,
                prominence=local_median / max(minimum, 1e-9),
                quietness=global_median / max(minimum, 1e-9),
            )

            key = (left, right)
            existing = unique.get(key)
            if existing is None or basin.minimum_rms < existing.minimum_rms:
                unique[key] = basin

        return tuple(unique.values())

    @staticmethod
    def _basin_base_score(basin: _QuietBasin) -> float:
        width_factor = 1.0 + min(basin.width_seconds, 0.75)
        return (
            math.sqrt(max(basin.quietness, 1e-9))
            * math.sqrt(max(basin.prominence, 1e-9))
            * width_factor
        )

    def _select_boundary_basin(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        semantic_time_seconds: float,
    ) -> _QuietBasin | None:
        basins = self._basins(
            profile=profile,
            window=window,
            semantic_time_seconds=semantic_time_seconds,
        )
        if not basins:
            return None

        def key(basin: _QuietBasin) -> tuple[float, float, float]:
            distance = semantic_time_seconds - basin.center_time_seconds
            score = self._basin_base_score(basin) / (
                1.0 + self._boundary_distance_penalty * distance
            )
            return (-score, distance, basin.center_time_seconds)

        return min(basins, key=key)

    def _select_crossfade_incoming_basin(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        semantic_time_seconds: float,
    ) -> _QuietBasin | None:
        eligible = []
        for basin in self._basins(
            profile=profile,
            window=window,
            semantic_time_seconds=semantic_time_seconds,
        ):
            lead = semantic_time_seconds - basin.center_time_seconds
            if not (self._crossfade_incoming_min_lead <= lead <= self._crossfade_incoming_max_lead):
                continue

            score = self._basin_base_score(basin) * (1.0 + min(basin.width_seconds, 0.75))
            eligible.append((score, lead, basin))

        if not eligible:
            return None

        eligible.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2].center_time_seconds,
            )
        )
        return eligible[0][2]

    def _select_crossfade_outgoing(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        semantic_time_seconds: float,
    ) -> _EnergyRise | None:
        levels = profile.levels
        if len(levels) <= self._persistence_span_levels:
            return None

        rms_values = [level.rms for level in levels]
        global_median = statistics.median(rms_values)
        if global_median <= 0:
            return None

        qualifying: list[_EnergyRise] = []

        for index in range(
            self._edge_span_levels,
            len(levels) - self._persistence_span_levels,
        ):
            time_seconds = window.start_time_seconds + levels[index].start_time_seconds
            delay = time_seconds - semantic_time_seconds
            if delay < self._crossfade_min_delay:
                continue

            before = statistics.mean(rms_values[index - self._edge_span_levels : index])
            immediate = statistics.mean(rms_values[index : index + self._edge_span_levels])
            persistent = statistics.mean(rms_values[index : index + self._persistence_span_levels])
            if before <= 0:
                continue

            rise = _EnergyRise(
                time_seconds=time_seconds,
                jump_ratio=immediate / before,
                persistence_ratio=persistent / before,
                after_global_ratio=persistent / global_median,
            )

            if (
                rise.jump_ratio >= self._crossfade_min_jump
                and rise.persistence_ratio >= self._crossfade_min_persistence
                and rise.after_global_ratio >= self._crossfade_min_after_global
            ):
                qualifying.append(rise)

        if not qualifying:
            return None

        return max(qualifying, key=lambda rise: rise.time_seconds)
