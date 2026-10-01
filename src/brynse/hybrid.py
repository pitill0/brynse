from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from brynse.models import (
    AcousticProfile,
    AcousticWindow,
    TemporalSplitDecision,
    TemporalSplitKind,
)
from brynse.transition_evidence import (
    CrossfadeBasinEvidence,
    CrossfadeRiseEvidence,
    FadeRoutingEvidence,
    IncomingFadeEvidence,
    OutgoingFadeEvidence,
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
class CrossfadeBoundaryGeometry:
    """Resolved temporal boundaries for one plausible crossfade."""

    incoming_start_seconds: float
    outgoing_end_seconds: float
    outgoing_jump_ratio: float
    outgoing_persistence_ratio: float
    outgoing_after_global_ratio: float

    def __post_init__(self) -> None:
        if self.incoming_start_seconds < 0:
            raise ValueError("incoming_start_seconds must be non-negative")
        if self.outgoing_end_seconds < 0:
            raise ValueError("outgoing_end_seconds must be non-negative")
        if self.incoming_start_seconds >= self.outgoing_end_seconds:
            raise ValueError(
                "crossfade geometry requires incoming_start_seconds < outgoing_end_seconds"
            )
        if self.outgoing_jump_ratio < 0:
            raise ValueError("outgoing_jump_ratio must be non-negative")
        if self.outgoing_persistence_ratio < 0:
            raise ValueError("outgoing_persistence_ratio must be non-negative")
        if self.outgoing_after_global_ratio < 0:
            raise ValueError("outgoing_after_global_ratio must be non-negative")


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
        fade_observation_seconds: float = 8.0,
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
        if fade_observation_seconds <= 0:
            raise ValueError("fade_observation_seconds must be greater than zero")

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
        self._fade_observation_seconds = fade_observation_seconds

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

        crossfade = self.resolve_crossfade_geometry(
            profile=profile,
            window=window,
            semantic_time_seconds=semantic_time_seconds,
        )

        if crossfade is None:
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

        return TemporalSplitDecision(
            kind=TemporalSplitKind.CROSSFADE,
            incoming_start_seconds=crossfade.incoming_start_seconds,
            outgoing_end_seconds=crossfade.outgoing_end_seconds,
        )

    def resolve_crossfade_geometry(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        semantic_time_seconds: float,
    ) -> CrossfadeBoundaryGeometry | None:
        """Resolve crossfade boundaries without choosing a split kind."""
        if semantic_time_seconds < 0:
            raise ValueError("semantic_time_seconds must be non-negative")
        if not profile.levels:
            return None

        outgoing = self.observe_crossfade_rise(
            profile=profile,
            window=window,
            semantic_time_seconds=semantic_time_seconds,
        )
        if outgoing is None:
            return None

        incoming = self._select_crossfade_incoming_basin(
            profile=profile,
            window=window,
            semantic_time_seconds=semantic_time_seconds,
        )
        if incoming is None:
            return None
        if incoming.center_time_seconds >= outgoing.time_seconds:
            return None

        return CrossfadeBoundaryGeometry(
            incoming_start_seconds=incoming.center_time_seconds,
            outgoing_end_seconds=outgoing.time_seconds,
            outgoing_jump_ratio=outgoing.jump_ratio,
            outgoing_persistence_ratio=outgoing.persistence_ratio,
            outgoing_after_global_ratio=outgoing.after_global_ratio,
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

    def _eligible_crossfade_incoming_basins(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        semantic_time_seconds: float,
    ) -> tuple[tuple[float, _QuietBasin], ...]:
        eligible: list[tuple[float, _QuietBasin]] = []

        for basin in self._basins(
            profile=profile,
            window=window,
            semantic_time_seconds=semantic_time_seconds,
        ):
            lead = semantic_time_seconds - basin.center_time_seconds
            if not (
                self._crossfade_incoming_min_lead
                <= lead
                <= self._crossfade_incoming_max_lead
            ):
                continue

            eligible.append((lead, basin))

        return tuple(eligible)

    def observe_fade_routing_evidence(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        semantic_time_seconds: float,
    ) -> FadeRoutingEvidence:
        """Compose descriptive fade evidence without resolving temporal boundaries."""
        return FadeRoutingEvidence(
            rise=self.observe_crossfade_rise(
                profile=profile,
                window=window,
                semantic_time_seconds=semantic_time_seconds,
            ),
            basins=self.observe_crossfade_basins(
                profile=profile,
                window=window,
                semantic_time_seconds=semantic_time_seconds,
            ),
            outgoing_fades=self.observe_outgoing_fades(
                profile=profile,
                window=window,
            ),
            incoming_fades=self.observe_incoming_fades(
                profile=profile,
                window=window,
            ),
        )

    def observe_crossfade_basins(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        semantic_time_seconds: float,
    ) -> tuple[CrossfadeBasinEvidence, ...]:
        """Return descriptive evidence for crossfade-compatible incoming basins."""
        if semantic_time_seconds < 0:
            raise ValueError("semantic_time_seconds must be non-negative")
        if not profile.levels:
            return ()

        return tuple(
            CrossfadeBasinEvidence(
                center_time_seconds=basin.center_time_seconds,
                lead_seconds=lead,
                width_seconds=basin.width_seconds,
                prominence=basin.prominence,
                quietness=basin.quietness,
            )
            for lead, basin in self._eligible_crossfade_incoming_basins(
                profile=profile,
                window=window,
                semantic_time_seconds=semantic_time_seconds,
            )
        )

    def _select_crossfade_incoming_basin(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        semantic_time_seconds: float,
    ) -> _QuietBasin | None:
        eligible = [
            (
                self._basin_base_score(basin)
                * (1.0 + min(basin.width_seconds, 0.75)),
                lead,
                basin,
            )
            for lead, basin in self._eligible_crossfade_incoming_basins(
                profile=profile,
                window=window,
                semantic_time_seconds=semantic_time_seconds,
            )
        ]

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

    def observe_energy_rises(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
    ) -> tuple[CrossfadeRiseEvidence, ...]:
        """Return strong persistent energy rises across the full RMS window."""
        levels = profile.levels
        if len(levels) <= self._persistence_span_levels:
            return ()

        rms_values = [level.rms for level in levels]
        global_median = statistics.median(rms_values)
        if global_median <= 0:
            return ()

        qualifying: list[CrossfadeRiseEvidence] = []

        for index in range(
            self._edge_span_levels,
            len(levels) - self._persistence_span_levels,
        ):
            before = statistics.mean(
                rms_values[index - self._edge_span_levels : index]
            )
            immediate = statistics.mean(
                rms_values[index : index + self._edge_span_levels]
            )
            persistent = statistics.mean(
                rms_values[index : index + self._persistence_span_levels]
            )
            if before <= 0:
                continue

            rise = CrossfadeRiseEvidence(
                time_seconds=(
                    window.start_time_seconds
                    + levels[index].start_time_seconds
                ),
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

        return tuple(qualifying)

    @staticmethod
    def _descent_consistency(
        rms_values: list[float],
    ) -> float:
        """Return how consistently a local RMS envelope descends."""
        if len(rms_values) < 2:
            return 0.0

        smoothed = [
            statistics.median(
                rms_values[
                    max(0, index - 1) :
                    min(len(rms_values), index + 2)
                ]
            )
            for index in range(len(rms_values))
        ]

        descending = sum(
            right <= left
            for left, right in zip(
                smoothed,
                smoothed[1:],
                strict=False,
            )
        )

        return descending / (len(smoothed) - 1)

    def observe_outgoing_fades(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
    ) -> tuple[OutgoingFadeEvidence, ...]:
        """Describe RMS decay geometry immediately preceding strong rises."""
        levels = profile.levels
        if len(levels) < 2:
            return ()

        rises = self.observe_energy_rises(
            profile=profile,
            window=window,
        )
        if not rises:
            return ()

        durations = [
            level.end_time_seconds - level.start_time_seconds
            for level in levels
        ]
        level_seconds = statistics.median(durations)
        if level_seconds <= 0:
            return ()

        lookback_levels = max(
            2,
            round(self._fade_observation_seconds / level_seconds),
        )
        observed: list[OutgoingFadeEvidence] = []

        absolute_times = [
            window.start_time_seconds + level.start_time_seconds
            for level in levels
        ]

        for rise in rises:
            rise_index = min(
                range(len(levels)),
                key=lambda index: abs(
                    absolute_times[index] - rise.time_seconds
                ),
            )
            if rise_index <= 0:
                continue

            pre_rise_start = max(
                0,
                rise_index - self._edge_span_levels,
            )
            floor_index = min(
                range(pre_rise_start, rise_index),
                key=lambda index: levels[index].rms,
            )

            lookback_start = max(
                0,
                floor_index - lookback_levels,
            )
            start_index = max(
                range(lookback_start, floor_index + 1),
                key=lambda index: levels[index].rms,
            )

            if start_index >= floor_index:
                continue

            start_rms = levels[start_index].rms
            floor_rms = levels[floor_index].rms
            if start_rms <= floor_rms:
                continue

            observed.append(
                OutgoingFadeEvidence(
                    start_time_seconds=absolute_times[start_index],
                    floor_time_seconds=absolute_times[floor_index],
                    rise_time_seconds=rise.time_seconds,
                    start_rms=start_rms,
                    floor_rms=floor_rms,
                    descent_ratio=start_rms / max(floor_rms, 1e-9),
                    descent_consistency=self._descent_consistency(
                        [
                            level.rms
                            for level in levels[start_index : floor_index + 1]
                        ]
                    ),
                )
            )

        return tuple(observed)

    @staticmethod
    def _ascent_consistency(
        rms_values: list[float],
    ) -> float:
        """Return how consistently a local RMS envelope ascends."""
        if len(rms_values) < 2:
            return 0.0

        smoothed = [
            statistics.median(
                rms_values[
                    max(0, index - 1) :
                    min(len(rms_values), index + 2)
                ]
            )
            for index in range(len(rms_values))
        ]

        ascending = sum(
            right >= left
            for left, right in zip(
                smoothed,
                smoothed[1:],
                strict=False,
            )
        )

        return ascending / (len(smoothed) - 1)

    def observe_incoming_fades(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
    ) -> tuple[IncomingFadeEvidence, ...]:
        """Describe gradual RMS ascent geometry across the analysis window."""
        levels = profile.levels
        if len(levels) < self._persistence_span_levels + 2:
            return ()

        durations = [
            level.end_time_seconds - level.start_time_seconds
            for level in levels
        ]
        level_seconds = statistics.median(durations)
        if level_seconds <= 0:
            return ()

        lookahead_levels = max(
            2,
            round(self._fade_observation_seconds / level_seconds),
        )
        absolute_times = [
            window.start_time_seconds + level.start_time_seconds
            for level in levels
        ]

        observed: list[IncomingFadeEvidence] = []

        for start_index in range(
            0,
            len(levels) - self._persistence_span_levels,
        ):
            end_index = min(
                len(levels) - self._persistence_span_levels,
                start_index + lookahead_levels,
            )
            if end_index <= start_index:
                continue

            established_index = max(
                range(start_index + 1, end_index + 1),
                key=lambda index: levels[index].rms,
            )

            start_rms = levels[start_index].rms
            established_rms = levels[established_index].rms

            if established_rms <= start_rms:
                continue

            persistence = statistics.mean(
                level.rms
                for level in levels[
                    established_index :
                    established_index + self._persistence_span_levels
                ]
            )

            observed.append(
                IncomingFadeEvidence(
                    start_time_seconds=absolute_times[start_index],
                    established_time_seconds=absolute_times[established_index],
                    start_rms=start_rms,
                    established_rms=established_rms,
                    ascent_ratio=(
                        established_rms / max(start_rms, 1e-9)
                    ),
                    ascent_consistency=self._ascent_consistency(
                        [
                            level.rms
                            for level in levels[
                                start_index :
                                min(
                                    len(levels),
                                    established_index
                                    + self._persistence_span_levels,
                                )
                            ]
                        ]
                    ),
                    persistence_ratio=(
                        persistence / max(start_rms, 1e-9)
                    ),
                )
            )

        return tuple(observed)

    def observe_crossfade_rise(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        semantic_time_seconds: float,
    ) -> CrossfadeRiseEvidence | None:
        qualifying = tuple(
            rise
            for rise in self.observe_energy_rises(
                profile=profile,
                window=window,
            )
            if (
                rise.time_seconds - semantic_time_seconds
                >= self._crossfade_min_delay
            )
        )

        if not qualifying:
            return None

        return max(qualifying, key=lambda rise: rise.time_seconds)
