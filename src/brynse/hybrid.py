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
    FadeGeometryEvidence,
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
        outgoing_fades = self.observe_outgoing_fades(
            profile=profile,
            window=window,
        )
        incoming_fades = self.observe_incoming_fades(
            profile=profile,
            window=window,
        )

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
            outgoing_fades=outgoing_fades,
            incoming_fades=incoming_fades,
            geometries=self.reconcile_fade_geometries(
                outgoing_fades=outgoing_fades,
                incoming_fades=incoming_fades,
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
        """Describe sustained descending RMS-envelope segments."""
        levels = profile.levels
        if len(levels) < 2:
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

        absolute_times = [
            window.start_time_seconds + level.start_time_seconds
            for level in levels
        ]
        rms_values = [level.rms for level in levels]
        smoothed = [
            statistics.median(
                rms_values[
                    max(0, index - 1) :
                    min(len(rms_values), index + 2)
                ]
            )
            for index in range(len(rms_values))
        ]

        rises = self.observe_energy_rises(
            profile=profile,
            window=window,
        )

        segments: list[tuple[int, int]] = []

        segment_start: int | None = None
        floor_index: int | None = None

        def close_segment() -> None:
            nonlocal segment_start, floor_index

            if (
                segment_start is not None
                and floor_index is not None
                and floor_index - segment_start
                >= self._edge_span_levels
            ):
                segments.append((segment_start, floor_index))

            segment_start = None
            floor_index = None

        for index in range(1, len(smoothed)):
            previous = smoothed[index - 1]
            current = smoothed[index]

            if segment_start is None:
                if current < previous:
                    segment_start = index - 1
                    floor_index = index
                continue

            if floor_index is None:
                raise AssertionError('floor index is required')

            if current < smoothed[floor_index]:
                floor_index = index

            start_level = smoothed[segment_start]

            if current >= start_level:
                close_segment()

        close_segment()

        observed: list[OutgoingFadeEvidence] = []

        for raw_start_index, smoothed_floor_index in segments:
            effective_start_index = max(
                raw_start_index,
                smoothed_floor_index - lookback_levels,
            )

            start_search_start = max(
                0,
                effective_start_index - 1,
            )
            start_search_end = min(
                len(levels),
                effective_start_index + 2,
            )
            start_index = max(
                range(start_search_start, start_search_end),
                key=lambda index: (
                    levels[index].rms,
                    index,
                ),
            )

            floor_search_start = max(
                start_index + 1,
                smoothed_floor_index - 1,
            )
            floor_search_end = min(
                len(levels),
                smoothed_floor_index + 2,
            )
            if floor_search_start >= floor_search_end:
                continue

            floor_index = min(
                range(floor_search_start, floor_search_end),
                key=lambda index: levels[index].rms,
            )

            start_rms = levels[start_index].rms
            floor_rms = levels[floor_index].rms
            if start_rms <= floor_rms:
                continue

            floor_time = absolute_times[floor_index]

            later_rises = [
                rise
                for rise in rises
                if (
                    rise.time_seconds > floor_time
                    and rise.time_seconds - floor_time
                    <= self._fade_observation_seconds
                )
            ]
            rise_time = (
                None
                if not later_rises
                else min(
                    later_rises,
                    key=lambda rise: rise.time_seconds,
                ).time_seconds
            )

            observed.append(
                OutgoingFadeEvidence(
                    start_time_seconds=absolute_times[start_index],
                    floor_time_seconds=floor_time,
                    rise_time_seconds=rise_time,
                    start_rms=start_rms,
                    floor_rms=floor_rms,
                    descent_ratio=start_rms / max(floor_rms, 1e-9),
                    descent_consistency=self._descent_consistency(
                        rms_values[start_index : floor_index + 1]
                    ),
                )
            )

        return tuple(observed)

    @staticmethod
    def reconcile_fade_geometries(
        *,
        outgoing_fades: tuple[OutgoingFadeEvidence, ...],
        incoming_fades: tuple[IncomingFadeEvidence, ...],
    ) -> tuple[FadeGeometryEvidence, ...]:
        """Pair each outgoing fade with the incoming fade nearest its rise."""
        if not outgoing_fades or not incoming_fades:
            return ()

        geometries: list[FadeGeometryEvidence] = []

        for outgoing in outgoing_fades:
            anchor_time = (
                outgoing.floor_time_seconds
                if outgoing.rise_time_seconds is None
                else outgoing.rise_time_seconds
            )
            eligible = [
                incoming
                for incoming in incoming_fades
                if incoming.start_time_seconds <= anchor_time
            ]
            if not eligible:
                continue

            incoming = min(
                eligible,
                key=lambda item: (
                    abs(
                        item.established_time_seconds
                        - anchor_time
                    ),
                    -item.ascent_consistency,
                    -item.ascent_ratio,
                    item.start_time_seconds,
                ),
            )

            geometries.append(
                FadeGeometryEvidence(
                    outgoing=outgoing,
                    incoming=incoming,
                )
            )

        return tuple(geometries)

    @staticmethod
    def _ascent_consistency(
        rms_values: list[float],
    ) -> float:
        """Return how consistently the observed RMS region ascends."""
        if len(rms_values) < 2:
            return 0.0

        ascending = sum(
            right >= left
            for left, right in zip(
                rms_values,
                rms_values[1:],
                strict=False,
            )
        )

        return ascending / (len(rms_values) - 1)

    def observe_incoming_fades(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
    ) -> tuple[IncomingFadeEvidence, ...]:
        """Describe sustained ascending RMS-envelope segments."""
        levels = profile.levels
        if len(levels) < 2:
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
        rms_values = [level.rms for level in levels]
        smoothed = [
            statistics.median(
                rms_values[
                    max(0, index - 1) :
                    min(len(rms_values), index + 2)
                ]
            )
            for index in range(len(rms_values))
        ]

        segments: list[tuple[int, int, int]] = []

        segment_start: int | None = None
        peak_index: int | None = None
        segment_end: int | None = None

        def close_segment() -> None:
            nonlocal segment_start, peak_index, segment_end

            if (
                segment_start is not None
                and peak_index is not None
                and segment_end is not None
                and segment_end - segment_start
                >= self._edge_span_levels
            ):
                segments.append(
                    (
                        segment_start,
                        peak_index,
                        segment_end,
                    )
                )

            segment_start = None
            peak_index = None
            segment_end = None

        for index in range(1, len(smoothed)):
            previous = smoothed[index - 1]
            current = smoothed[index]

            if segment_start is None:
                if current > previous:
                    segment_start = index - 1
                    peak_index = index
                    segment_end = index
                continue

            if peak_index is None:
                raise AssertionError('peak index is required')

            start_level = smoothed[segment_start]

            if current <= start_level:
                close_segment()
                continue

            segment_end = index

            if current > smoothed[peak_index]:
                peak_index = index

        close_segment()

        observed: list[IncomingFadeEvidence] = []

        for raw_start_index, raw_peak_index, raw_end_index in segments:
            start_search_start = max(
                0,
                raw_start_index - 1,
            )
            start_search_end = min(
                len(levels),
                raw_start_index + 2,
            )
            start_index = min(
                range(start_search_start, start_search_end),
                key=lambda index: (
                    levels[index].rms,
                    -index,
                ),
            )

            established_search_end = min(
                len(levels),
                raw_peak_index + 2,
                start_index + lookahead_levels + 1,
            )
            if established_search_end <= start_index + 1:
                continue

            established_index = max(
                range(start_index + 1, established_search_end),
                key=lambda index: (
                    levels[index].rms,
                    -index,
                ),
            )

            start_rms = levels[start_index].rms
            established_rms = levels[established_index].rms

            if established_rms <= start_rms:
                continue

            persistence_end = min(
                len(levels),
                established_index + self._persistence_span_levels,
            )
            persistence_values = [
                level.rms
                for level in levels[
                    established_index:persistence_end
                ]
            ]
            if not persistence_values:
                continue

            persistence = statistics.mean(persistence_values)

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
                        rms_values[
                            start_index :
                            min(
                                raw_end_index,
                                start_index + lookahead_levels,
                            )
                            + 1
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
