from __future__ import annotations

import pytest

from fluxtuner_ripper.hybrid import HybridAcousticSplitResolver
from fluxtuner_ripper.models import (
    AcousticLevel,
    AcousticProfile,
    AcousticWindow,
    TemporalSplitKind,
)


def _profile(
    rms_values: list[float],
    *,
    window_seconds: float = 0.05,
) -> AcousticProfile:
    return AcousticProfile(
        levels=tuple(
            AcousticLevel(
                start_time_seconds=index * window_seconds,
                end_time_seconds=(index + 1) * window_seconds,
                rms=rms,
            )
            for index, rms in enumerate(rms_values)
        )
    )


def _window(
    level_count: int,
    *,
    start_time_seconds: float = 90.0,
    window_seconds: float = 0.05,
) -> AcousticWindow:
    duration = level_count * window_seconds
    return AcousticWindow(
        start_offset=0,
        end_offset=level_count,
        start_time_seconds=start_time_seconds,
        end_time_seconds=start_time_seconds + duration,
        data=b"x" * level_count,
    )


def test_hybrid_resolver_prefers_meaningful_boundary_over_distant_quieter_valley() -> None:
    values = [1000.0] * 80
    values[2] = 200.0
    values[3] = 1000.0
    values[61] = 700.0
    values[62] = 300.0
    values[63] = 700.0

    decision = HybridAcousticSplitResolver().resolve(
        profile=_profile(values),
        window=_window(len(values), start_time_seconds=96.0),
        semantic_time_seconds=99.5,
    )

    assert decision is not None
    assert decision.kind is TemporalSplitKind.HARD_CUT
    assert decision.incoming_start_seconds == pytest.approx(99.125)
    assert decision.outgoing_end_seconds == pytest.approx(99.125)


def test_hybrid_resolver_uses_center_of_broad_quiet_basin_for_hard_cut() -> None:
    values = [1000.0] * 100
    values[36:45] = [240.0, 220.0, 205.0, 190.0, 180.0, 185.0, 200.0, 215.0, 235.0]

    decision = HybridAcousticSplitResolver().resolve(
        profile=_profile(values),
        window=_window(len(values), start_time_seconds=40.0),
        semantic_time_seconds=43.8,
    )

    assert decision is not None
    assert decision.kind is TemporalSplitKind.HARD_CUT
    assert decision.incoming_start_seconds == pytest.approx(42.025)
    assert decision.outgoing_end_seconds == pytest.approx(42.025)


def test_hybrid_resolver_detects_crossfade_with_independent_edges() -> None:
    values = [1000.0] * 180
    values[28:42] = [
        620.0,
        600.0,
        590.0,
        580.0,
        570.0,
        560.0,
        550.0,
        540.0,
        535.0,
        540.0,
        550.0,
        565.0,
        585.0,
        610.0,
    ]
    values[120:123] = [780.0, 800.0, 820.0]
    values[123:140] = [3600.0] * 17

    decision = HybridAcousticSplitResolver().resolve(
        profile=_profile(values),
        window=_window(len(values), start_time_seconds=425.0),
        semantic_time_seconds=427.6,
    )

    assert decision is not None
    assert decision.kind is TemporalSplitKind.CROSSFADE
    assert decision.incoming_start_seconds == pytest.approx(426.75)
    assert decision.outgoing_end_seconds == pytest.approx(431.15)
    assert decision.incoming_start_seconds < decision.outgoing_end_seconds


def test_hybrid_resolver_does_not_treat_weak_global_rise_as_crossfade() -> None:
    values = [600.0] * 180
    values[40:49] = [230.0, 215.0, 200.0, 185.0, 175.0, 180.0, 195.0, 210.0, 225.0]
    values[125:128] = [200.0, 210.0, 220.0]
    values[128:145] = [700.0] * 17

    decision = HybridAcousticSplitResolver().resolve(
        profile=_profile(values),
        window=_window(len(values), start_time_seconds=40.0),
        semantic_time_seconds=43.5,
    )

    assert decision is not None
    assert decision.kind is TemporalSplitKind.HARD_CUT


def test_hybrid_resolver_rejects_oversized_quiet_region() -> None:
    values = [1000.0] * 120
    values[20:70] = [200.0] * 50
    values[86:89] = [650.0, 300.0, 650.0]

    decision = HybridAcousticSplitResolver().resolve(
        profile=_profile(values),
        window=_window(len(values), start_time_seconds=90.0),
        semantic_time_seconds=94.8,
    )

    assert decision is not None
    assert decision.kind is TemporalSplitKind.HARD_CUT
    assert decision.incoming_start_seconds == pytest.approx(94.375)


def test_hybrid_resolver_returns_none_for_empty_profile() -> None:
    decision = HybridAcousticSplitResolver().resolve(
        profile=AcousticProfile(levels=()),
        window=_window(1),
        semantic_time_seconds=90.0,
    )

    assert decision is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"basin_ratio": 1.0}, "basin_ratio"),
        ({"max_basin_width_seconds": 0.0}, "max_basin_width_seconds"),
        ({"boundary_distance_penalty": -0.1}, "boundary_distance_penalty"),
        ({"edge_span_levels": 0}, "edge_span_levels"),
        ({"persistence_span_levels": 0}, "persistence_span_levels"),
        ({"crossfade_min_jump_ratio": 1.0}, "crossfade_min_jump_ratio"),
        ({"crossfade_min_persistence_ratio": 1.0}, "crossfade_min_persistence_ratio"),
        ({"crossfade_min_after_global_ratio": 0.0}, "crossfade_min_after_global_ratio"),
        ({"crossfade_min_delay_seconds": -0.1}, "crossfade_min_delay_seconds"),
        ({"crossfade_incoming_min_lead_seconds": -0.1}, "crossfade_incoming_min_lead_seconds"),
        (
            {
                "crossfade_incoming_min_lead_seconds": 1.0,
                "crossfade_incoming_max_lead_seconds": 1.0,
            },
            "crossfade_incoming_max_lead_seconds",
        ),
    ],
)
def test_hybrid_resolver_rejects_invalid_configuration(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        HybridAcousticSplitResolver(**kwargs)


def test_hybrid_resolver_rejects_negative_semantic_time() -> None:
    with pytest.raises(ValueError, match="semantic_time_seconds"):
        HybridAcousticSplitResolver().resolve(
            profile=_profile([1.0, 2.0, 1.0]),
            window=_window(3),
            semantic_time_seconds=-1.0,
        )
