from __future__ import annotations

import pytest

from brynse.hybrid import HybridAcousticSplitResolver
from brynse.models import (
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


def test_hybrid_resolver_exposes_crossfade_geometry_without_split_decision() -> None:
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

    geometry = HybridAcousticSplitResolver().resolve_crossfade_geometry(
        profile=_profile(values),
        window=_window(
            len(values),
            start_time_seconds=425.0,
        ),
        semantic_time_seconds=427.6,
    )

    assert geometry is not None
    assert geometry.incoming_start_seconds == pytest.approx(426.75)
    assert geometry.outgoing_end_seconds == pytest.approx(431.15)
    assert geometry.incoming_start_seconds < geometry.outgoing_end_seconds
    assert not hasattr(geometry, "kind")


def test_hybrid_resolver_exposes_no_crossfade_geometry_for_weak_rise() -> None:
    values = [600.0] * 180
    values[40:49] = [
        230.0,
        215.0,
        200.0,
        185.0,
        175.0,
        180.0,
        195.0,
        210.0,
        225.0,
    ]
    values[125:128] = [200.0, 210.0, 220.0]
    values[128:145] = [700.0] * 17

    geometry = HybridAcousticSplitResolver().resolve_crossfade_geometry(
        profile=_profile(values),
        window=_window(
            len(values),
            start_time_seconds=40.0,
        ),
        semantic_time_seconds=43.5,
    )

    assert geometry is None


def test_hybrid_resolver_exposes_crossfade_rise_as_descriptive_evidence() -> None:
    values = [1000.0] * 180
    values[120:123] = [780.0, 800.0, 820.0]
    values[123:140] = [3600.0] * 17

    evidence = HybridAcousticSplitResolver().observe_crossfade_rise(
        profile=_profile(values),
        window=_window(
            len(values),
            start_time_seconds=425.0,
        ),
        semantic_time_seconds=427.6,
    )

    assert evidence is not None
    assert evidence.time_seconds == pytest.approx(431.15)
    assert evidence.jump_ratio > 0
    assert evidence.persistence_ratio > 0
    assert evidence.after_global_ratio > 0
    assert not hasattr(evidence, "incoming_start_seconds")
    assert not hasattr(evidence, "outgoing_end_seconds")
    assert not hasattr(evidence, "kind")


def test_hybrid_resolver_exposes_crossfade_basins_without_selecting_boundary() -> None:
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

    evidence = HybridAcousticSplitResolver().observe_crossfade_basins(
        profile=_profile(values),
        window=_window(
            len(values),
            start_time_seconds=425.0,
        ),
        semantic_time_seconds=427.6,
    )

    assert evidence

    basin = min(
        evidence,
        key=lambda item: abs(item.center_time_seconds - 426.75),
    )

    assert basin.center_time_seconds == pytest.approx(426.75)
    assert basin.lead_seconds == pytest.approx(0.85)
    assert basin.width_seconds > 0
    assert basin.prominence > 0
    assert basin.quietness > 0

    assert not hasattr(basin, "incoming_start_seconds")
    assert not hasattr(basin, "outgoing_end_seconds")
    assert not hasattr(basin, "kind")


def test_hybrid_resolver_composes_fade_routing_evidence_without_boundaries() -> None:
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

    evidence = HybridAcousticSplitResolver().observe_fade_routing_evidence(
        profile=_profile(values),
        window=_window(
            len(values),
            start_time_seconds=425.0,
        ),
        semantic_time_seconds=427.6,
    )

    assert evidence.rise is not None
    assert evidence.rise.time_seconds == pytest.approx(431.15)
    assert evidence.basins

    assert any(
        basin.center_time_seconds == pytest.approx(426.75)
        for basin in evidence.basins
    )

    assert not hasattr(evidence, "incoming_start_seconds")
    assert not hasattr(evidence, "outgoing_end_seconds")
    assert not hasattr(evidence, "kind")


def test_hybrid_observes_strong_energy_rise_before_semantic_reference() -> None:
    values = [1000.0] * 200
    values[80:83] = [180.0, 190.0, 200.0]
    values[83:100] = [3600.0] * 17

    resolver = HybridAcousticSplitResolver()
    profile = _profile(values)
    window = _window(
        len(values),
        start_time_seconds=100.0,
    )

    rises = resolver.observe_energy_rises(
        profile=profile,
        window=window,
    )

    assert rises
    assert any(
        rise.time_seconds == pytest.approx(104.15)
        for rise in rises
    )


def test_crossfade_rise_keeps_existing_post_semantic_policy() -> None:
    values = [1000.0] * 200
    values[80:83] = [180.0, 190.0, 200.0]
    values[83:100] = [3600.0] * 17

    resolver = HybridAcousticSplitResolver()
    profile = _profile(values)
    window = _window(
        len(values),
        start_time_seconds=100.0,
    )

    generic = resolver.observe_energy_rises(
        profile=profile,
        window=window,
    )
    crossfade = resolver.observe_crossfade_rise(
        profile=profile,
        window=window,
        semantic_time_seconds=106.0,
    )

    assert generic
    assert any(
        rise.time_seconds == pytest.approx(104.15)
        for rise in generic
    )
    assert crossfade is None


def test_hybrid_observes_outgoing_fade_before_strong_rise() -> None:
    values = [1000.0] * 200
    values[40:100] = [
        1200.0 - index * (1020.0 / 59)
        for index in range(60)
    ]
    values[100:117] = [3600.0] * 17

    evidence = HybridAcousticSplitResolver().observe_outgoing_fades(
        profile=_profile(values),
        window=_window(
            len(values),
            start_time_seconds=100.0,
        ),
    )

    assert evidence
    fade = min(
        evidence,
        key=lambda item: abs(item.rise_time_seconds - 105.0),
    )

    assert fade.start_time_seconds == pytest.approx(102.0)
    assert fade.floor_time_seconds == pytest.approx(104.95)
    assert fade.rise_time_seconds == pytest.approx(105.0)
    assert fade.start_rms > fade.floor_rms
    assert fade.descent_ratio > 1.0
    assert fade.descent_consistency == pytest.approx(1.0)


def test_fade_routing_evidence_exposes_outgoing_fade_without_changing_boundaries() -> None:
    values = [1000.0] * 200
    values[40:100] = [
        1200.0 - index * (1020.0 / 59)
        for index in range(60)
    ]
    values[100:117] = [3600.0] * 17

    evidence = HybridAcousticSplitResolver().observe_fade_routing_evidence(
        profile=_profile(values),
        window=_window(
            len(values),
            start_time_seconds=100.0,
        ),
        semantic_time_seconds=106.0,
    )

    assert evidence.outgoing_fades
    assert evidence.rise is None
    assert not hasattr(evidence, "incoming_start_seconds")
    assert not hasattr(evidence, "outgoing_end_seconds")
    assert not hasattr(evidence, "kind")


def test_outgoing_fade_consistency_distinguishes_fluctuating_energy() -> None:
    values = [1000.0] * 200
    values[40:100] = [
        1200.0 if index % 2 == 0 else 300.0
        for index in range(60)
    ]
    values[99] = 180.0
    values[100:117] = [3600.0] * 17

    evidence = HybridAcousticSplitResolver().observe_outgoing_fades(
        profile=_profile(values),
        window=_window(
            len(values),
            start_time_seconds=100.0,
        ),
    )

    assert evidence
    fade = min(
        evidence,
        key=lambda item: abs(item.rise_time_seconds - 105.0),
    )

    assert fade.descent_ratio > 1.0
    assert fade.descent_consistency < 0.75


def test_hybrid_observes_gradual_incoming_fade() -> None:
    values = [180.0] * 200
    values[40:100] = [
        180.0 + index * (1020.0 / 59)
        for index in range(60)
    ]
    values[100:120] = [1200.0] * 20

    evidence = HybridAcousticSplitResolver().observe_incoming_fades(
        profile=_profile(values),
        window=_window(
            len(values),
            start_time_seconds=100.0,
        ),
    )

    assert evidence

    fade = min(
        evidence,
        key=lambda item: abs(item.start_time_seconds - 102.0),
    )

    assert fade.start_time_seconds == pytest.approx(102.0)
    assert fade.established_time_seconds >= fade.start_time_seconds
    assert fade.established_rms > fade.start_rms
    assert fade.ascent_ratio > 1.0
    assert fade.ascent_consistency == pytest.approx(1.0)
    assert fade.persistence_ratio > 1.0


def test_incoming_fade_consistency_distinguishes_fluctuating_energy() -> None:
    values = [180.0] * 200
    values[40:100] = [
        300.0 if index % 2 == 0 else 1200.0
        for index in range(60)
    ]
    values[100:120] = [1200.0] * 20

    evidence = HybridAcousticSplitResolver().observe_incoming_fades(
        profile=_profile(values),
        window=_window(
            len(values),
            start_time_seconds=100.0,
        ),
    )

    assert evidence

    fade = min(
        evidence,
        key=lambda item: abs(item.start_time_seconds - 102.0),
    )

    assert fade.ascent_ratio > 1.0
    assert fade.ascent_consistency < 0.75


def test_fade_routing_evidence_exposes_incoming_fades_descriptively() -> None:
    values = [180.0] * 200
    values[40:100] = [
        180.0 + index * (1020.0 / 59)
        for index in range(60)
    ]
    values[100:120] = [1200.0] * 20

    evidence = HybridAcousticSplitResolver().observe_fade_routing_evidence(
        profile=_profile(values),
        window=_window(
            len(values),
            start_time_seconds=100.0,
        ),
        semantic_time_seconds=106.0,
    )

    assert evidence.incoming_fades
    assert not hasattr(evidence, "incoming_start_seconds")
    assert not hasattr(evidence, "outgoing_end_seconds")
    assert not hasattr(evidence, "kind")
