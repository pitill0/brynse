from __future__ import annotations

from fluxtuner_ripper.models import (
    AcousticProfile,
    AcousticWindow,
    TemporalSplitDecision,
    TemporalSplitKind,
)
from fluxtuner_ripper.mp3_refinement import Mp3BoundaryRefiner, Mp3TransitionGeometry


class _GeometryRefiner(Mp3BoundaryRefiner):
    def __init__(self, geometry: Mp3TransitionGeometry | None) -> None:
        super().__init__()
        self._geometry = geometry

    def detect_geometry(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        current_time_seconds: float,
    ) -> Mp3TransitionGeometry | None:
        return self._geometry


def _decision(current: float = 10.0) -> TemporalSplitDecision:
    return TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=current,
        outgoing_end_seconds=current,
    )


def _window() -> AcousticWindow:
    return AcousticWindow(
        start_offset=0,
        end_offset=1,
        start_time_seconds=0.0,
        end_time_seconds=20.0,
        data=b"x",
    )


def _profile() -> AcousticProfile:
    return AcousticProfile(levels=())


def test_refiner_uses_basin_mid_for_normal_transition() -> None:
    refiner = _GeometryRefiner(Mp3TransitionGeometry(9.0, 9.5, 10.4))
    refined = refiner.refine(decision=_decision(), profile=_profile(), window=_window())
    assert refined.incoming_start_seconds == 9.5
    assert refined.outgoing_end_seconds == 9.5


def test_refiner_uses_pre_basin_when_attack_is_at_current_boundary() -> None:
    refiner = _GeometryRefiner(Mp3TransitionGeometry(9.45, 9.725, 10.0))
    refined = refiner.refine(decision=_decision(), profile=_profile(), window=_window())
    assert refined.incoming_start_seconds == 9.25
    assert refined.outgoing_end_seconds == 9.25


def test_refiner_keeps_current_for_very_long_transition_span() -> None:
    refiner = _GeometryRefiner(Mp3TransitionGeometry(2.5, 6.35, 10.2))
    assert refiner.refine(decision=_decision(), profile=_profile(), window=_window()) == _decision()


def test_refiner_keeps_current_when_detected_basin_is_too_late() -> None:
    refiner = _GeometryRefiner(Mp3TransitionGeometry(10.1, 10.35, 10.6))
    assert refiner.refine(decision=_decision(), profile=_profile(), window=_window()) == _decision()


def test_refiner_does_not_modify_crossfade_decisions() -> None:
    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.CROSSFADE,
        incoming_start_seconds=9.0,
        outgoing_end_seconds=10.0,
    )
    refiner = _GeometryRefiner(Mp3TransitionGeometry(8.0, 8.5, 9.0))
    assert refiner.refine(decision=decision, profile=_profile(), window=_window()) is decision
