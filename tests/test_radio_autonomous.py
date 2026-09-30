from __future__ import annotations

from brynse.integrations.radio.autonomous import (
    AutonomousSegmentState,
    untracked_segment_label,
)
from brynse.models import (
    BoundaryCandidate,
    Segment,
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
)
from brynse.orchestrator import CandidateResolution


def _resolution(
    *,
    time_seconds: float,
    offset: int,
) -> CandidateResolution:
    return CandidateResolution(
        candidate=BoundaryCandidate(
            time_seconds=time_seconds,
            source="autonomous",
        ),
        temporal=TemporalSplitDecision(
            kind=TemporalSplitKind.HARD_CUT,
            incoming_start_seconds=time_seconds,
            outgoing_end_seconds=time_seconds,
        ),
        split=SplitDecision(
            kind=SplitKind.HARD_CUT,
            incoming_start=offset,
            outgoing_end=offset,
        ),
    )


def test_untracked_segment_label_is_deterministic() -> None:
    assert untracked_segment_label(345.217) == "untracked_000000345217"


def test_autonomous_segment_state_requires_an_open_segment() -> None:
    state = AutonomousSegmentState()

    transition = state.transition(
        _resolution(
            time_seconds=100.0,
            offset=5000,
        )
    )

    assert transition is None
    assert state.current is None


def test_autonomous_segment_state_promotes_known_to_untracked() -> None:
    outgoing = Segment(
        start_offset=1000,
        start_time_seconds=10.0,
        label="Artist - Known Track",
    )
    state = AutonomousSegmentState(current=outgoing)
    resolution = _resolution(
        time_seconds=100.25,
        offset=5000,
    )

    transition = state.transition(resolution)

    assert transition is not None
    assert transition.outgoing is outgoing
    assert transition.incoming == Segment(
        start_offset=5000,
        start_time_seconds=100.25,
        label="untracked_000000100250",
    )
    assert transition.boundary is resolution
    assert state.current is transition.incoming


def test_autonomous_segment_state_supports_consecutive_unknown_segments() -> None:
    state = AutonomousSegmentState(
        current=Segment(
            start_offset=1000,
            start_time_seconds=10.0,
            label="Artist - Known Track",
        )
    )

    first = state.transition(
        _resolution(
            time_seconds=100.0,
            offset=5000,
        )
    )
    second = state.transition(
        _resolution(
            time_seconds=200.0,
            offset=9000,
        )
    )

    assert first is not None
    assert second is not None

    assert first.incoming.label == "untracked_000000100000"
    assert second.outgoing is first.incoming
    assert second.incoming.label == "untracked_000000200000"
    assert state.current is second.incoming


def test_known_segment_only_seeds_empty_autonomous_state() -> None:
    known = Segment(
        start_offset=1000,
        start_time_seconds=10.0,
        label="Known",
    )
    autonomous = Segment(
        start_offset=5000,
        start_time_seconds=100.0,
        label="untracked_000000100000",
    )

    state = AutonomousSegmentState()

    state.observe_known_segment(known)
    assert state.current is known

    state.current = autonomous
    state.observe_known_segment(
        Segment(
            start_offset=9000,
            start_time_seconds=200.0,
            label="Later metadata",
        )
    )

    assert state.current is autonomous


# --- materialized segment synchronization ---


def test_materialized_segment_replaces_current_autonomous_state() -> None:
    state = AutonomousSegmentState(
        current=Segment(
            start_offset=3000,
            start_time_seconds=3.0,
            label="untracked_000000003000",
        )
    )

    materialized = Segment(
        start_offset=4500,
        start_time_seconds=4.5,
        label="Known Track",
    )

    state.observe_materialized_segment(materialized)

    assert state.current == materialized



def test_autonomous_segment_state_rejects_boundary_before_minimum_open_duration() -> None:
    state = AutonomousSegmentState(
        current=Segment(
            start_offset=1000,
            start_time_seconds=326.296,
            label="Jazzy James Jr. - Suisse Trip",
        )
    )

    transition = state.transition(
        _resolution(
            time_seconds=330.178,
            offset=9000,
        ),
        minimum_open_seconds=8.0,
    )

    assert transition is None
    assert state.current == Segment(
        start_offset=1000,
        start_time_seconds=326.296,
        label="Jazzy James Jr. - Suisse Trip",
    )


def test_autonomous_segment_state_rejects_temporally_stale_boundary() -> None:
    state = AutonomousSegmentState(
        current=Segment(
            start_offset=5000,
            start_time_seconds=326.296,
            label="Jazzy James Jr. - Suisse Trip",
        )
    )

    transition = state.transition(
        _resolution(
            time_seconds=326.278,
            offset=6000,
        ),
        minimum_open_seconds=8.0,
    )

    assert transition is None


def test_autonomous_segment_state_accepts_boundary_after_minimum_open_duration() -> None:
    state = AutonomousSegmentState(
        current=Segment(
            start_offset=1000,
            start_time_seconds=100.0,
            label="Known Track",
        )
    )

    transition = state.transition(
        _resolution(
            time_seconds=108.001,
            offset=5000,
        ),
        minimum_open_seconds=8.0,
    )

    assert transition is not None
    assert transition.outgoing.label == "Known Track"
    assert transition.incoming.start_time_seconds == 108.001


def test_autonomous_segment_state_rejects_negative_minimum_open_duration() -> None:
    state = AutonomousSegmentState(
        current=Segment(
            start_offset=1000,
            start_time_seconds=100.0,
            label="Known Track",
        )
    )

    import pytest

    with pytest.raises(ValueError, match="minimum_open_seconds"):
        state.transition(
            _resolution(
                time_seconds=110.0,
                offset=5000,
            ),
            minimum_open_seconds=-0.1,
        )
