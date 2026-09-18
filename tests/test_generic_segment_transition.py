from fluxtuner_ripper.models import (
    Segment,
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
)
from fluxtuner_ripper.radio_models import TrackCandidate
from fluxtuner_ripper.radio_orchestrator import BoundaryResolution
from fluxtuner_ripper.session import SegmentTransition, TrackTransition


def _track(
    title: str,
    start_offset: int,
    start_time_seconds: float,
) -> TrackCandidate:
    return TrackCandidate(
        title=title,
        start_offset=start_offset,
        start_time_seconds=start_time_seconds,
        confirmed_at_offset=start_offset + 1000,
        confirmed_at_time_seconds=start_time_seconds + 8.0,
    )


def test_track_transition_projects_to_generic_segment_transition() -> None:
    boundary = BoundaryResolution(
        track=_track("Incoming", 2000, 20.0),
        match=None,
        relation=None,
        temporal=TemporalSplitDecision(
            kind=TemporalSplitKind.HARD_CUT,
            incoming_start_seconds=20.0,
            outgoing_end_seconds=20.0,
        ),
        split=SplitDecision(
            kind=SplitKind.HARD_CUT,
            incoming_start=2000,
            outgoing_end=2000,
        ),
    )

    transition = TrackTransition(
        outgoing=_track("Outgoing", 1000, 10.0),
        incoming=boundary.track,
        boundary=boundary,
    )

    generic = transition.as_segment_transition()

    assert generic == SegmentTransition(
        outgoing=Segment(
            start_offset=1000,
            start_time_seconds=10.0,
            label="Outgoing",
        ),
        incoming=Segment(
            start_offset=2000,
            start_time_seconds=20.0,
            label="Incoming",
        ),
        boundary=boundary,
    )
