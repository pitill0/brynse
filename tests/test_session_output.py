from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from fluxtuner_ripper.models import (
    AcousticBoundaryCandidate,
    BoundaryMatch,
    BoundaryRelation,
    BoundaryRelationResult,
    SegmentByteRange,
    SegmentWritePlan,
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
    TrackCandidate,
)
from fluxtuner_ripper.orchestrator import BoundaryResolution
from fluxtuner_ripper.session import TrackTransition
from fluxtuner_ripper.session_output import SessionOutputWriter, safe_track_stem


@dataclass
class _Frame:
    offset: int
    length: int


class _Timeline:
    def __init__(self) -> None:
        self.frames = (_Frame(offset=9000, length=1000),)


class _Ingestor:
    def __init__(self) -> None:
        self.timeline = _Timeline()
        self.ring_buffer = object()


class _Planner:
    def __init__(self, plan: SegmentWritePlan) -> None:
        self.plan_result = plan
        self.calls: list[dict[str, object]] = []

    def plan(self, **kwargs: object) -> SegmentWritePlan:
        self.calls.append(kwargs)
        return self.plan_result


class _Output:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls: list[dict[str, object]] = []

    def write_segment(self, **kwargs: object) -> Path:
        self.calls.append(kwargs)
        return self.path


def _track(title: str, start_offset: int, start_time: float) -> TrackCandidate:
    return TrackCandidate(
        title=title,
        start_offset=start_offset,
        start_time_seconds=start_time,
        confirmed_at_offset=start_offset + 100,
        confirmed_at_time_seconds=start_time + 1.0,
    )


def _transition() -> TrackTransition:
    outgoing = _track("Artist / Track: One?", 1000, 10.0)
    incoming = _track("Artist - Track Two", 5000, 20.0)

    acoustic = AcousticBoundaryCandidate(
        time_seconds=20.0,
        rms=0.05,
        relative_time_seconds=0.0,
    )
    match = BoundaryMatch(
        track=incoming,
        acoustic=acoustic,
        delta_seconds=0.0,
    )
    relation = BoundaryRelationResult(
        relation=BoundaryRelation.AGREEMENT,
        semantic_time_seconds=20.0,
        acoustic_time_seconds=20.0,
        signed_delta_seconds=0.0,
    )
    temporal = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=20.0,
        outgoing_end_seconds=20.0,
    )
    split = SplitDecision(
        kind=SplitKind.HARD_CUT,
        incoming_start=5000,
        outgoing_end=5000,
    )

    boundary = BoundaryResolution(
        track=incoming,
        match=match,
        relation=relation,
        temporal=temporal,
        split=split,
    )
    return TrackTransition(
        outgoing=outgoing,
        incoming=incoming,
        boundary=boundary,
    )


def test_safe_track_stem_replaces_unsafe_characters() -> None:
    assert safe_track_stem("Artist / Track: One?") == "Artist _ Track_ One"


def test_safe_track_stem_falls_back_for_empty_title() -> None:
    assert safe_track_stem("///") == "track"


def test_session_output_writer_plans_and_writes_outgoing_track(
    tmp_path: Path,
) -> None:
    transition = _transition()
    plan = SegmentWritePlan(
        outgoing=SegmentByteRange(start_offset=1000, end_offset=5000),
        incoming=SegmentByteRange(start_offset=5000, end_offset=10000),
    )
    planner = _Planner(plan)
    output_path = tmp_path / "Artist _ Track_ One.mp3"
    output = _Output(output_path)
    ingestor = _Ingestor()

    writer = SessionOutputWriter(
        ingestor=ingestor,
        directory=tmp_path,
        codec="mp3",
        range_planner=planner,
        output_service=output,
    )

    written = writer.write_transition(transition)

    assert written.path == output_path
    assert written.transition == transition
    assert planner.calls == [
        {
            "previous_start_offset": 1000,
            "next_end_offset": 10000,
            "decision": transition.boundary.split,
        }
    ]
    assert output.calls == [
        {
            "source": ingestor.ring_buffer,
            "byte_range": plan.outgoing,
            "directory": tmp_path,
            "stem": "Artist _ Track_ One",
            "codec": "mp3",
        }
    ]


def test_session_output_writer_reuses_resolved_incoming_start_for_next_track(
    tmp_path: Path,
) -> None:
    first_transition = _transition()
    third_track = _track("Artist - Track Three", 9000, 30.0)
    second_transition = TrackTransition(
        outgoing=first_transition.incoming,
        incoming=third_track,
        boundary=BoundaryResolution(
            track=third_track,
            match=None,
            relation=None,
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                incoming_start_seconds=30.0,
                outgoing_end_seconds=30.0,
            ),
            split=SplitDecision(
                kind=SplitKind.HARD_CUT,
                incoming_start=9000,
                outgoing_end=9000,
            ),
        ),
    )

    plans = [
        SegmentWritePlan(
            outgoing=SegmentByteRange(start_offset=1000, end_offset=5200),
            incoming=SegmentByteRange(start_offset=4800, end_offset=10000),
        ),
        SegmentWritePlan(
            outgoing=SegmentByteRange(start_offset=4800, end_offset=9000),
            incoming=SegmentByteRange(start_offset=9000, end_offset=10000),
        ),
    ]

    class _SequentialPlanner:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def plan(self, **kwargs: object) -> SegmentWritePlan:
            self.calls.append(kwargs)
            return plans[len(self.calls) - 1]

    planner = _SequentialPlanner()
    output = _Output(tmp_path / "track.mp3")
    writer = SessionOutputWriter(
        ingestor=_Ingestor(),
        directory=tmp_path,
        codec="mp3",
        range_planner=planner,
        output_service=output,
    )

    writer.write_transition(first_transition)
    writer.write_transition(second_transition)

    assert planner.calls[0]["previous_start_offset"] == 1000
    assert planner.calls[1]["previous_start_offset"] == 4800


def test_session_output_writer_reuses_exclusion_incoming_start_for_next_track(
    tmp_path: Path,
) -> None:
    base_transition = _transition()
    exclusion_transition = TrackTransition(
        outgoing=base_transition.outgoing,
        incoming=base_transition.incoming,
        boundary=BoundaryResolution(
            track=base_transition.incoming,
            match=None,
            relation=None,
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                incoming_start_seconds=22.0,
                outgoing_end_seconds=20.0,
            ),
            split=SplitDecision(
                kind=SplitKind.EXCLUSION,
                incoming_start=6200,
                outgoing_end=5000,
            ),
        ),
    )
    exclusion_plan = SegmentWritePlan(
        outgoing=SegmentByteRange(start_offset=1000, end_offset=5000),
        incoming=SegmentByteRange(start_offset=6200, end_offset=10000),
    )
    next_plan = SegmentWritePlan(
        outgoing=SegmentByteRange(start_offset=6200, end_offset=9000),
        incoming=SegmentByteRange(start_offset=9000, end_offset=10000),
    )

    third_track = _track("Artist - Track Three", 9000, 30.0)
    second_transition = TrackTransition(
        outgoing=exclusion_transition.incoming,
        incoming=third_track,
        boundary=BoundaryResolution(
            track=third_track,
            match=None,
            relation=None,
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                incoming_start_seconds=30.0,
                outgoing_end_seconds=30.0,
            ),
            split=SplitDecision(
                kind=SplitKind.HARD_CUT,
                incoming_start=9000,
                outgoing_end=9000,
            ),
        ),
    )

    class _SequentialPlanner:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def plan(self, **kwargs: object) -> SegmentWritePlan:
            self.calls.append(kwargs)
            return (exclusion_plan, next_plan)[len(self.calls) - 1]

    planner = _SequentialPlanner()
    output = _Output(tmp_path / "track.mp3")
    writer = SessionOutputWriter(
        ingestor=_Ingestor(),
        directory=tmp_path,
        codec="mp3",
        range_planner=planner,
        output_service=output,
    )

    writer.write_transition(exclusion_transition)
    writer.write_transition(second_transition)

    assert planner.calls[0]["decision"] == exclusion_transition.boundary.split
    assert planner.calls[1]["previous_start_offset"] == 6200


def test_session_output_writer_rejects_unknown_codec(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="codec must be"):
        SessionOutputWriter(
            ingestor=_Ingestor(),
            directory=tmp_path,
            codec="flac",
        )


def test_session_output_writer_requires_timeline_frames(tmp_path: Path) -> None:
    ingestor = _Ingestor()
    ingestor.timeline.frames = ()

    writer = SessionOutputWriter(
        ingestor=ingestor,
        directory=tmp_path,
        codec="mp3",
    )

    with pytest.raises(RuntimeError, match="without timeline frames"):
        writer.write_transition(_transition())


def test_session_output_writer_uses_explicit_encoded_source(
    tmp_path: Path,
) -> None:
    transition = _transition()
    plan = SegmentWritePlan(
        outgoing=SegmentByteRange(start_offset=1000, end_offset=5000),
        incoming=SegmentByteRange(start_offset=5000, end_offset=10000),
    )
    planner = _Planner(plan)
    output = _Output(tmp_path / "track.mp3")
    ingestor = _Ingestor()
    retained_source = object()

    writer = SessionOutputWriter(
        ingestor=ingestor,
        directory=tmp_path,
        codec="mp3",
        range_planner=planner,
        output_service=output,
        source=retained_source,
    )

    writer.write_transition(transition)

    assert output.calls[0]["source"] is retained_source
    assert output.calls[0]["source"] is not ingestor.ring_buffer


def test_session_output_writer_exposes_next_retained_start_offset(
    tmp_path: Path,
) -> None:
    transition = _transition()
    plan = SegmentWritePlan(
        outgoing=SegmentByteRange(start_offset=1000, end_offset=5200),
        incoming=SegmentByteRange(start_offset=4800, end_offset=10000),
    )
    writer = SessionOutputWriter(
        ingestor=_Ingestor(),
        directory=tmp_path,
        codec="mp3",
        range_planner=_Planner(plan),
        output_service=_Output(tmp_path / "track.mp3"),
    )

    assert writer.retained_start_offset is None

    writer.write_transition(transition)

    assert writer.retained_start_offset == 4800
