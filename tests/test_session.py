from __future__ import annotations

from dataclasses import dataclass

from fluxtuner_ripper.models import (
    AcousticBoundaryCandidate,
    BoundaryMatch,
    BoundaryRelation,
    BoundaryRelationResult,
    MetadataSemanticDecision,
    RippingIngestResult,
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
    TimedMetadataEvent,
    TrackCandidate,
)
from fluxtuner_ripper.orchestrator import BoundaryResolution
from fluxtuner_ripper.session import RippingSession


@dataclass
class _Frame:
    end_offset: int
    end_time_seconds: float


class _Timeline:
    def __init__(self) -> None:
        self.frames = (_Frame(end_offset=3000, end_time_seconds=30.0),)


class _Ingestor:
    def __init__(self, results: list[RippingIngestResult]) -> None:
        self._results = list(results)
        self.timeline = _Timeline()
        self.ring_buffer = object()

    def feed(self, chunk: bytes) -> RippingIngestResult:
        return self._results.pop(0)


class _MetadataTracker:
    def __init__(
        self,
        results: list[
            tuple[
                tuple[MetadataSemanticDecision, ...],
                tuple[TrackCandidate, ...],
            ]
        ],
        confirm_tracks: tuple[TrackCandidate, ...] = (),
    ) -> None:
        self._results = list(results)
        self._confirm_tracks = confirm_tracks

    def feed(
        self,
        event: TimedMetadataEvent,
    ) -> tuple[
        tuple[MetadataSemanticDecision, ...],
        tuple[TrackCandidate, ...],
    ]:
        return self._results.pop(0)

    def confirm_current(
        self,
        *,
        audio_offset: int,
        audio_time_seconds: float,
    ) -> tuple[TrackCandidate, ...]:
        return self._confirm_tracks


class _Orchestrator:
    def __init__(self, resolutions: list[BoundaryResolution | None]) -> None:
        self._resolutions = list(resolutions)

    def resolve_boundary(self, **kwargs: object) -> BoundaryResolution | None:
        return self._resolutions.pop(0)


def _track(title: str, start: int, time: float) -> TrackCandidate:
    return TrackCandidate(
        title=title,
        start_offset=start,
        start_time_seconds=time,
        confirmed_at_offset=start + 100,
        confirmed_at_time_seconds=time + 1.0,
    )


def _boundary(track: TrackCandidate) -> BoundaryResolution:
    acoustic = AcousticBoundaryCandidate(
        time_seconds=track.start_time_seconds,
        rms=0.05,
        relative_time_seconds=0.0,
    )
    match = BoundaryMatch(
        track=track,
        acoustic=acoustic,
        delta_seconds=0.0,
    )
    relation = BoundaryRelationResult(
        relation=BoundaryRelation.AGREEMENT,
        semantic_time_seconds=track.start_time_seconds,
        acoustic_time_seconds=track.start_time_seconds,
        signed_delta_seconds=0.0,
    )
    temporal = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=track.start_time_seconds,
        outgoing_end_seconds=track.start_time_seconds,
    )
    split = SplitDecision(
        kind=SplitKind.HARD_CUT,
        incoming_start=track.start_offset,
        outgoing_end=track.start_offset,
    )
    return BoundaryResolution(
        track=track,
        match=match,
        relation=relation,
        temporal=temporal,
        split=split,
    )


def _ingest_event(title: str, offset: int, time: float) -> RippingIngestResult:
    return RippingIngestResult(
        audio=b"",
        metadata_events=(),
        timed_metadata_events=(
            TimedMetadataEvent(
                title=title,
                audio_offset=offset,
                audio_time_seconds=time,
            ),
        ),
    )


def test_session_sets_first_confirmed_track_without_transition() -> None:
    first = _track("Artist - First", 1000, 10.0)
    session = RippingSession(
        ingestor=_Ingestor([_ingest_event(first.title, 1000, 10.0)]),
        metadata_tracker=_MetadataTracker([((), (first,))]),
        orchestrator=_Orchestrator([]),
    )

    result = session.feed(b"chunk")

    assert result.confirmed_tracks == (first,)
    assert result.transitions == ()
    assert session.current_track == first


def test_session_emits_transition_after_resolved_second_track() -> None:
    first = _track("Artist - First", 1000, 10.0)
    second = _track("Artist - Second", 2000, 20.0)
    boundary = _boundary(second)

    session = RippingSession(
        ingestor=_Ingestor(
            [
                _ingest_event(first.title, 1000, 10.0),
                _ingest_event(second.title, 2000, 20.0),
            ]
        ),
        metadata_tracker=_MetadataTracker(
            [
                ((), (first,)),
                ((), (second,)),
            ]
        ),
        orchestrator=_Orchestrator([boundary]),
    )

    first_result = session.feed(b"one")
    second_result = session.feed(b"two")

    assert first_result.transitions == ()
    assert len(second_result.transitions) == 1
    transition = second_result.transitions[0]
    assert transition.outgoing == first
    assert transition.incoming == second
    assert transition.boundary == boundary
    assert session.current_track == second


def test_session_keeps_current_track_when_boundary_is_unresolved() -> None:
    first = _track("Artist - First", 1000, 10.0)
    second = _track("Artist - Second", 2000, 20.0)

    session = RippingSession(
        ingestor=_Ingestor(
            [
                _ingest_event(first.title, 1000, 10.0),
                _ingest_event(second.title, 2000, 20.0),
            ]
        ),
        metadata_tracker=_MetadataTracker(
            [
                ((), (first,)),
                ((), (second,)),
            ]
        ),
        orchestrator=_Orchestrator([None]),
    )

    session.feed(b"one")
    result = session.feed(b"two")

    assert result.transitions == ()
    assert session.current_track == first


def test_session_exposes_semantic_decisions() -> None:
    first = _track("Artist - First", 1000, 10.0)
    decision = MetadataSemanticDecision(
        title=first.title,
        kind=SplitKind.HARD_CUT,
        start_offset=first.start_offset,
        start_time_seconds=first.start_time_seconds,
        lifetime_seconds=9.0,
    )

    session = RippingSession(
        ingestor=_Ingestor([_ingest_event(first.title, 1000, 10.0)]),
        metadata_tracker=_MetadataTracker([((decision,), (first,))]),
        orchestrator=_Orchestrator([]),
    )

    result = session.feed(b"chunk")

    assert result.semantic_decisions == (decision,)
