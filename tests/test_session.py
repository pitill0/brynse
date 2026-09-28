from __future__ import annotations

from dataclasses import dataclass

from brynse.integrations.radio.models import (
    BoundaryMatch,
    MetadataSemanticDecision,
    RippingIngestResult,
    TimedMetadataEvent,
    TrackCandidate,
)
from brynse.integrations.radio.orchestrator import BoundaryResolution
from brynse.integrations.radio.session import RippingSession
from brynse.models import (
    AcousticBoundaryCandidate,
    BoundaryRelation,
    BoundaryRelationResult,
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
)


@dataclass
class _Frame:
    offset: int
    length: int
    time_seconds: float
    samples: int
    sample_rate: int


class _Timeline:
    def __init__(self) -> None:
        self.frames = (
            _Frame(
                offset=2900,
                length=100,
                time_seconds=29.0,
                samples=44100,
                sample_rate=44100,
            ),
        )


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
        confirm_results: list[tuple[TrackCandidate, ...]] | None = None,
    ) -> None:
        self._results = list(results)
        self._confirm_tracks = (
            list(confirm_results) if confirm_results is not None else [confirm_tracks]
        )

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
        if not self._confirm_tracks:
            return ()
        return self._confirm_tracks.pop(0)


class _Orchestrator:
    def __init__(self, resolutions: list[BoundaryResolution | None]) -> None:
        self._resolutions = list(resolutions)
        self.calls: list[dict[str, object]] = []

    def resolve_boundary(self, **kwargs: object) -> BoundaryResolution | None:
        self.calls.append(kwargs)
        return self._resolutions.pop(0)


class _SplitAligner:
    def __init__(self, decision: SplitDecision) -> None:
        self.decision = decision
        self.calls: list[dict[str, object]] = []

    def align(self, **kwargs: object) -> SplitDecision:
        self.calls.append(kwargs)
        return self.decision


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


def test_session_auto_confirms_current_track_as_audio_advances() -> None:
    first = _track("Artist - First", 1000, 10.0)
    session = RippingSession(
        ingestor=_Ingestor([_ingest_event(first.title, 1000, 10.0)]),
        metadata_tracker=_MetadataTracker(
            [((), ())],
            confirm_tracks=(first,),
        ),
        orchestrator=_Orchestrator([]),
    )

    result = session.feed(b"chunk")

    assert result.confirmed_tracks == (first,)
    assert result.transitions == ()
    assert session.current_track == first


def test_session_returns_transition_created_by_automatic_confirmation() -> None:
    first = _track("Artist - First", 1000, 10.0)
    second = _track("Artist - Second", 2000, 20.0)
    boundary = _boundary(second)

    tracker = _MetadataTracker(
        [
            ((), (first,)),
            ((), ()),
        ],
        confirm_results=[
            (),
            (second,),
        ],
    )
    session = RippingSession(
        ingestor=_Ingestor(
            [
                _ingest_event(first.title, 1000, 10.0),
                _ingest_event(second.title, 2000, 20.0),
            ]
        ),
        metadata_tracker=tracker,
        orchestrator=_Orchestrator([boundary]),
    )

    first_result = session.feed(b"one")
    second_result = session.feed(b"two")

    assert first_result.transitions == ()
    assert second_result.confirmed_tracks == (second,)
    assert len(second_result.transitions) == 1
    transition = second_result.transitions[0]
    assert transition.outgoing == first
    assert transition.incoming == second
    assert transition.boundary == boundary
    assert session.current_track == second


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


def test_session_converts_authorized_transient_bridge_to_exclusion() -> None:
    first = _track("Artist - First", 1000, 10.0)
    second = _track("Artist - Second", 3000, 22.0)
    transient = MetadataSemanticDecision(
        title="Short break",
        kind=SplitKind.NO_BOUNDARY,
        start_offset=2000,
        start_time_seconds=20.0,
        lifetime_seconds=2.0,
    )
    aligned = SplitDecision(
        kind=SplitKind.EXCLUSION,
        outgoing_end=2000,
        incoming_start=3000,
    )

    orchestrator = _Orchestrator([])
    aligner = _SplitAligner(aligned)
    session = RippingSession(
        ingestor=_Ingestor(
            [
                _ingest_event(first.title, 1000, 10.0),
                _ingest_event("Short break", 2000, 20.0),
                _ingest_event(second.title, 3000, 22.0),
            ]
        ),
        metadata_tracker=_MetadataTracker(
            [
                ((), (first,)),
                ((transient,), ()),
                ((), (second,)),
            ]
        ),
        orchestrator=orchestrator,
        transient_exclusion_policy=lambda decision: decision.title == "Short break",
        split_aligner=aligner,
    )

    session.feed(b"one")
    session.feed(b"two")
    result = session.feed(b"three")

    assert len(result.transitions) == 1
    transition = result.transitions[0]
    assert transition.outgoing == first
    assert transition.incoming == second
    assert transition.boundary.match is None
    assert transition.boundary.relation is None
    assert transition.boundary.temporal.kind is TemporalSplitKind.EXCLUSION
    assert transition.boundary.temporal.outgoing_end_seconds == 20.0
    assert transition.boundary.temporal.incoming_start_seconds == 22.0
    assert transition.boundary.split == aligned
    assert len(aligner.calls) == 1
    assert orchestrator.calls == []


def test_session_ignores_transient_bridge_without_exclusion_policy() -> None:
    first = _track("Artist - First", 1000, 10.0)
    second = _track("Artist - Second", 3000, 22.0)
    transient = MetadataSemanticDecision(
        title="Short break",
        kind=SplitKind.NO_BOUNDARY,
        start_offset=2000,
        start_time_seconds=20.0,
        lifetime_seconds=2.0,
    )
    regular = _boundary(second)
    orchestrator = _Orchestrator([regular])

    session = RippingSession(
        ingestor=_Ingestor(
            [
                _ingest_event(first.title, 1000, 10.0),
                _ingest_event("Short break", 2000, 20.0),
                _ingest_event(second.title, 3000, 22.0),
            ]
        ),
        metadata_tracker=_MetadataTracker(
            [
                ((), (first,)),
                ((transient,), ()),
                ((), (second,)),
            ]
        ),
        orchestrator=orchestrator,
    )

    session.feed(b"one")
    session.feed(b"two")
    result = session.feed(b"three")

    assert len(result.transitions) == 1
    assert result.transitions[0].boundary == regular
    assert len(orchestrator.calls) == 1


def test_session_defers_acoustic_boundary_until_settle_window_is_complete() -> None:
    first = _track("Artist - First", 1000, 10.0)
    second = _track("Artist - Second", 2000, 20.0)
    boundary = _boundary(second)

    ingestor = _Ingestor(
        [
            _ingest_event(first.title, 1000, 10.0),
            _ingest_event(second.title, 2000, 20.0),
            _ingest_event(second.title, 3000, 43.9),
            _ingest_event(second.title, 3100, 44.0),
        ]
    )
    orchestrator = _Orchestrator([boundary])

    session = RippingSession(
        ingestor=ingestor,
        metadata_tracker=_MetadataTracker(
            [
                ((), (first,)),
                ((), (second,)),
                ((), ()),
                ((), ()),
            ]
        ),
        orchestrator=orchestrator,
        acoustic_settle_seconds=24.0,
    )

    first_result = session.feed(b"one")
    assert first_result.transitions == ()
    assert session.current_track == first

    # second becomes semantically durable, but acoustic evidence is not mature.
    ingestor.timeline.frames = (
        _Frame(
            offset=3000,
            length=100,
            time_seconds=29.0,
            samples=44100,
            sample_rate=44100,
        ),
    )
    second_result = session.feed(b"two")

    assert second_result.confirmed_tracks == (second,)
    assert second_result.transitions == ()
    assert orchestrator.calls == []
    assert session.current_track == first

    # 23.9 s of post-anchor audio: still not mature.
    ingestor.timeline.frames = (
        _Frame(
            offset=4390,
            length=100,
            time_seconds=42.9,
            samples=44100,
            sample_rate=44100,
        ),
    )
    almost_result = session.feed(b"almost")

    assert almost_result.transitions == ()
    assert orchestrator.calls == []
    assert session.current_track == first

    # Exactly 24.0 s after the semantic anchor: resolve exactly once.
    ingestor.timeline.frames = (
        _Frame(
            offset=4400,
            length=100,
            time_seconds=43.0,
            samples=44100,
            sample_rate=44100,
        ),
    )
    ready_result = session.feed(b"ready")

    assert len(ready_result.transitions) == 1
    transition = ready_result.transitions[0]
    assert transition.outgoing == first
    assert transition.incoming == second
    assert transition.boundary == boundary

    assert len(orchestrator.calls) == 1
    assert session.current_track == second
