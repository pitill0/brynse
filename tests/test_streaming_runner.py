from types import SimpleNamespace

from fluxtuner_ripper.models import BoundaryCandidate
from fluxtuner_ripper.streaming_runner import StreamingGenericRunner


class _Ingestor:
    def __init__(self) -> None:
        self.timeline = SimpleNamespace(frames=())
        self.ring_buffer = object()

    def feed(self, chunk: bytes) -> None:
        seconds = int(chunk.decode("ascii"))

        if seconds <= 0:
            self.timeline.frames = ()
            return

        self.timeline.frames = (
            SimpleNamespace(
                time_seconds=float(seconds - 1),
                samples=1,
                sample_rate=1,
            ),
        )


class _Resolver:
    def __init__(self) -> None:
        self.calls: list[BoundaryCandidate] = []

    def resolve_candidate(
        self,
        *,
        candidate: BoundaryCandidate,
        timeline: object,
        ring_buffer: object,
    ) -> object:
        self.calls.append(candidate)
        return SimpleNamespace(candidate=candidate)


def test_streaming_runner_keeps_candidate_pending_until_settled() -> None:
    ingestor = _Ingestor()
    resolver = _Resolver()

    runner = StreamingGenericRunner(
        ingestor=ingestor,  # type: ignore[arg-type]
        resolver=resolver,  # type: ignore[arg-type]
        settle_seconds=2.0,
    )

    candidate = BoundaryCandidate(
        time_seconds=10.0,
        source="agent",
    )

    assert runner.submit_candidate(candidate) == ()
    assert runner.pending == (candidate,)

    assert runner.feed(b"11") == ()
    assert runner.pending == (candidate,)
    assert resolver.calls == []

    completed = runner.feed(b"12")

    assert len(completed) == 1
    assert completed[0].candidate == candidate
    assert completed[0].resolution is not None
    assert runner.pending == ()
    assert resolver.calls == [candidate]


def test_streaming_runner_resolves_candidate_submitted_after_audio() -> None:
    ingestor = _Ingestor()
    resolver = _Resolver()

    runner = StreamingGenericRunner(
        ingestor=ingestor,  # type: ignore[arg-type]
        resolver=resolver,  # type: ignore[arg-type]
        settle_seconds=2.0,
    )

    assert runner.feed(b"20") == ()

    candidate = BoundaryCandidate(
        time_seconds=10.0,
        source="vad",
    )

    completed = runner.submit_candidate(candidate)

    assert len(completed) == 1
    assert completed[0].candidate == candidate
    assert runner.pending == ()


def test_streaming_runner_preserves_candidate_order() -> None:
    ingestor = _Ingestor()
    resolver = _Resolver()

    runner = StreamingGenericRunner(
        ingestor=ingestor,  # type: ignore[arg-type]
        resolver=resolver,  # type: ignore[arg-type]
        settle_seconds=1.0,
    )

    later = BoundaryCandidate(
        time_seconds=20.0,
        source="agent",
    )
    earlier = BoundaryCandidate(
        time_seconds=10.0,
        source="semantic_model",
    )

    assert runner.submit_candidate(later) == ()
    assert runner.submit_candidate(earlier) == ()

    completed = runner.feed(b"25")

    assert [result.candidate for result in completed] == [
        earlier,
        later,
    ]


def test_streaming_runner_reports_unresolved_attempt() -> None:
    ingestor = _Ingestor()

    class _UnresolvedResolver:
        def resolve_candidate(
            self,
            *,
            candidate: BoundaryCandidate,
            timeline: object,
            ring_buffer: object,
        ) -> None:
            return None

    runner = StreamingGenericRunner(
        ingestor=ingestor,  # type: ignore[arg-type]
        resolver=_UnresolvedResolver(),  # type: ignore[arg-type]
        settle_seconds=1.0,
    )

    candidate = BoundaryCandidate(
        time_seconds=5.0,
        source="external",
    )

    runner.submit_candidate(candidate)
    completed = runner.feed(b"10")

    assert len(completed) == 1
    assert completed[0].candidate == candidate
    assert completed[0].resolution is None
    assert runner.pending == ()
