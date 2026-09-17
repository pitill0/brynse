from types import SimpleNamespace

import pytest

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


def test_streaming_provider_cursor_proposes_only_new_time_window() -> None:
    from fluxtuner_ripper.models import BoundaryCandidate
    from fluxtuner_ripper.streaming_runner import StreamingProviderCursor

    calls: list[tuple[float, float]] = []

    class _Provider:
        def propose(
            self,
            *,
            start_time_seconds: float,
            end_time_seconds: float,
        ) -> tuple[BoundaryCandidate, ...]:
            calls.append(
                (
                    start_time_seconds,
                    end_time_seconds,
                )
            )
            return (
                BoundaryCandidate(
                    time_seconds=end_time_seconds,
                    source="test",
                ),
            )

    cursor = StreamingProviderCursor(provider=_Provider())

    first = cursor.propose_until(12.0)
    second = cursor.propose_until(25.0)
    third = cursor.propose_until(25.0)

    assert calls == [
        (0.0, 12.0),
        (12.0, 25.0),
    ]

    assert tuple(candidate.time_seconds for candidate in first) == (12.0,)
    assert tuple(candidate.time_seconds for candidate in second) == (25.0,)
    assert third == ()


def test_streaming_generic_runner_polls_provider_after_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import fluxtuner_ripper.streaming_runner as streaming_runner

    candidate = SimpleNamespace(
        time_seconds=1.0,
        source="test",
    )
    resolution = SimpleNamespace(candidate=candidate)

    class _Ingestor:
        timeline = SimpleNamespace(
            frames=(
                SimpleNamespace(
                    time_seconds=0.0,
                    samples=1250,
                    sample_rate=1000,
                ),
            ),
        )
        ring_buffer = object()

        def feed(self, chunk: bytes) -> object:
            return object()

    class _ProviderCursor:
        def __init__(self, *, provider: object) -> None:
            self.provider = provider
            self.calls: list[float] = []

        def propose_until(
            self,
            end_time_seconds: float,
        ) -> tuple[object, ...]:
            self.calls.append(end_time_seconds)
            return (candidate,)

    class _Resolver:
        def resolve_candidate(
            self,
            *,
            candidate: object,
            timeline: object,
            ring_buffer: object,
        ) -> object:
            return resolution

    cursor_instances: list[_ProviderCursor] = []

    def cursor_factory(*, provider: object) -> _ProviderCursor:
        cursor = _ProviderCursor(provider=provider)
        cursor_instances.append(cursor)
        return cursor

    monkeypatch.setattr(
        streaming_runner,
        "StreamingProviderCursor",
        cursor_factory,
    )

    runner = streaming_runner.StreamingGenericRunner(
        ingestor=_Ingestor(),
        resolver=_Resolver(),
        settle_seconds=0.2,
        provider=object(),
    )

    completed = runner.feed(b"encoded")

    assert cursor_instances[0].calls == [1.25]
    assert len(completed) == 1
    assert completed[0].candidate is candidate
    assert completed[0].resolution is resolution


def test_streaming_runner_finalize_resolves_pending_candidate_at_eof() -> None:
    from fluxtuner_ripper.providers import ManualBoundaryProvider

    ingestor = _Ingestor()
    resolver = _Resolver()

    runner = StreamingGenericRunner(
        ingestor=ingestor,  # type: ignore[arg-type]
        resolver=resolver,  # type: ignore[arg-type]
        settle_seconds=2.0,
        provider=ManualBoundaryProvider(
            boundary_times_seconds=(10.0,),
        ),
    )

    assert runner.feed(b"11") == ()

    assert tuple(candidate.time_seconds for candidate in runner.pending) == (10.0,)
    assert resolver.calls == []

    completed = runner.finalize()

    assert len(completed) == 1
    assert completed[0].candidate.time_seconds == 10.0
    assert completed[0].resolution is not None

    assert runner.pending == ()
    assert resolver.calls == [completed[0].candidate]

    # EOF finalization must not resolve the same candidate twice.
    assert runner.finalize() == ()
    assert resolver.calls == [completed[0].candidate]
