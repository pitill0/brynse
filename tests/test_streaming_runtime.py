import asyncio
from types import SimpleNamespace

from fluxtuner_ripper.models import BoundaryCandidate
from fluxtuner_ripper.streaming_runner import StreamingGenericRunner
from fluxtuner_ripper.streaming_runtime import AsyncStreamingRuntime


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


def test_async_streaming_runtime_multiplexes_audio_and_boundaries() -> None:
    async def scenario() -> None:
        ingestor = _Ingestor()
        resolver = _Resolver()

        runner = StreamingGenericRunner(
            ingestor=ingestor,  # type: ignore[arg-type]
            resolver=resolver,  # type: ignore[arg-type]
            settle_seconds=2.0,
        )
        runtime = AsyncStreamingRuntime(runner=runner)

        first_audio_seen = asyncio.Event()
        first_boundary_seen = asyncio.Event()

        async def audio_source():
            yield b"5"
            first_audio_seen.set()

            await first_boundary_seen.wait()

            yield b"11"
            await asyncio.sleep(0)
            yield b"12"
            await asyncio.sleep(0)
            yield b"20"

        async def boundary_source():
            await first_audio_seen.wait()

            candidate = BoundaryCandidate(
                time_seconds=10.0,
                source="agent",
            )
            yield candidate
            first_boundary_seen.set()

            await asyncio.sleep(0)

            yield BoundaryCandidate(
                time_seconds=15.0,
                source="vad",
            )

        result = await runtime.run(
            audio_source=audio_source(),
            boundary_source=boundary_source(),
        )

        assert [item.candidate.time_seconds for item in result.completed] == [
            10.0,
            15.0,
        ]
        assert [item.candidate.source for item in result.completed] == [
            "agent",
            "vad",
        ]
        assert result.pending == ()

        assert [candidate.time_seconds for candidate in resolver.calls] == [
            10.0,
            15.0,
        ]

    asyncio.run(scenario())


def test_async_streaming_runtime_handles_boundary_after_audio_is_ready() -> None:
    async def scenario() -> None:
        ingestor = _Ingestor()
        resolver = _Resolver()

        runner = StreamingGenericRunner(
            ingestor=ingestor,  # type: ignore[arg-type]
            resolver=resolver,  # type: ignore[arg-type]
            settle_seconds=2.0,
        )
        runtime = AsyncStreamingRuntime(runner=runner)

        audio_finished = asyncio.Event()

        async def audio_source():
            yield b"20"
            audio_finished.set()

        async def boundary_source():
            await audio_finished.wait()

            yield BoundaryCandidate(
                time_seconds=10.0,
                source="semantic_model",
            )

        result = await runtime.run(
            audio_source=audio_source(),
            boundary_source=boundary_source(),
        )

        assert len(result.completed) == 1
        assert result.completed[0].candidate.time_seconds == 10.0
        assert result.completed[0].candidate.source == "semantic_model"
        assert result.pending == ()

    asyncio.run(scenario())


def test_async_streaming_runtime_preserves_unmatured_candidate() -> None:
    async def scenario() -> None:
        ingestor = _Ingestor()
        resolver = _Resolver()

        runner = StreamingGenericRunner(
            ingestor=ingestor,  # type: ignore[arg-type]
            resolver=resolver,  # type: ignore[arg-type]
            settle_seconds=5.0,
        )
        runtime = AsyncStreamingRuntime(runner=runner)

        async def audio_source():
            yield b"12"

        candidate = BoundaryCandidate(
            time_seconds=10.0,
            source="agent",
        )

        async def boundary_source():
            yield candidate

        result = await runtime.run(
            audio_source=audio_source(),
            boundary_source=boundary_source(),
        )

        assert result.completed == ()
        assert result.pending == (candidate,)
        assert resolver.calls == []

    asyncio.run(scenario())


def test_async_streaming_runtime_materializes_resolved_segments() -> None:
    async def scenario() -> None:
        ingestor = _Ingestor()
        resolver = _Resolver()

        runner = StreamingGenericRunner(
            ingestor=ingestor,  # type: ignore[arg-type]
            resolver=resolver,  # type: ignore[arg-type]
            settle_seconds=1.0,
        )

        materialized: list[object] = []

        class _Sink:
            def accept(self, resolution: object) -> object:
                segment = SimpleNamespace(
                    index=len(materialized) + 1,
                    resolution=resolution,
                )
                materialized.append(segment)
                return segment

            def finalize(self) -> object:
                segment = SimpleNamespace(
                    index=len(materialized) + 1,
                    tail=True,
                )
                materialized.append(segment)
                return segment

        runtime = AsyncStreamingRuntime(
            runner=runner,
            sink=_Sink(),  # type: ignore[arg-type]
        )

        async def audio_source():
            yield b"20"

        async def boundary_source():
            yield BoundaryCandidate(
                time_seconds=10.0,
                source="agent",
            )
            yield BoundaryCandidate(
                time_seconds=15.0,
                source="vad",
            )

        result = await runtime.run(
            audio_source=audio_source(),
            boundary_source=boundary_source(),
        )

        assert len(result.completed) == 2
        assert len(result.segments) == 3

        assert result.segments[0].index == 1
        assert result.segments[1].index == 2

        assert result.segments[2].index == 3
        assert result.segments[2].tail is True

        assert result.pending == ()

    asyncio.run(scenario())
