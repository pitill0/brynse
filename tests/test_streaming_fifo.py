import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fluxtuner_ripper.models import BoundaryCandidate
from fluxtuner_ripper.streaming_fifo import open_fifo_reader
from fluxtuner_ripper.streaming_runner import StreamingGenericRunner
from fluxtuner_ripper.streaming_runtime import AsyncStreamingRuntime
from fluxtuner_ripper.streaming_sources import (
    iter_audio_chunks,
    iter_boundary_jsonl,
)


class _Ingestor:
    def __init__(self) -> None:
        self.timeline = SimpleNamespace(frames=())
        self.ring_buffer = object()
        self._bytes_ingested = 0

    def feed(self, chunk: bytes) -> None:
        self._bytes_ingested += len(chunk)

        self.timeline.frames = (
            SimpleNamespace(
                time_seconds=float(self._bytes_ingested - 1),
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


def _write_fifo(path: Path, payloads: tuple[bytes, ...]) -> None:
    with path.open("wb", buffering=0) as handle:
        for payload in payloads:
            handle.write(payload)


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"),
    reason="Unix FIFOs are not supported on this platform",
)
def test_two_real_fifos_feed_concurrent_streaming_runtime(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        audio_fifo = tmp_path / "audio.fifo"
        boundary_fifo = tmp_path / "boundaries.fifo"

        os.mkfifo(audio_fifo)
        os.mkfifo(boundary_fifo)

        audio_writer = asyncio.create_task(
            asyncio.to_thread(
                _write_fifo,
                audio_fifo,
                (
                    b"x" * 5,
                    b"x" * 7,
                    b"x" * 8,
                ),
            )
        )
        boundary_writer = asyncio.create_task(
            asyncio.to_thread(
                _write_fifo,
                boundary_fifo,
                (
                    b'{"time_seconds":10.0,"source":"agent"}\n',
                    b'{"time_seconds":15.0,"source":"vad"}\n',
                ),
            )
        )

        audio_input, boundary_input = await asyncio.gather(
            open_fifo_reader(audio_fifo),
            open_fifo_reader(boundary_fifo),
        )

        ingestor = _Ingestor()
        resolver = _Resolver()

        runner = StreamingGenericRunner(
            ingestor=ingestor,  # type: ignore[arg-type]
            resolver=resolver,  # type: ignore[arg-type]
            settle_seconds=2.0,
        )
        runtime = AsyncStreamingRuntime(runner=runner)

        try:
            result = await runtime.run(
                audio_source=iter_audio_chunks(
                    audio_input.reader,
                    chunk_size=2,
                ),
                boundary_source=iter_boundary_jsonl(
                    boundary_input.reader,
                ),
            )
        finally:
            audio_input.close()
            boundary_input.close()

        await asyncio.gather(
            audio_writer,
            boundary_writer,
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
