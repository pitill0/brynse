"""Concurrent event runtime for incremental generic stream segmentation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from fluxtuner_ripper.generic_output import MaterializedSegment
from fluxtuner_ripper.ingest import EncodedStreamIngestor
from fluxtuner_ripper.models import BoundaryCandidate
from fluxtuner_ripper.orchestrator import CandidateResolver
from fluxtuner_ripper.spooling_ingest import (
    SpoolingEncodedStreamIngestor,
    create_safe_spooling_ingestor,
)
from fluxtuner_ripper.streaming_runner import (
    StreamingCandidateResult,
    StreamingGenericRunner,
)
from fluxtuner_ripper.streaming_sink import StreamingSegmentSink
from fluxtuner_ripper.streaming_spool import StreamingSpool


@dataclass(frozen=True)
class SafeStreamingPipeline:
    """Safely assembled live ingestion, retention, resolution, and output."""

    spooling_ingestor: SpoolingEncodedStreamIngestor
    spool: StreamingSpool
    runner: StreamingGenericRunner
    sink: StreamingSegmentSink
    runtime: AsyncStreamingRuntime

    def close(self) -> None:
        """Release resources owned by the pipeline."""
        self.spool.close()

    def __enter__(self) -> SafeStreamingPipeline:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()


def create_safe_streaming_pipeline(
    *,
    ingestor: EncodedStreamIngestor,
    resolver: CandidateResolver,
    settle_seconds: float,
    output_directory: Path,
    spool_directory: Path,
    codec: str,
) -> SafeStreamingPipeline:
    """Assemble the live runtime with one shared bounded spool."""

    spooling_ingestor = create_safe_spooling_ingestor(
        ingestor=ingestor,
        spool_directory=spool_directory,
    )
    spool = spooling_ingestor.spool

    try:
        runner = StreamingGenericRunner(
            ingestor=spooling_ingestor,
            resolver=resolver,
            settle_seconds=settle_seconds,
        )

        sink = StreamingSegmentSink(
            ingestor=ingestor,
            directory=output_directory,
            codec=codec,
            spool=spool,
            initial_start_source=spooling_ingestor,
        )

        runtime = AsyncStreamingRuntime(
            runner=runner,
            sink=sink,
        )

        return SafeStreamingPipeline(
            spooling_ingestor=spooling_ingestor,
            spool=spool,
            runner=runner,
            sink=sink,
            runtime=runtime,
        )
    except BaseException:
        spool.close()
        raise


@dataclass(frozen=True)
class StreamingRuntimeResult:
    """Completed work, materialized segments, and pending candidates."""

    completed: tuple[StreamingCandidateResult, ...]
    segments: tuple[MaterializedSegment, ...]
    pending: tuple[BoundaryCandidate, ...]


@dataclass(frozen=True)
class _AudioEvent:
    chunk: bytes


@dataclass(frozen=True)
class _BoundaryEvent:
    candidate: BoundaryCandidate


@dataclass(frozen=True)
class _ProducerDone:
    producer: str


_RuntimeEvent: TypeAlias = _AudioEvent | _BoundaryEvent | _ProducerDone


class AsyncStreamingRuntime:
    """Multiplex concurrent audio and boundary producers into one stateful core."""

    def __init__(
        self,
        *,
        runner: StreamingGenericRunner,
        sink: StreamingSegmentSink | None = None,
    ) -> None:
        self._runner = runner
        self._sink = sink

    async def run(
        self,
        *,
        audio_source: AsyncIterable[bytes],
        boundary_source: AsyncIterable[BoundaryCandidate],
    ) -> StreamingRuntimeResult:
        queue: asyncio.Queue[_RuntimeEvent] = asyncio.Queue()

        async def produce_audio() -> None:
            try:
                async for chunk in audio_source:
                    await queue.put(_AudioEvent(chunk=chunk))
            finally:
                await queue.put(_ProducerDone(producer="audio"))

        async def produce_boundaries() -> None:
            try:
                async for candidate in boundary_source:
                    await queue.put(_BoundaryEvent(candidate=candidate))
            finally:
                await queue.put(_ProducerDone(producer="boundaries"))

        audio_task = asyncio.create_task(produce_audio())
        boundary_task = asyncio.create_task(produce_boundaries())

        completed: list[StreamingCandidateResult] = []
        segments: list[MaterializedSegment] = []
        finished: set[str] = set()

        try:
            while len(finished) < 2:
                event = await queue.get()

                if isinstance(event, _AudioEvent):
                    results = self._runner.feed(event.chunk)
                    completed.extend(results)
                    segments.extend(self._materialize(results))
                    continue

                if isinstance(event, _BoundaryEvent):
                    results = self._runner.submit_candidate(event.candidate)
                    completed.extend(results)
                    segments.extend(self._materialize(results))
                    continue

                finished.add(event.producer)

            await asyncio.gather(audio_task, boundary_task)
        finally:
            for task in (audio_task, boundary_task):
                if not task.done():
                    task.cancel()

            await asyncio.gather(
                audio_task,
                boundary_task,
                return_exceptions=True,
            )

        if self._sink is not None:
            tail = self._sink.finalize()
            if tail is not None:
                segments.append(tail)

        return StreamingRuntimeResult(
            completed=tuple(completed),
            segments=tuple(segments),
            pending=self._runner.pending,
        )

    def _materialize(
        self,
        results: tuple[StreamingCandidateResult, ...],
    ) -> tuple[MaterializedSegment, ...]:
        if self._sink is None:
            return ()

        segments: list[MaterializedSegment] = []

        for result in results:
            if result.resolution is None:
                continue

            segments.append(self._sink.accept(result.resolution))

        return tuple(segments)
