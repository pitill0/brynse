"""Concurrent event runtime for incremental generic stream segmentation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import TypeAlias

from fluxtuner_ripper.models import BoundaryCandidate
from fluxtuner_ripper.streaming_runner import (
    StreamingCandidateResult,
    StreamingGenericRunner,
)


@dataclass(frozen=True)
class StreamingRuntimeResult:
    """Completed and still-pending work after both producers finish."""

    completed: tuple[StreamingCandidateResult, ...]
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
    ) -> None:
        self._runner = runner

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
        finished: set[str] = set()

        try:
            while len(finished) < 2:
                event = await queue.get()

                if isinstance(event, _AudioEvent):
                    completed.extend(self._runner.feed(event.chunk))
                    continue

                if isinstance(event, _BoundaryEvent):
                    completed.extend(self._runner.submit_candidate(event.candidate))
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

        return StreamingRuntimeResult(
            completed=tuple(completed),
            pending=self._runner.pending,
        )
