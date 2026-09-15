import asyncio

import pytest

from fluxtuner_ripper.external_boundaries import ExternalBoundaryParseError
from fluxtuner_ripper.streaming_sources import (
    iter_audio_chunks,
    iter_boundary_jsonl,
)


def test_iter_audio_chunks_reads_until_eof() -> None:
    async def scenario() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(b"abcdef")
        reader.feed_eof()

        chunks = [
            chunk
            async for chunk in iter_audio_chunks(
                reader,
                chunk_size=2,
            )
        ]

        assert chunks == [
            b"ab",
            b"cd",
            b"ef",
        ]

    asyncio.run(scenario())


def test_iter_boundary_jsonl_reads_incremental_candidates() -> None:
    async def scenario() -> None:
        reader = asyncio.StreamReader()

        reader.feed_data(b'{"time_seconds":12.5,"source":"agent"}\n')
        reader.feed_data(b'{"time_seconds":30.0,"source":"vad","reference_offset":720000}\n')
        reader.feed_eof()

        candidates = [candidate async for candidate in iter_boundary_jsonl(reader)]

        assert [candidate.time_seconds for candidate in candidates] == [
            12.5,
            30.0,
        ]
        assert [candidate.source for candidate in candidates] == [
            "agent",
            "vad",
        ]
        assert candidates[0].reference_offset is None
        assert candidates[1].reference_offset == 720000

    asyncio.run(scenario())


def test_iter_boundary_jsonl_ignores_blank_lines() -> None:
    async def scenario() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(b'\n{"time_seconds":10.0}\n\n')
        reader.feed_eof()

        candidates = [candidate async for candidate in iter_boundary_jsonl(reader)]

        assert len(candidates) == 1
        assert candidates[0].time_seconds == 10.0
        assert candidates[0].source == "external"

    asyncio.run(scenario())


def test_iter_boundary_jsonl_reports_invalid_json() -> None:
    async def scenario() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(b'{"time_seconds":\n')
        reader.feed_eof()

        with pytest.raises(
            ExternalBoundaryParseError,
            match="invalid JSON on line 1",
        ):
            async for _ in iter_boundary_jsonl(reader):
                pass

    asyncio.run(scenario())
