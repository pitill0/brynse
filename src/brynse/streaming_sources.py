"""Async adapters for live audio and boundary input streams."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from brynse.external_boundaries import ExternalBoundaryParseError
from brynse.models import BoundaryCandidate
from brynse.providers import ExternalBoundary


async def iter_audio_chunks(
    reader: asyncio.StreamReader,
    *,
    chunk_size: int = 64 * 1024,
) -> AsyncIterator[bytes]:
    """Yield binary audio chunks until EOF."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    while True:
        chunk = await reader.read(chunk_size)
        if not chunk:
            return

        yield chunk


async def iter_boundary_jsonl(
    reader: asyncio.StreamReader,
) -> AsyncIterator[BoundaryCandidate]:
    """Yield boundary candidates from newline-delimited JSON objects."""

    line_number = 0

    while True:
        raw_line = await reader.readline()
        if not raw_line:
            return

        line_number += 1
        stripped = raw_line.strip()
        if not stripped:
            continue

        try:
            text = stripped.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExternalBoundaryParseError(
                f"boundary line {line_number} is not valid UTF-8"
            ) from exc

        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExternalBoundaryParseError(
                f"invalid JSON on line {line_number}: {exc.msg}"
            ) from exc

        if not isinstance(value, dict):
            raise ExternalBoundaryParseError(f"boundary {line_number} must be a JSON object")

        if "time_seconds" not in value:
            raise ExternalBoundaryParseError(f"boundary {line_number} is missing time_seconds")

        time_seconds = value["time_seconds"]
        source = value.get("source", "external")
        reference_offset = value.get("reference_offset")

        if not isinstance(time_seconds, (int, float)) or isinstance(
            time_seconds,
            bool,
        ):
            raise ExternalBoundaryParseError(f"boundary {line_number} time_seconds must be numeric")

        if not isinstance(source, str):
            raise ExternalBoundaryParseError(f"boundary {line_number} source must be a string")

        if reference_offset is not None and (
            not isinstance(reference_offset, int) or isinstance(reference_offset, bool)
        ):
            raise ExternalBoundaryParseError(
                f"boundary {line_number} reference_offset must be an integer or null"
            )

        try:
            boundary = ExternalBoundary(
                time_seconds=float(time_seconds),
                source=source,
                reference_offset=reference_offset,
            )
        except ValueError as exc:
            raise ExternalBoundaryParseError(f"boundary {line_number}: {exc}") from exc

        yield BoundaryCandidate(
            time_seconds=boundary.time_seconds,
            source=boundary.source,
            reference_offset=boundary.reference_offset,
        )
