"""Parse externally supplied boundary descriptions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from fluxtuner_ripper.providers import ExternalBoundary


class ExternalBoundaryParseError(ValueError):
    """Raised when external boundary input is invalid."""


def _parse_boundary_object(
    value: object,
    *,
    index: int,
) -> ExternalBoundary:
    if not isinstance(value, Mapping):
        raise ExternalBoundaryParseError(f"boundary {index} must be a JSON object")

    if "time_seconds" not in value:
        raise ExternalBoundaryParseError(f"boundary {index} is missing time_seconds")

    time_seconds = value["time_seconds"]
    source = value.get("source", "external")
    reference_offset = value.get("reference_offset")

    if not isinstance(time_seconds, (int, float)) or isinstance(
        time_seconds,
        bool,
    ):
        raise ExternalBoundaryParseError(f"boundary {index} time_seconds must be numeric")

    if not isinstance(source, str):
        raise ExternalBoundaryParseError(f"boundary {index} source must be a string")

    if reference_offset is not None and (
        not isinstance(reference_offset, int) or isinstance(reference_offset, bool)
    ):
        raise ExternalBoundaryParseError(
            f"boundary {index} reference_offset must be an integer or null"
        )

    try:
        return ExternalBoundary(
            time_seconds=float(time_seconds),
            source=source,
            reference_offset=reference_offset,
        )
    except ValueError as exc:
        raise ExternalBoundaryParseError(f"boundary {index}: {exc}") from exc


def parse_external_boundaries_json(
    text: str,
) -> tuple[ExternalBoundary, ...]:
    """Parse a JSON array of external boundaries."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExternalBoundaryParseError(f"invalid JSON: {exc.msg}") from exc

    if not isinstance(payload, list):
        raise ExternalBoundaryParseError("external boundary JSON must contain an array")

    return tuple(
        _parse_boundary_object(value, index=index) for index, value in enumerate(payload, start=1)
    )


def parse_external_boundaries_jsonl(
    text: str,
) -> tuple[ExternalBoundary, ...]:
    """Parse newline-delimited external boundary objects."""

    boundaries: list[ExternalBoundary] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ExternalBoundaryParseError(
                f"invalid JSON on line {line_number}: {exc.msg}"
            ) from exc

        boundaries.append(
            _parse_boundary_object(
                value,
                index=line_number,
            )
        )

    return tuple(boundaries)


def load_external_boundaries(
    path: Path,
) -> tuple[ExternalBoundary, ...]:
    """Load external boundaries from .json or .jsonl input."""

    if not path.exists():
        raise ExternalBoundaryParseError(f"boundary file does not exist: {path}")
    if not path.is_file():
        raise ExternalBoundaryParseError(f"boundary path is not a file: {path}")

    text = path.read_text(encoding="utf-8")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return parse_external_boundaries_json(text)
    if suffix == ".jsonl":
        return parse_external_boundaries_jsonl(text)

    raise ExternalBoundaryParseError("boundary file must use .json or .jsonl extension")
