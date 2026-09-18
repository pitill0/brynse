"""Machine-oriented programmatic interface."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from fluxtuner_ripper.generic_cli import GenericCliError, _run_generic_pipeline
from fluxtuner_ripper.output import TrackFileWriteError, TrackFinalizeError
from fluxtuner_ripper.source import StreamSource

_CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True)
class MachineError:
    code: str
    message: str
    kind: str

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "kind": self.kind,
        }


def classify_machine_error(exc: Exception) -> MachineError:
    if isinstance(exc, GenericCliError):
        return MachineError(
            code="invalid_request",
            message=str(exc),
            kind="validation",
        )

    if isinstance(exc, TrackFinalizeError):
        return MachineError(
            code="finalize_error",
            message=str(exc),
            kind="operational",
        )

    if isinstance(exc, TrackFileWriteError):
        return MachineError(
            code="write_error",
            message=str(exc),
            kind="operational",
        )

    if isinstance(exc, OSError):
        return MachineError(
            code="io_error",
            message=str(exc),
            kind="operational",
        )

    return MachineError(
        code="internal_error",
        message=str(exc),
        kind="internal",
    )


def machine_response_to_json(
    value: SegmentResult | MachineError,
) -> str:
    return json.dumps(
        value.to_dict(),
        sort_keys=True,
    )


def segment_request_from_json(raw: str) -> SegmentRequest | MachineError:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return MachineError(
            code="invalid_request",
            message=str(exc),
            kind="validation",
        )

    if not isinstance(payload, dict):
        return MachineError(
            code="invalid_request",
            message="request JSON must be an object",
            kind="validation",
        )

    try:
        codec = payload["codec"]
        provider = payload["provider"]
    except KeyError as exc:
        return MachineError(
            code="invalid_request",
            message=f"missing required field: {exc.args[0]}",
            kind="validation",
        )

    return SegmentRequest(
        codec=codec,
        provider=provider,
        interval_seconds=payload.get("interval_seconds"),
    )


@dataclass(frozen=True)
class SegmentRequest:
    codec: str
    provider: str
    interval_seconds: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": self.codec,
            "provider": self.provider,
            "interval_seconds": self.interval_seconds,
        }


@dataclass(frozen=True)
class SegmentResult:
    bytes_ingested: int
    codec: str
    provider: str
    interval_seconds: float | None
    boundaries: tuple[dict[str, object], ...]
    segments: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "bytes_ingested": self.bytes_ingested,
            "codec": self.codec,
            "provider": self.provider,
            "interval_seconds": self.interval_seconds,
            "boundaries": list(self.boundaries),
            "segments": list(self.segments),
        }


def run_segment_source(
    *,
    source: StreamSource,
    request: SegmentRequest,
) -> SegmentResult | MachineError:
    try:
        return segment_source(
            source=source,
            request=request,
        )
    except Exception as exc:
        return classify_machine_error(exc)


def segment_source(
    *,
    source: StreamSource,
    request: SegmentRequest,
) -> SegmentResult:
    """Segment one encoded StreamSource from a structured request."""

    def chunks() -> Iterator[bytes]:
        try:
            while True:
                chunk = source.read(_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            source.close()

    payload = _run_generic_pipeline(
        chunks=chunks(),
        codec=request.codec,
        provider_name=request.provider,
        interval_seconds=request.interval_seconds,
    )

    boundaries = cast(
        list[dict[str, object]],
        payload.get("boundaries", []),
    )
    segments = cast(
        list[dict[str, object]],
        payload.get("segments", []),
    )

    return SegmentResult(
        bytes_ingested=cast(int, payload["bytes_ingested"]),
        codec=cast(str, payload["codec"]),
        provider=cast(str, payload["provider"]),
        interval_seconds=cast(
            float | None,
            payload["interval_seconds"],
        ),
        boundaries=tuple(boundaries),
        segments=tuple(segments),
    )
