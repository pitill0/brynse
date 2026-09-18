"""Machine-oriented programmatic interface."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from fluxtuner_ripper.generic_cli import _run_generic_pipeline
from fluxtuner_ripper.source import StreamSource

_CHUNK_SIZE = 64 * 1024


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


def segment_source(
    *,
    source: StreamSource,
    request: SegmentRequest,
) -> dict[str, object]:
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

    return _run_generic_pipeline(
        chunks=chunks(),
        codec=request.codec,
        provider_name=request.provider,
        interval_seconds=request.interval_seconds,
    )
