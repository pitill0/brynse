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


def segment_source(
    *,
    source: StreamSource,
    codec: str,
    provider_name: str = "fixed",
    interval_seconds: float | None = None,
) -> dict[str, object]:
    """Segment one encoded StreamSource without CLI/stdin/stdout coupling."""

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
        codec=codec,
        provider_name=provider_name,
        interval_seconds=interval_seconds,
    )
