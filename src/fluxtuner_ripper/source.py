"""Source contracts for incremental encoded stream ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class StreamSource(Protocol):
    """Minimal readable and closeable encoded stream source contract."""

    @property
    def metadata(self) -> Mapping[str, str]:
        """Transport/source metadata associated with the stream."""
        ...

    def read(self, max_bytes: int) -> bytes:
        """Read up to max_bytes encoded bytes from the source."""
        ...

    def close(self) -> None:
        """Close the source and release or unblock its underlying resource."""
        ...
