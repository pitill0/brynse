from __future__ import annotations


class EncodedAudioRingBuffer:
    """Bounded encoded-audio buffer addressed by absolute byte offsets."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")

        self._max_bytes = max_bytes
        self._data = bytearray()
        self._start_offset = 0
        self._end_offset = 0

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def start_offset(self) -> int:
        """Oldest absolute byte offset still retained."""

        return self._start_offset

    @property
    def end_offset(self) -> int:
        """Absolute byte offset immediately after the newest byte."""

        return self._end_offset

    @property
    def retained_bytes(self) -> int:
        return len(self._data)

    def append(self, data: bytes) -> tuple[int, int]:
        """Append bytes and return their absolute half-open span."""

        if not data:
            return self._end_offset, self._end_offset

        chunk_start = self._end_offset
        self._data.extend(data)
        self._end_offset += len(data)

        overflow = len(self._data) - self._max_bytes
        if overflow > 0:
            del self._data[:overflow]
            self._start_offset += overflow

        return chunk_start, self._end_offset

    def contains(self, start: int, end: int) -> bool:
        """Return whether the complete half-open span is retained."""

        if start < 0 or end < start:
            return False
        return self._start_offset <= start and end <= self._end_offset

    def read(self, start: int, end: int) -> bytes:
        """Return a retained absolute half-open byte range."""

        if start < 0:
            raise ValueError("start must be non-negative")
        if end < start:
            raise ValueError("end must not be smaller than start")
        if not self.contains(start, end):
            raise ValueError(
                f"requested span [{start}, {end}) is outside retained range "
                f"[{self._start_offset}, {self._end_offset})"
            )

        local_start = start - self._start_offset
        local_end = end - self._start_offset
        return bytes(self._data[local_start:local_end])
