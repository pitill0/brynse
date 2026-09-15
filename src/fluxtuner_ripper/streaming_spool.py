"""Disk-backed retention for an open live stream segment."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO


class StreamingSpool:
    """Retain encoded stream bytes on disk using absolute byte offsets."""

    def __init__(
        self,
        *,
        directory: Path,
        copy_chunk_size: int = 64 * 1024,
    ) -> None:
        if copy_chunk_size <= 0:
            raise ValueError("copy_chunk_size must be greater than zero")

        directory.mkdir(parents=True, exist_ok=True)

        fd, name = tempfile.mkstemp(
            dir=directory,
            prefix=".fluxtuner-spool-",
            suffix=".tmp",
        )

        self._path = Path(name)
        self._handle: BinaryIO = os.fdopen(
            fd,
            "w+b",
            buffering=0,
        )
        self._copy_chunk_size = copy_chunk_size
        self._start_offset = 0
        self._end_offset = 0
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def start_offset(self) -> int:
        return self._start_offset

    @property
    def end_offset(self) -> int:
        return self._end_offset

    @property
    def retained_bytes(self) -> int:
        return self._end_offset - self._start_offset

    def append(self, data: bytes) -> tuple[int, int]:
        """Append encoded bytes and return their absolute half-open span."""

        self._require_open()

        if not data:
            return self._end_offset, self._end_offset

        chunk_start = self._end_offset

        self._handle.seek(0, os.SEEK_END)
        written = self._handle.write(data)

        if written != len(data):
            raise RuntimeError("failed to append complete encoded chunk to spool")

        self._end_offset += len(data)

        return chunk_start, self._end_offset

    def contains(self, start: int, end: int) -> bool:
        """Return whether the complete absolute span is retained."""

        if start < 0 or end < start:
            return False

        return self._start_offset <= start and end <= self._end_offset

    def read(self, start: int, end: int) -> bytes:
        """Read one retained absolute half-open byte range."""

        self._require_open()

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
        length = end - start

        self._handle.seek(local_start)
        data = self._handle.read(length)

        if len(data) != length:
            raise RuntimeError("failed to read complete retained span from spool")

        return data

    def discard_before(self, offset: int) -> None:
        """Discard bytes older than ``offset`` while preserving absolute offsets."""

        self._require_open()

        if offset < self._start_offset or offset > self._end_offset:
            raise ValueError(
                f"discard offset {offset} is outside retained range "
                f"[{self._start_offset}, {self._end_offset}]"
            )

        if offset == self._start_offset:
            return

        if offset == self._end_offset:
            self._handle.seek(0)
            self._handle.truncate(0)
            self._start_offset = offset
            return

        fd, temp_name = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=".fluxtuner-spool-compact-",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)

        try:
            with os.fdopen(fd, "wb", buffering=0) as target:
                self._handle.seek(offset - self._start_offset)

                remaining = self._end_offset - offset

                while remaining:
                    chunk = self._handle.read(min(self._copy_chunk_size, remaining))

                    if not chunk:
                        raise RuntimeError("unexpected EOF while compacting streaming spool")

                    written = target.write(chunk)
                    if written != len(chunk):
                        raise RuntimeError("failed to write complete chunk while compacting spool")

                    remaining -= len(chunk)

            self._handle.close()
            os.replace(temp_path, self._path)

            self._handle = self._path.open(
                "r+b",
                buffering=0,
            )
            self._start_offset = offset
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def close(self) -> None:
        """Close and remove the transient spool."""

        if self._closed:
            return

        self._closed = True
        self._handle.close()
        self._path.unlink(missing_ok=True)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("streaming spool is closed")
