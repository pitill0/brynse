"""Disk-backed retention for an open live stream segment."""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass
class _StorageChunk:
    start_offset: int
    end_offset: int
    path: Path


def create_safe_streaming_spool(
    *,
    directory: Path,
    storage_chunk_size: int = 1024 * 1024,
    copy_chunk_size: int = 64 * 1024,
) -> StreamingSpool:
    """Create a runtime spool with bounded retained storage."""
    return StreamingSpool(
        directory=directory,
        storage_chunk_size=storage_chunk_size,
        copy_chunk_size=copy_chunk_size,
        max_retained_bytes=8 * 1024 * 1024 * 1024,
        min_free_bytes=512 * 1024 * 1024,
    )


class StreamingSpool:
    """Retain encoded stream bytes on disk using absolute byte offsets."""

    def __init__(
        self,
        *,
        directory: Path,
        storage_chunk_size: int = 1024 * 1024,
        copy_chunk_size: int = 64 * 1024,
        max_retained_bytes: int | None = None,
        min_free_bytes: int | None = None,
    ) -> None:
        if storage_chunk_size <= 0:
            raise ValueError("storage_chunk_size must be greater than zero")
        if copy_chunk_size <= 0:
            raise ValueError("copy_chunk_size must be greater than zero")
        if max_retained_bytes is not None and max_retained_bytes <= 0:
            raise ValueError("max_retained_bytes must be greater than zero")
        if min_free_bytes is not None and min_free_bytes < 0:
            raise ValueError("min_free_bytes must not be negative")

        directory.mkdir(parents=True, exist_ok=True)

        self._directory = directory
        self._session_id = secrets.token_hex(8)
        self._directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self._storage_chunk_size = storage_chunk_size
        self._copy_chunk_size = copy_chunk_size
        self._max_retained_bytes = max_retained_bytes
        self._min_free_bytes = min_free_bytes
        self._chunks: list[_StorageChunk] = []
        self._start_offset = 0
        self._end_offset = 0
        self._closed = False
        self._write_handle: BinaryIO | None = None
        self._write_path: Path | None = None

        try:
            self._create_chunk()
        except BaseException:
            os.close(self._directory_fd)
            raise

    @property
    def path(self) -> Path:
        """Return the oldest active spool chunk path."""

        self._require_open()
        return self._chunks[0].path

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

        if (
            self._max_retained_bytes is not None
            and self.retained_bytes + len(data) > self._max_retained_bytes
        ):
            raise RuntimeError(
                "streaming spool retained byte limit exceeded: "
                f"{self.retained_bytes + len(data)} > "
                f"{self._max_retained_bytes} bytes"
            )

        if self._min_free_bytes is not None:
            import shutil

            free_bytes = shutil.disk_usage(self._directory).free
            required_free_bytes = self._min_free_bytes + len(data)

            if free_bytes < required_free_bytes:
                raise RuntimeError(
                    "insufficient free disk space for streaming spool: "
                    f"{free_bytes} < {required_free_bytes} bytes"
                )

        chunk_start = self._end_offset
        remaining = memoryview(data)

        while remaining:
            chunk = self._ensure_writable_chunk()

            available = self._storage_chunk_size - (chunk.end_offset - chunk.start_offset)
            write_size = min(len(remaining), available)

            handle = self._write_handle
            if handle is None or self._write_path != chunk.path:
                raise RuntimeError("writable spool chunk has no active write handle")

            handle.seek(0, os.SEEK_END)
            written = handle.write(remaining[:write_size])

            if written != write_size:
                raise RuntimeError("failed to append complete encoded chunk to spool")

            chunk.end_offset += written
            self._end_offset += written
            remaining = remaining[written:]

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

        if start == end:
            return b""

        result = bytearray()
        current = start

        for chunk in self._chunks:
            if chunk.end_offset <= current:
                continue

            if chunk.start_offset >= end:
                break

            local_start = max(current, chunk.start_offset)
            local_end = min(end, chunk.end_offset)

            if local_start >= local_end:
                continue

            length = local_end - local_start

            fd = os.open(chunk.path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._directory_fd)
            with os.fdopen(fd, "rb", buffering=0) as handle:
                handle.seek(local_start - chunk.start_offset)
                data = handle.read(length)

            if len(data) != length:
                raise RuntimeError("failed to read complete retained span from spool")

            result.extend(data)
            current = local_end

            if current == end:
                break

        if current != end:
            raise RuntimeError("failed to resolve complete retained span across spool chunks")

        return bytes(result)

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

        self._start_offset = offset

        while self._chunks and self._chunks[0].end_offset <= offset:
            chunk = self._chunks.pop(0)

            if self._write_path == chunk.path:
                self._close_write_handle()

            with contextlib.suppress(FileNotFoundError):
                os.unlink(chunk.path.name, dir_fd=self._directory_fd)

        if not self._chunks:
            self._create_chunk()

    @property
    def recovery_paths(self) -> tuple[Path, ...]:
        """Existing chunk paths containing unfinalized encoded stream data."""
        # Linux procfs resolves the pinned directory even after it is renamed.
        try:
            current = Path(os.readlink(f"/proc/self/fd/{self._directory_fd}"))
        except OSError:
            current = self._directory
        return tuple(current / chunk.path.name for chunk in self._chunks)

    def sync_for_recovery(self) -> None:
        """Durably persist retained chunks and their directory entries.

        If this fails, callers must report that recovery durability is
        unconfirmed; the chunks are left in place for manual salvage.
        """
        self._require_open()
        if self._write_handle is not None:
            self._write_handle.flush()
        for chunk in self._chunks:
            fd = os.open(chunk.path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._directory_fd)
            with os.fdopen(fd, "rb") as handle:
                os.fsync(handle.fileno())
        os.fsync(self._directory_fd)

    def close(self, *, preserve: bool = False) -> None:
        """Close spool; retain chunks for recovery on publication failure."""

        if self._closed:
            return

        self._closed = True
        self._close_write_handle()

        try:
            if not preserve:
                for chunk in self._chunks:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(chunk.path.name, dir_fd=self._directory_fd)
        finally:
            os.close(self._directory_fd)
            self._chunks.clear()

    def _ensure_writable_chunk(self) -> _StorageChunk:
        chunk = self._chunks[-1]

        if chunk.end_offset - chunk.start_offset < self._storage_chunk_size:
            return chunk

        return self._create_chunk()

    def _create_chunk(self) -> _StorageChunk:
        self._close_write_handle()

        fd, name = tempfile.mkstemp(
            dir=f"/proc/self/fd/{self._directory_fd}",
            prefix=f".brynse-spool-{self._session_id}-{self._end_offset:016x}-",
            suffix=".chunk",
        )

        path = self._directory / Path(name).name
        handle = os.fdopen(
            fd,
            "w+b",
            buffering=0,
        )

        chunk = _StorageChunk(
            start_offset=self._end_offset,
            end_offset=self._end_offset,
            path=path,
        )
        self._chunks.append(chunk)
        self._write_handle = handle
        self._write_path = path

        return chunk

    def _close_write_handle(self) -> None:
        handle = self._write_handle
        if handle is None:
            return

        handle.close()
        self._write_handle = None
        self._write_path = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("streaming spool is closed")


def inspect_retained_spool_chunks(directory: Path) -> tuple[Path, ...]:
    """Inventory recorded chunks without following links or deleting anything."""
    if not directory.is_dir():
        return ()
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        names = sorted(os.listdir(fd))
        result = []
        for name in names:
            if not (name.startswith(".brynse-spool-") and name.endswith(".chunk")):
                continue
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                result.append(directory / name)
        return tuple(result)
    finally:
        os.close(fd)
