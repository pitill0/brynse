"""Exact accounting of published and explicitly retained encoded audio.

This ledger tracks source-byte ownership, not container file sizes.
Its entries must ultimately be supplied by the publication path itself.
"""

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class CoverageError(RuntimeError):
    """A source byte has no valid disposition or an undeclared duplicate."""


PublicationKind = Literal["track", "unowned"]


@dataclass(frozen=True)
class PublishedRange:
    """A source-byte interval confirmed as published."""

    kind: PublicationKind
    start: int
    end: int
    path: Path
    declared_overlap: bool = False


@dataclass(frozen=True)
class RetainedTail:
    """Unfinalized source bytes deliberately retained for recovery."""

    start: int
    end: int


class PublicationCoverageLedger:
    """Check a monotonic partition of absolute encoded-source offsets.

    Ordinary published ranges must meet exactly. An explicitly declared
    overlap may begin before the existing covered endpoint, but must still
    advance it. Retained terminal bytes are not considered published.
    """

    def __init__(self, *, start_offset: int) -> None:
        if start_offset < 0:
            raise ValueError("start_offset must be non-negative")

        self._start_offset = start_offset
        self._covered_end = start_offset
        self._published: list[PublishedRange] = []
        self._retained_tail: RetainedTail | None = None

    @property
    def start_offset(self) -> int:
        return self._start_offset

    @property
    def covered_end(self) -> int:
        return self._covered_end

    @property
    def published(self) -> tuple[PublishedRange, ...]:
        return tuple(self._published)

    @property
    def retained_tail(self) -> RetainedTail | None:
        return self._retained_tail

    def record_published(
        self,
        *,
        kind: PublicationKind,
        start: int,
        end: int,
        path: Path,
        declared_overlap: bool = False,
    ) -> None:
        """Record a confirmed publication, rejecting unexplained boundaries."""
        if kind not in ("track", "unowned"):
            raise ValueError("unknown publication kind")
        if start < 0 or end <= start:
            raise ValueError("published range must be positive")
        if declared_overlap and kind != "track":
            raise CoverageError("only tracks may declare an overlap")
        if self._retained_tail is not None:
            raise CoverageError("publication after terminal tail declaration")

        if start > self._covered_end:
            raise CoverageError(f"coverage gap [{self._covered_end}, {start})")

        if start < self._covered_end:
            if not declared_overlap:
                raise CoverageError(f"undeclared overlap [{start}, {self._covered_end})")
            if not self._published:
                raise CoverageError("overlap without preceding publication")
            if end <= self._covered_end:
                raise CoverageError("declared overlap must extend published coverage")

        if declared_overlap and start == self._covered_end:
            raise CoverageError("declared overlap has no overlapping bytes")

        # A returned pathname is not evidence of completed publication.
        # O_NONBLOCK avoids hanging on an accidentally substituted FIFO.
        try:
            fd = os.open(
                path,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            )
        except OSError as exc:
            raise CoverageError(f"published artifact missing or unsafe: {path}") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                raise CoverageError(f"artifact not published as regular nonempty file: {path}")
        finally:
            os.close(fd)

        self._published.append(
            PublishedRange(
                kind=kind,
                start=start,
                end=end,
                path=path,
                declared_overlap=declared_overlap,
            )
        )
        self._covered_end = end

    def declare_retained_tail(self, *, start: int, end: int) -> None:
        """Declare exactly the unpublished remainder, without hiding gaps."""
        if self._retained_tail is not None:
            raise CoverageError("tail already declared")
        if start < self._covered_end:
            raise CoverageError("tail overlaps published coverage")
        if start > self._covered_end:
            raise CoverageError(f"coverage gap [{self._covered_end}, {start}) before tail")
        if end <= start:
            raise ValueError("retained tail must be positive")

        self._retained_tail = RetainedTail(start=start, end=end)

    def verify_complete(self, *, end_offset: int) -> None:
        """Ensure all source bytes have an explicit disposition."""
        if end_offset < self._start_offset:
            raise ValueError("end_offset precedes coverage start")

        tail = self._retained_tail
        if tail is not None:
            if tail.end != end_offset:
                raise CoverageError(f"tail ends at {tail.end}, expected {end_offset}")
            return

        if self._covered_end != end_offset:
            raise CoverageError(f"undeclared tail [{self._covered_end}, {end_offset})")
