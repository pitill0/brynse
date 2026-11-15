"""Write resolved session transitions to finalized track files."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from brynse.integrations.radio.metadata_order import (
    ResolvedMetadataOrder,
    filename_title,
)
from brynse.integrations.radio.ripping import RippingStreamIngestor
from brynse.integrations.radio.session import SegmentTransition, TrackTransition
from brynse.models import SegmentByteRange
from brynse.output import (
    EncodedByteSource,
    SegmentOutputService,
    SegmentRangePlanner,
    create_safe_segment_output_service,
)
from brynse.publication_coverage import CoverageError, PublicationCoverageLedger, RetainedTail

_MULTI_SPACE = re.compile(r"\s+")
_UNSAFE_PATH_CHARS = frozenset(chr(code) for code in (47, 92, 58, 42, 63, 34, 60, 62, 124))


def safe_track_stem(title: str) -> str:
    """Return a filesystem-friendly stem while preserving readable Unicode."""
    cleaned = "".join(
        "_" if char in _UNSAFE_PATH_CHARS or unicodedata.category(char).startswith("C") else char
        for char in unicodedata.normalize("NFC", title)
    )
    stem = _MULTI_SPACE.sub(" ", cleaned).strip(" ._-")
    return stem or "track"


@dataclass(frozen=True)
class WrittenSegment:
    """A finalized outgoing segment written from one resolved transition."""

    transition: SegmentTransition
    path: Path


@dataclass(frozen=True)
class WrittenTrack:
    """A finalized outgoing track written from one resolved transition."""

    transition: TrackTransition
    path: Path


@dataclass(frozen=True)
class WrittenUnownedRegion:
    """A published excluded interval, separate from normal tracks."""

    transition: SegmentTransition
    byte_range: SegmentByteRange
    path: Path


@dataclass(frozen=True)
class PublicationReceipt:
    """Confirmed outputs and exact source-byte ranges for one transition."""

    transition: SegmentTransition
    outgoing_range: SegmentByteRange
    outgoing_path: Path
    incoming_start: int
    unowned_range: SegmentByteRange | None = None
    unowned_path: Path | None = None


class SessionOutputWriter:
    """Convert resolved transitions into finalized outgoing track files."""

    def __init__(
        self,
        *,
        ingestor: RippingStreamIngestor,
        directory: Path,
        codec: str,
        range_planner: SegmentRangePlanner | None = None,
        output_service: SegmentOutputService | None = None,
        source: EncodedByteSource | None = None,
        metadata_order: ResolvedMetadataOrder = "preserve",
        on_unowned_written: Callable[[WrittenUnownedRegion], None] | None = None,
        on_publication_confirmed: Callable[[PublicationReceipt], None] | None = None,
        initial_offset: int | None = None,
    ) -> None:
        if codec not in {"mp3", "aac"}:
            raise ValueError("codec must be 'mp3' or 'aac'")

        self._ingestor = ingestor
        self._source = source
        self._directory = directory
        self._codec = codec
        self._metadata_order = metadata_order
        self._on_unowned_written = on_unowned_written
        self._on_publication_confirmed = on_publication_confirmed
        self._range_planner = range_planner or SegmentRangePlanner()
        self._output_service = output_service or create_safe_segment_output_service()
        self._current_start_offset: int | None = None
        self._coverage_ledger: PublicationCoverageLedger | None = (
            PublicationCoverageLedger(start_offset=initial_offset)
            if initial_offset is not None
            else None
        )
        self._pending_overlap_start: int | None = None

    @property
    def retained_start_offset(self) -> int | None:
        """Return the earliest encoded offset still needed for the next track."""
        return self._current_start_offset

    @property
    def publication_coverage(self) -> PublicationCoverageLedger | None:
        """Inspect source-byte dispositions without mutating the ledger."""
        return getattr(self, "_coverage_ledger", None)

    def finalize_coverage(self, *, spool_start: int, spool_end: int) -> RetainedTail | None:
        """Account for the complete session, including the unfinalized spool tail.

        A session starts at offset zero. Never use a later first publication as
        implicit evidence that earlier source bytes did not exist.
        """
        if not 0 <= spool_start <= spool_end:
            raise ValueError("invalid retained spool offsets")

        ledger = getattr(self, "_coverage_ledger", None)
        if ledger is None:
            ledger = PublicationCoverageLedger(start_offset=0)
            self._coverage_ledger = ledger
        if ledger.start_offset != 0:
            raise CoverageError(f"unaccounted initial head [0, {ledger.start_offset})")
        if spool_start > ledger.covered_end:
            raise CoverageError(f"unaccounted discard [{ledger.covered_end}, {spool_start})")
        if spool_end < ledger.covered_end:
            raise CoverageError("published coverage exceeds received source bytes")

        if spool_end > ledger.covered_end and ledger.retained_tail is None:
            ledger.declare_retained_tail(start=ledger.covered_end, end=spool_end)
        ledger.verify_complete(end_offset=spool_end)
        return ledger.retained_tail

    def can_write_boundary_at(self, incoming_start: int) -> bool:
        """Return whether a boundary advances the currently open segment."""
        if incoming_start < 0:
            raise ValueError("incoming_start must be non-negative")

        current_start = self._current_start_offset
        return current_start is None or incoming_start > current_start

    def can_write_transition(
        self,
        *,
        outgoing_end: int,
        incoming_start: int,
    ) -> bool:
        """Return whether a transition advances the currently open segment."""
        if outgoing_end < 0:
            raise ValueError("outgoing_end must be non-negative")
        if incoming_start < 0:
            raise ValueError("incoming_start must be non-negative")

        current_start = self._current_start_offset
        if current_start is None:
            return True

        return outgoing_end > current_start and incoming_start > current_start

    def write_segment_transition(
        self,
        transition: SegmentTransition,
    ) -> WrittenSegment:
        frames = self._ingestor.timeline.frames
        if not frames:
            raise RuntimeError("cannot write segment without timeline frames")

        last_frame = frames[-1]
        next_end_offset = last_frame.offset + last_frame.length
        ledger = getattr(self, "_coverage_ledger", None)
        previous_start_offset = (
            self._current_start_offset
            if self._current_start_offset is not None
            else ledger.start_offset
            if ledger is not None
            else transition.outgoing.start_offset
        )

        plan = self._range_planner.plan(
            previous_start_offset=previous_start_offset,
            next_end_offset=next_end_offset,
            decision=transition.boundary.split,
        )

        if ledger is None:
            ledger = PublicationCoverageLedger(start_offset=plan.outgoing.start_offset)
            self._coverage_ledger = ledger

        # Reject a missing session head *before* publishing any artifact.
        if plan.outgoing.start_offset > ledger.covered_end:
            raise CoverageError(
                f"unaccounted initial head or gap "
                f"[{ledger.covered_end}, {plan.outgoing.start_offset})"
            )

        path = self._output_service.write_segment(
            source=self._source if self._source is not None else self._ingestor.ring_buffer,
            byte_range=plan.outgoing,
            directory=self._directory,
            stem=safe_track_stem(
                filename_title(
                    transition.outgoing.label or "segment",
                    self._metadata_order,
                )
            ),
            codec=self._codec,
        )

        ledger.record_published(
            kind="track",
            start=plan.outgoing.start_offset,
            end=plan.outgoing.end_offset,
            path=path,
            declared_overlap=(
                plan.outgoing.start_offset < ledger.covered_end
                and plan.outgoing.start_offset == getattr(self, "_pending_overlap_start", None)
            ),
        )

        unowned_path: Path | None = None
        if plan.unowned is not None:
            unowned_path = self._output_service.write_segment(
                source=self._source if self._source is not None else self._ingestor.ring_buffer,
                byte_range=plan.unowned,
                directory=self._directory,
                stem=(f"unowned-{plan.unowned.start_offset:012d}-{plan.unowned.end_offset:012d}"),
                codec=self._codec,
            )
            ledger.record_published(
                kind="unowned",
                start=plan.unowned.start_offset,
                end=plan.unowned.end_offset,
                path=unowned_path,
            )
            callback = getattr(self, "_on_unowned_written", None)
            if callback is not None:
                callback(
                    WrittenUnownedRegion(
                        transition=transition,
                        byte_range=plan.unowned,
                        path=unowned_path,
                    )
                )

        # Validate the exact source-byte disposition before allowing discard.
        expected_end = max(plan.outgoing.end_offset, plan.incoming.start_offset)
        if ledger.covered_end != expected_end:
            raise CoverageError(
                f"coverage gap or mismatch: published through "
                f"{ledger.covered_end}, expected {expected_end}"
            )

        # A provenance failure must never authorize spool discard.
        receipt_callback = getattr(self, "_on_publication_confirmed", None)
        if receipt_callback is not None:
            receipt_callback(
                PublicationReceipt(
                    transition=transition,
                    outgoing_range=plan.outgoing,
                    outgoing_path=path,
                    incoming_start=plan.incoming.start_offset,
                    unowned_range=plan.unowned,
                    unowned_path=unowned_path,
                )
            )

        # Only a previously resolved crossfade may authorize overlap
        # at the beginning of the following outgoing segment.
        self._pending_overlap_start = (
            plan.incoming.start_offset
            if plan.incoming.start_offset < plan.outgoing.end_offset
            else None
        )

        # The runner may discard spool data only after both publications succeed.
        self._current_start_offset = plan.incoming.start_offset

        return WrittenSegment(
            transition=transition,
            path=path,
        )

    def write_transition(self, transition: TrackTransition) -> WrittenTrack:
        written = self.write_segment_transition(
            transition.as_segment_transition(),
        )

        return WrittenTrack(
            transition=transition,
            path=written.path,
        )
