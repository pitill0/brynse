"""F-2: emitted intervals must cover committed audio without gaps."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from brynse.integrations.radio.session_output import SessionOutputWriter
from brynse.models import SplitDecision, SplitKind
from brynse.output import SegmentRangePlanner


class RecordingService:
    def __init__(self, *, omit_unowned: bool = False) -> None:
        self.omit_unowned = omit_unowned
        self.published: list[tuple[int, int]] = []

    def write_segment(
        self,
        *,
        byte_range,
        stem: str,
        directory: Path,
        **kwargs,
    ) -> Path:
        path = directory / f"{stem}.mp3"
        if not (self.omit_unowned and stem.startswith("unowned-")):
            path.write_bytes(b"simulated-published-audio")
            self.published.append((byte_range.start_offset, byte_range.end_offset))
        return path


def make_writer(service: RecordingService) -> SessionOutputWriter:
    writer = object.__new__(SessionOutputWriter)
    writer._ingestor = SimpleNamespace(
        timeline=SimpleNamespace(frames=[SimpleNamespace(offset=900, length=100)]),
        ring_buffer=object(),
    )
    writer._source = object()
    writer._directory = Path("/tmp/f2-coverage-test")
    writer._codec = "mp3"
    writer._metadata_order = "preserve"
    writer._range_planner = SegmentRangePlanner()
    writer._output_service = service
    writer._current_start_offset = 100
    writer._coverage_ledger = None
    writer._on_publication_confirmed = None
    writer._pending_overlap_start = None
    return writer


def transition():
    return SimpleNamespace(
        outgoing=SimpleNamespace(start_offset=100, label="Song A"),
        boundary=SimpleNamespace(
            split=SplitDecision(
                kind=SplitKind.EXCLUSION,
                outgoing_end=400,
                incoming_start=500,
            )
        ),
    )


def assert_contiguous_coverage(
    ranges: list[tuple[int, int]],
    *,
    start: int,
    end: int,
) -> None:
    cursor = start
    for lower, upper in sorted(ranges):
        assert lower == cursor, f"coverage gap or overlap at {cursor}: [{lower}, {upper})"
        assert upper > lower
        cursor = upper
    assert cursor == end, f"uncovered tail [{cursor}, {end})"


def test_transition_has_contiguous_published_coverage(tmp_path: Path) -> None:
    service = RecordingService()
    writer = make_writer(service)
    writer._directory = tmp_path

    writer.write_segment_transition(transition())

    assert service.published == [(100, 400), (400, 500)]
    assert_contiguous_coverage(
        service.published,
        start=100,
        end=writer.retained_start_offset,
    )


def test_coverage_checker_detects_silently_omitted_unowned(tmp_path: Path) -> None:
    service = RecordingService(omit_unowned=True)
    writer = make_writer(service)
    writer._directory = tmp_path

    from brynse.publication_coverage import CoverageError

    with pytest.raises(CoverageError, match="missing|not published"):
        writer.write_segment_transition(transition())

    assert service.published == [(100, 400)]
    assert writer.retained_start_offset == 100


def test_phantom_unowned_publication_is_rejected(tmp_path: Path) -> None:
    """A returned pathname cannot prove that excluded audio was published."""
    from brynse.publication_coverage import CoverageError

    class PhantomUnownedService(RecordingService):
        def write_segment(
            self,
            *,
            byte_range,
            stem: str,
            directory: Path,
            **kwargs,
        ) -> Path:
            path = directory / f"{stem}.mp3"

            if stem.startswith("unowned-"):
                # Mutation: return apparent success without creating audio.
                return path

            path.write_bytes(b"published-track")
            self.published.append((byte_range.start_offset, byte_range.end_offset))
            return path

    service = PhantomUnownedService()
    writer = make_writer(service)
    writer._directory = tmp_path

    with pytest.raises(CoverageError, match="not published|missing"):
        writer.write_segment_transition(transition())

    assert service.published == [(100, 400)]
    assert writer.retained_start_offset == 100
    assert not list(tmp_path.glob("unowned-*.mp3"))
