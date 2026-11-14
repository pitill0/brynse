"""F-2: publishing excluded audio must precede spool advancement."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from brynse.integrations.radio.session_output import SessionOutputWriter
from brynse.models import SplitKind
from brynse.output import SegmentRangePlanner


class RecordingOutputService:
    def __init__(self, *, fail_unowned: bool = False) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.fail_unowned = fail_unowned

    def write_segment(self, *, byte_range, stem, **kwargs):
        self.calls.append((stem, byte_range.start_offset, byte_range.end_offset))
        if self.fail_unowned and stem.startswith("unowned-"):
            raise OSError("injected unowned publication failure")
        path = kwargs["directory"] / f"{stem}.mp3"
        path.write_bytes(b"simulated-published-audio")
        return path


def make_writer(service: RecordingOutputService) -> SessionOutputWriter:
    writer = object.__new__(SessionOutputWriter)
    writer._ingestor = SimpleNamespace(
        timeline=SimpleNamespace(frames=[SimpleNamespace(offset=900, length=100)]),
        ring_buffer=object(),
    )
    writer._source = object()
    writer._directory = Path("/tmp/f2-test-output")
    writer._codec = "mp3"
    writer._metadata_order = "preserve"
    writer._range_planner = SegmentRangePlanner()
    writer._output_service = service
    writer._current_start_offset = 100
    return writer


def transition(outgoing_end: int, incoming_start: int):
    return SimpleNamespace(
        outgoing=SimpleNamespace(start_offset=100, label="Song A"),
        boundary=SimpleNamespace(
            split=SimpleNamespace(
                kind=SplitKind.HARD_CUT,
                outgoing_end=outgoing_end,
                incoming_start=incoming_start,
            )
        ),
    )


def test_exclusion_publishes_outgoing_then_unowned(tmp_path: Path) -> None:
    service = RecordingOutputService()
    writer = make_writer(service)
    writer._directory = tmp_path

    written = writer.write_segment_transition(transition(400, 500))

    assert len(service.calls) == 2
    assert service.calls[0][1:] == (100, 400)
    assert service.calls[1] == ("unowned-000000000400-000000000500", 400, 500)
    assert written.path.name.endswith(".mp3")
    assert not written.path.name.startswith("unowned-")
    assert writer.retained_start_offset == 500


def test_hard_cut_does_not_create_unowned_file(tmp_path: Path) -> None:
    service = RecordingOutputService()
    writer = make_writer(service)
    writer._directory = tmp_path

    writer.write_segment_transition(transition(400, 400))

    assert len(service.calls) == 1
    assert writer.retained_start_offset == 400


def test_unowned_failure_does_not_advance_retained_offset(tmp_path: Path) -> None:
    service = RecordingOutputService(fail_unowned=True)
    writer = make_writer(service)
    writer._directory = tmp_path

    with pytest.raises(OSError, match="injected"):
        writer.write_segment_transition(transition(400, 500))

    assert len(service.calls) == 2
    assert service.calls[0][1:] == (100, 400)
    assert service.calls[1][1:] == (400, 500)
    assert writer.retained_start_offset == 100


def test_publication_receipt_contains_exact_ranges(tmp_path: Path) -> None:
    """The writer reports both confirmed outputs before offset advancement."""
    service = RecordingOutputService()
    writer = make_writer(service)
    writer._directory = tmp_path

    receipts = []

    def record(receipt) -> None:
        assert writer.retained_start_offset == 100
        receipts.append(receipt)

    writer._on_publication_confirmed = record

    writer.write_segment_transition(transition(400, 500))

    assert len(receipts) == 1
    receipt = receipts[0]

    assert (receipt.outgoing_range.start_offset, receipt.outgoing_range.end_offset) == (
        100,
        400,
    )
    assert receipt.unowned_range is not None
    assert (receipt.unowned_range.start_offset, receipt.unowned_range.end_offset) == (
        400,
        500,
    )
    assert receipt.incoming_start == 500
    assert receipt.outgoing_path.is_file()
    assert receipt.unowned_path is not None
    assert receipt.unowned_path.is_file()
    assert writer.retained_start_offset == 500


def test_receipt_failure_blocks_spool_advancement(tmp_path: Path) -> None:
    """A failed provenance operation must not authorize source-byte discard."""
    service = RecordingOutputService()
    writer = make_writer(service)
    writer._directory = tmp_path

    def fail_receipt(_receipt) -> None:
        raise OSError("injected provenance publication failure")

    writer._on_publication_confirmed = fail_receipt

    with pytest.raises(OSError, match="injected provenance"):
        writer.write_segment_transition(transition(400, 500))

    # Audio files may already exist, but the spool cannot be discarded.
    assert len(service.calls) == 2
    assert writer.retained_start_offset == 100


def test_terminal_coverage_without_transitions(tmp_path: Path) -> None:
    """All received audio remains explicitly retained when nothing was published."""
    service = RecordingOutputService()
    writer = make_writer(service)
    writer._directory = tmp_path
    writer._current_start_offset = None

    tail = writer.finalize_coverage(spool_start=0, spool_end=1000)

    assert tail is not None
    assert (tail.start, tail.end) == (0, 1000)
    assert writer._coverage_ledger is not None
    assert writer._coverage_ledger.published == ()
    writer._coverage_ledger.verify_complete(end_offset=1000)


def test_terminal_coverage_after_confirmed_exclusion(tmp_path: Path) -> None:
    """Published track and exclusion precede the exact retained terminal tail."""
    service = RecordingOutputService()
    writer = make_writer(service)
    writer._directory = tmp_path
    writer._current_start_offset = 0

    writer.write_segment_transition(transition(400, 500))

    assert writer.retained_start_offset == 500

    tail = writer.finalize_coverage(spool_start=500, spool_end=1000)

    assert tail is not None
    assert (tail.start, tail.end) == (500, 1000)
    assert writer._coverage_ledger is not None
    assert [
        (entry.kind, entry.start, entry.end) for entry in writer._coverage_ledger.published
    ] == [
        ("track", 0, 400),
        ("unowned", 400, 500),
    ]
    writer._coverage_ledger.verify_complete(end_offset=1000)


def test_terminal_coverage_rejects_unaccounted_session_head(
    tmp_path: Path,
) -> None:
    """Starting the first publication at byte 100 cannot hide bytes [0,100)."""
    from brynse.publication_coverage import CoverageError

    service = RecordingOutputService()
    writer = make_writer(service)
    writer._directory = tmp_path

    # Existing fixture initializes the outgoing range at byte 100.
    writer.write_segment_transition(transition(400, 500))

    with pytest.raises(CoverageError, match="head|initial|unaccounted"):
        writer.finalize_coverage(spool_start=500, spool_end=1000)


def test_real_writer_rejects_initial_gap_before_any_publication(tmp_path: Path) -> None:
    """The runner's zero-based ledger refuses missing head before writing."""
    from brynse.publication_coverage import CoverageError, PublicationCoverageLedger

    service = RecordingOutputService()
    writer = make_writer(service)
    writer._directory = tmp_path
    writer._coverage_ledger = PublicationCoverageLedger(start_offset=0)
    with pytest.raises(CoverageError, match="unaccounted initial head"):
        writer.write_segment_transition(transition(400, 500))
    assert service.calls == []
    assert writer.retained_start_offset == 100
