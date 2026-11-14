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
        return Path("/tmp") / f"{stem}.mp3"


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


def test_exclusion_publishes_outgoing_then_unowned() -> None:
    service = RecordingOutputService()
    writer = make_writer(service)

    written = writer.write_segment_transition(transition(400, 500))

    assert len(service.calls) == 2
    assert service.calls[0][1:] == (100, 400)
    assert service.calls[1] == ("unowned-000000000400-000000000500", 400, 500)
    assert written.path.name.endswith(".mp3")
    assert not written.path.name.startswith("unowned-")
    assert writer.retained_start_offset == 500


def test_hard_cut_does_not_create_unowned_file() -> None:
    service = RecordingOutputService()
    writer = make_writer(service)

    writer.write_segment_transition(transition(400, 400))

    assert len(service.calls) == 1
    assert writer.retained_start_offset == 400


def test_unowned_failure_does_not_advance_retained_offset() -> None:
    service = RecordingOutputService(fail_unowned=True)
    writer = make_writer(service)

    with pytest.raises(OSError, match="injected"):
        writer.write_segment_transition(transition(400, 500))

    assert len(service.calls) == 2
    assert service.calls[0][1:] == (100, 400)
    assert service.calls[1][1:] == (400, 500)
    assert writer.retained_start_offset == 100
