"""F-2: AAC and metadata exclusions use the ordinary output pipeline."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from brynse.integrations.radio.session import RippingSession
from brynse.integrations.radio.session_output import SessionOutputWriter
from brynse.models import SplitDecision, SplitKind, TemporalSplitKind
from brynse.output import SegmentOutputService
from brynse.publication_coverage import PublicationCoverageLedger


class Source:
    def __init__(self) -> None:
        self.data = bytes(range(256)) * 4

    @property
    def end_offset(self) -> int:
        return len(self.data)

    def contains(self, start: int, end: int) -> bool:
        return 0 <= start <= end <= len(self.data)

    def read(self, start: int, end: int) -> bytes:
        return self.data[start:end]


class StreamingFinalizer:
    def finalize_stream(self, *, chunks, output_path: Path) -> None:
        with output_path.open("wb") as handle:
            for chunk in chunks:
                handle.write(chunk)


def make_writer(tmp_path: Path, codec: str) -> tuple[SessionOutputWriter, Source]:
    source = Source()
    timeline = SimpleNamespace(frames=[SimpleNamespace(offset=900, length=100)])
    ingestor = SimpleNamespace(timeline=timeline, ring_buffer=source)
    finalizer = StreamingFinalizer()
    service = SegmentOutputService(
        mp3_finalizer=finalizer,
        aac_finalizer=finalizer,
    )
    writer = SessionOutputWriter(
        ingestor=ingestor,
        directory=tmp_path,
        codec=codec,
        output_service=service,
        source=source,
    )
    return writer, source


def test_aac_unowned_uses_existing_m4a_publication(tmp_path: Path) -> None:
    writer, source = make_writer(tmp_path, "aac")
    decision = SplitDecision(
        kind=SplitKind.EXCLUSION,
        outgoing_end=400,
        incoming_start=500,
    )
    transition = SimpleNamespace(
        outgoing=SimpleNamespace(start_offset=100, label="Track A"),
        boundary=SimpleNamespace(split=decision),
    )

    written = writer.write_segment_transition(transition)

    assert written.path.suffix == ".m4a"
    assert written.path.read_bytes() == source.data[100:400]
    unowned = list(tmp_path.glob("unowned-*.m4a"))
    assert len(unowned) == 1
    assert unowned[0].read_bytes() == source.data[400:500]
    assert writer.retained_start_offset == 500


def test_metadata_only_exclusion_reaches_same_writer(tmp_path: Path) -> None:
    writer, source = make_writer(tmp_path, "mp3")
    decision = SplitDecision(
        kind=SplitKind.EXCLUSION,
        outgoing_end=400,
        incoming_start=500,
    )
    session = object.__new__(RippingSession)
    session._pending_exclusion = SimpleNamespace(
        start_time_seconds=20.0,
        lifetime_seconds=2.0,
    )
    session._ingestor = SimpleNamespace(timeline=writer._ingestor.timeline)
    expected_split = decision
    session._split_aligner = SimpleNamespace(align=lambda *, decision, timeline: expected_split)
    incoming = SimpleNamespace(start_time_seconds=22.0)

    boundary = session._resolve_pending_exclusion(incoming)

    assert boundary is not None
    assert boundary.temporal.kind is TemporalSplitKind.EXCLUSION
    assert boundary.split == decision
    assert session._pending_exclusion is None

    transition = SimpleNamespace(
        outgoing=SimpleNamespace(start_offset=100, label="Track A"),
        boundary=boundary,
    )
    written = writer.write_segment_transition(transition)

    assert written.path.read_bytes() == source.data[100:400]
    unowned = list(tmp_path.glob("unowned-*.mp3"))
    assert len(unowned) == 1
    assert unowned[0].read_bytes() == source.data[400:500]
    assert writer.retained_start_offset == 500


def test_unowned_event_is_separate_from_written_track(tmp_path: Path) -> None:
    from brynse.integrations.radio.session_output import WrittenUnownedRegion

    writer, source = make_writer(tmp_path, "mp3")
    events: list[WrittenUnownedRegion] = []
    writer._on_unowned_written = events.append

    decision = SplitDecision(
        kind=SplitKind.EXCLUSION,
        outgoing_end=400,
        incoming_start=500,
    )
    transition = SimpleNamespace(
        outgoing=SimpleNamespace(start_offset=100, label="Track A"),
        incoming=SimpleNamespace(label="Track B"),
        boundary=SimpleNamespace(split=decision),
    )

    track = writer.write_segment_transition(transition)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, WrittenUnownedRegion)
    assert event.transition is transition
    assert (event.byte_range.start_offset, event.byte_range.end_offset) == (400, 500)
    assert event.path.name == "unowned-000000000400-000000000500.mp3"
    assert event.path.read_bytes().endswith(source.data[400:500])
    assert track.path != event.path
    assert writer.retained_start_offset == 500


def test_hard_cut_emits_no_unowned_event(tmp_path: Path) -> None:
    writer, _ = make_writer(tmp_path, "mp3")
    events: list[object] = []
    writer._on_unowned_written = events.append

    transition = SimpleNamespace(
        outgoing=SimpleNamespace(start_offset=100, label="Track A"),
        boundary=SimpleNamespace(
            split=SplitDecision(
                kind=SplitKind.HARD_CUT,
                outgoing_end=400,
                incoming_start=400,
            )
        ),
    )

    writer.write_segment_transition(transition)
    assert events == []


def test_real_audio_publication_survives_provenance_failure(tmp_path: Path) -> None:
    """Both real outputs exist; a failed receipt cannot authorize discard."""
    writer, source = make_writer(tmp_path, "mp3")
    writer._coverage_ledger = PublicationCoverageLedger(start_offset=0)

    def fail_receipt(_receipt: object) -> None:
        raise OSError("injected durable receipt error")

    writer._on_publication_confirmed = fail_receipt
    transition = SimpleNamespace(
        outgoing=SimpleNamespace(start_offset=0, label="Track A"),
        boundary=SimpleNamespace(
            split=SplitDecision(
                kind=SplitKind.EXCLUSION,
                outgoing_end=400,
                incoming_start=500,
            )
        ),
    )
    with pytest.raises(OSError, match="injected durable receipt error"):
        writer.write_segment_transition(transition)

    assert writer.retained_start_offset is None
    assert len(list(tmp_path.glob("unowned-*.mp3"))) == 1
    assert (tmp_path / "Track A.mp3").read_bytes() == source.data[:400]
    assert list(tmp_path.glob("unowned-*.mp3"))[0].read_bytes() == source.data[400:500]
    assert writer.publication_coverage is not None
    assert writer.publication_coverage.covered_end == 500
