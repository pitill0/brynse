from pathlib import Path
from types import SimpleNamespace

from fluxtuner_ripper.generic_output import GenericSegmentWriter
from fluxtuner_ripper.models import SplitDecision, SplitKind


class _OutputService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def write_segment(
        self,
        *,
        source: object,
        byte_range: object,
        directory: Path,
        stem: str,
        codec: str,
    ) -> Path:
        self.calls.append(
            {
                "source": source,
                "byte_range": byte_range,
                "directory": directory,
                "stem": stem,
                "codec": codec,
            }
        )
        return directory / f"{stem}.{codec}"


def _ingestor() -> object:
    frames = (
        SimpleNamespace(
            offset=0,
            length=100,
            time_seconds=0.0,
            samples=1,
            sample_rate=1,
        ),
        SimpleNamespace(
            offset=100,
            length=100,
            time_seconds=1.0,
            samples=1,
            sample_rate=1,
        ),
        SimpleNamespace(
            offset=200,
            length=100,
            time_seconds=2.0,
            samples=1,
            sample_rate=1,
        ),
        SimpleNamespace(
            offset=300,
            length=100,
            time_seconds=3.0,
            samples=1,
            sample_rate=1,
        ),
    )

    return SimpleNamespace(
        timeline=SimpleNamespace(frames=frames),
        ring_buffer=object(),
    )


def _resolution(
    *,
    incoming_start: int,
    outgoing_end: int,
) -> object:
    return SimpleNamespace(
        temporal=SimpleNamespace(
            incoming_start_seconds=incoming_start / 100,
        ),
        split=SplitDecision(
            kind=SplitKind.HARD_CUT,
            incoming_start=incoming_start,
            outgoing_end=outgoing_end,
        ),
    )


def test_generic_segment_writer_materializes_segments_and_final_tail(
    tmp_path: Path,
) -> None:
    ingestor = _ingestor()
    output_service = _OutputService()

    writer = GenericSegmentWriter(
        ingestor=ingestor,
        directory=tmp_path,
        codec="mp3",
        output_service=output_service,
    )

    written = writer.write(
        (
            _resolution(
                incoming_start=100,
                outgoing_end=100,
            ),
            _resolution(
                incoming_start=300,
                outgoing_end=300,
            ),
        )
    )

    assert len(written) == 3

    assert written[0].index == 1
    assert written[0].start_offset == 0
    assert written[0].end_offset == 100

    assert written[1].index == 2
    assert written[1].start_offset == 100
    assert written[1].end_offset == 300

    assert written[2].index == 3
    assert written[2].start_offset == 300
    assert written[2].end_offset == 400

    assert [call["stem"] for call in output_service.calls] == [
        "segment-0001",
        "segment-0002",
        "segment-0003",
    ]


def test_generic_segment_writer_without_boundaries_writes_whole_stream(
    tmp_path: Path,
) -> None:
    ingestor = _ingestor()
    output_service = _OutputService()

    writer = GenericSegmentWriter(
        ingestor=ingestor,
        directory=tmp_path,
        codec="mp3",
        output_service=output_service,
    )

    written = writer.write(())

    assert len(written) == 1
    assert written[0].start_offset == 0
    assert written[0].end_offset == 400

    byte_range = output_service.calls[0]["byte_range"]

    assert byte_range.start_offset == 0
    assert byte_range.end_offset == 400


def test_generic_segment_writer_does_not_create_empty_final_segment(
    tmp_path: Path,
) -> None:
    ingestor = _ingestor()
    output_service = _OutputService()

    writer = GenericSegmentWriter(
        ingestor=ingestor,
        directory=tmp_path,
        codec="mp3",
        output_service=output_service,
    )

    written = writer.write(
        (
            _resolution(
                incoming_start=400,
                outgoing_end=400,
            ),
        )
    )

    assert len(written) == 1
    assert written[0].start_offset == 0
    assert written[0].end_offset == 400


def test_generic_output_uses_segment_byte_range_contract() -> None:
    source = Path("src/fluxtuner_ripper/generic_output.py").read_text()

    assert "TrackByteRange" not in source
    assert "SegmentByteRange" in source


def test_generic_output_uses_segment_range_planner() -> None:
    source = Path("src/fluxtuner_ripper/generic_output.py").read_text()

    assert "TrackRangePlanner" not in source
    assert "SegmentRangePlanner" in source


def test_generic_output_uses_segment_output_service_contract() -> None:
    source = Path("src/fluxtuner_ripper/generic_output.py").read_text()

    assert "TrackOutputService" not in source
    assert "write_track" not in source
    assert "SegmentOutputService" in source
    assert "write_segment" in source
