from pathlib import Path
from types import SimpleNamespace

import pytest

from brynse.models import SplitDecision, SplitKind
from brynse.streaming_sink import StreamingSegmentSink


class _Ring:
    def __init__(
        self,
        *,
        start_offset: int,
        end_offset: int,
    ) -> None:
        self.start_offset = start_offset
        self.end_offset = end_offset

    def contains(self, start: int, end: int) -> bool:
        return self.start_offset <= start and end <= self.end_offset


class _OutputService:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def write_segment(
        self,
        *,
        source: object,
        byte_range: object,
        directory: Path,
        stem: str,
        codec: str,
    ) -> Path:
        self.calls.append(byte_range)
        return directory / f"{stem}.{codec}"


def _ingestor() -> object:
    return SimpleNamespace(
        timeline=SimpleNamespace(
            frames=(
                SimpleNamespace(offset=100, length=100),
                SimpleNamespace(offset=200, length=100),
                SimpleNamespace(offset=300, length=100),
                SimpleNamespace(offset=400, length=100),
            )
        ),
        ring_buffer=_Ring(
            start_offset=100,
            end_offset=500,
        ),
    )


def _resolution(
    *,
    incoming_start: int,
    outgoing_end: int,
    kind: SplitKind = SplitKind.HARD_CUT,
) -> object:
    return SimpleNamespace(
        split=SplitDecision(
            kind=kind,
            incoming_start=incoming_start,
            outgoing_end=outgoing_end,
        )
    )


def test_streaming_segment_sink_writes_closed_segment_immediately(
    tmp_path: Path,
) -> None:
    ingestor = _ingestor()
    output_service = _OutputService()

    sink = StreamingSegmentSink(
        ingestor=ingestor,  # type: ignore[arg-type]
        directory=tmp_path,
        codec="mp3",
        output_service=output_service,  # type: ignore[arg-type]
    )

    segment = sink.accept(
        _resolution(
            incoming_start=300,
            outgoing_end=300,
        )  # type: ignore[arg-type]
    )

    assert segment.index == 1
    assert segment.start_offset == 100
    assert segment.end_offset == 300
    assert sink.current_start_offset == 300

    assert len(output_service.calls) == 1
    assert output_service.calls[0].start_offset == 100
    assert output_service.calls[0].end_offset == 300


def test_streaming_segment_sink_chains_multiple_boundaries(
    tmp_path: Path,
) -> None:
    ingestor = _ingestor()
    output_service = _OutputService()

    sink = StreamingSegmentSink(
        ingestor=ingestor,  # type: ignore[arg-type]
        directory=tmp_path,
        codec="mp3",
        output_service=output_service,  # type: ignore[arg-type]
    )

    first = sink.accept(
        _resolution(
            incoming_start=200,
            outgoing_end=200,
        )  # type: ignore[arg-type]
    )
    second = sink.accept(
        _resolution(
            incoming_start=400,
            outgoing_end=400,
        )  # type: ignore[arg-type]
    )

    assert (first.start_offset, first.end_offset) == (100, 200)
    assert (second.start_offset, second.end_offset) == (200, 400)
    assert second.index == 2
    assert sink.current_start_offset == 400


def test_streaming_segment_sink_preserves_crossfade_overlap(
    tmp_path: Path,
) -> None:
    ingestor = _ingestor()
    output_service = _OutputService()

    sink = StreamingSegmentSink(
        ingestor=ingestor,  # type: ignore[arg-type]
        directory=tmp_path,
        codec="mp3",
        output_service=output_service,  # type: ignore[arg-type]
    )

    segment = sink.accept(
        _resolution(
            incoming_start=250,
            outgoing_end=300,
            kind=SplitKind.CROSSFADE,
        )  # type: ignore[arg-type]
    )

    assert segment.start_offset == 100
    assert segment.end_offset == 300
    assert sink.current_start_offset == 250


def test_streaming_segment_sink_finalize_writes_open_tail(
    tmp_path: Path,
) -> None:
    ingestor = _ingestor()
    output_service = _OutputService()

    sink = StreamingSegmentSink(
        ingestor=ingestor,  # type: ignore[arg-type]
        directory=tmp_path,
        codec="mp3",
        output_service=output_service,  # type: ignore[arg-type]
    )

    sink.accept(
        _resolution(
            incoming_start=300,
            outgoing_end=300,
        )  # type: ignore[arg-type]
    )

    tail = sink.finalize()

    assert tail is not None
    assert tail.index == 2
    assert tail.start_offset == 300
    assert tail.end_offset == 500

    assert sink.finalize() is None


def test_streaming_segment_sink_rejects_evicted_segment(
    tmp_path: Path,
) -> None:
    ingestor = _ingestor()
    ingestor.ring_buffer.start_offset = 250

    sink = StreamingSegmentSink(
        ingestor=ingestor,  # type: ignore[arg-type]
        directory=tmp_path,
        codec="mp3",
        output_service=_OutputService(),  # type: ignore[arg-type]
    )

    with pytest.raises(
        RuntimeError,
        match="no longer retained",
    ):
        sink.accept(
            _resolution(
                incoming_start=300,
                outgoing_end=300,
            )  # type: ignore[arg-type]
        )


def test_streaming_segment_sink_uses_spool_and_compacts_after_boundary(
    tmp_path: Path,
) -> None:
    from brynse.streaming_spool import StreamingSpool

    ingestor = _ingestor()
    output_service = _OutputService()
    spool = StreamingSpool(directory=tmp_path / "spool")

    try:
        spool.append(b"x" * 500)

        # Simulate the analysis ring having already evicted the first
        # part of the still-open segment.
        ingestor.ring_buffer.start_offset = 250

        sink = StreamingSegmentSink(
            ingestor=ingestor,  # type: ignore[arg-type]
            directory=tmp_path / "output",
            codec="mp3",
            spool=spool,
            output_service=output_service,  # type: ignore[arg-type]
        )

        segment = sink.accept(
            _resolution(
                incoming_start=300,
                outgoing_end=300,
            )  # type: ignore[arg-type]
        )

        assert segment.start_offset == 100
        assert segment.end_offset == 300

        # The ring no longer had [100, 300), so successful output proves
        # materialization came from the spool.
        assert ingestor.ring_buffer.start_offset == 250

        # The closed prefix has been discarded. The currently open segment
        # begins exactly at the incoming boundary.
        assert spool.start_offset == 300
        assert spool.end_offset == 500
        assert spool.retained_bytes == 200
    finally:
        spool.close()


def test_streaming_segment_sink_spool_preserves_crossfade_overlap(
    tmp_path: Path,
) -> None:
    from brynse.streaming_spool import StreamingSpool

    ingestor = _ingestor()
    output_service = _OutputService()
    spool = StreamingSpool(directory=tmp_path / "spool")

    try:
        spool.append(b"x" * 500)

        sink = StreamingSegmentSink(
            ingestor=ingestor,  # type: ignore[arg-type]
            directory=tmp_path / "output",
            codec="mp3",
            spool=spool,
            output_service=output_service,  # type: ignore[arg-type]
        )

        segment = sink.accept(
            _resolution(
                incoming_start=250,
                outgoing_end=300,
                kind=SplitKind.CROSSFADE,
            )  # type: ignore[arg-type]
        )

        assert segment.start_offset == 100
        assert segment.end_offset == 300

        # [250, 300) belonged to the outgoing track but must remain
        # available for the incoming track too.
        assert spool.start_offset == 250
        assert spool.end_offset == 500
        assert spool.read(250, 300) == b"x" * 50
    finally:
        spool.close()


def test_streaming_segment_sink_preserves_explicit_initial_start_offset(
    tmp_path: Path,
) -> None:
    ingestor = _ingestor()
    output_service = _OutputService()

    # Simulate a timeline already pruned by bounded ring retention.
    ingestor.timeline.frames = ingestor.timeline.frames[1:]

    sink = StreamingSegmentSink(
        ingestor=ingestor,  # type: ignore[arg-type]
        directory=tmp_path,
        codec="mp3",
        initial_start_offset=100,
        output_service=output_service,  # type: ignore[arg-type]
    )

    segment = sink.accept(
        _resolution(
            incoming_start=300,
            outgoing_end=300,
        )  # type: ignore[arg-type]
    )

    assert segment.start_offset == 100
    assert segment.end_offset == 300


def test_streaming_segment_sink_resolves_initial_start_offset_lazily(
    tmp_path: Path,
) -> None:
    class _InitialStartSource:
        first_frame_offset: int | None = None

    ingestor = _ingestor()
    output_service = _OutputService()
    start_source = _InitialStartSource()

    sink = StreamingSegmentSink(
        ingestor=ingestor,  # type: ignore[arg-type]
        directory=tmp_path,
        codec="mp3",
        initial_start_source=start_source,
        output_service=output_service,  # type: ignore[arg-type]
    )

    # The sink already exists, but ingestion discovers the actual first
    # encoded frame only afterwards.
    start_source.first_frame_offset = 100

    segment = sink.accept(
        _resolution(
            incoming_start=300,
            outgoing_end=300,
        )  # type: ignore[arg-type]
    )

    assert segment.start_offset == 100
    assert segment.end_offset == 300


def test_streaming_sink_uses_generic_segment_range_contracts() -> None:
    source = Path("src/brynse/streaming_sink.py").read_text()

    assert "TrackByteRange" not in source
    assert "TrackRangePlanner" not in source
    assert "SegmentByteRange" in source
    assert "SegmentRangePlanner" in source


def test_streaming_sink_uses_segment_output_service_contract() -> None:
    source = Path("src/brynse/streaming_sink.py").read_text()

    assert "TrackOutputService" not in source
    assert "create_safe_track_output_service" not in source
    assert "write_track" not in source

    assert "SegmentOutputService" in source
    assert "create_safe_segment_output_service" in source
    assert "write_segment" in source
