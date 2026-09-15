from pathlib import Path
from types import SimpleNamespace

import pytest

from fluxtuner_ripper.models import SplitDecision, SplitKind
from fluxtuner_ripper.streaming_sink import StreamingSegmentSink


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

    def write_track(
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
