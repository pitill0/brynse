from __future__ import annotations

from pathlib import Path

import pytest

from brynse.models import Segment
from brynse.integrations.radio.runner import (
    RippingRunConfig,
    RippingRunError,
    RippingRunner,
    resolve_codec,
    resolve_metaint,
)
from brynse.source import BinaryIOStreamSource


def _stream_source(stream: object, metadata: dict[str, str]) -> BinaryIOStreamSource:
    return BinaryIOStreamSource(
        stream,  # type: ignore[arg-type]
        metadata=metadata,
    )


def test_runner_rejects_non_positive_ring(tmp_path: Path) -> None:
    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            ring_max_bytes=0,
        ),
    )

    with pytest.raises(RippingRunError, match="ring_max_bytes"):
        runner.run()


def test_runner_codec_resolution_matches_supported_content_types() -> None:
    assert resolve_codec("auto", {"Content-Type": "audio/mpeg"}) == "mp3"
    assert resolve_codec("auto", {"Content-Type": "audio/aacp"}) == "aac"


def test_runner_rejects_missing_metaint() -> None:
    with pytest.raises(RippingRunError, match="icy-metaint"):
        resolve_metaint({})


def test_runner_treats_read_failure_after_stop_as_clean_shutdown(
    tmp_path: Path,
) -> None:
    class StopRaceStream:
        def __init__(self) -> None:
            self.runner: RippingRunner | None = None
            self.closed = False

        def read(self, size: int = -1) -> bytes:
            assert self.runner is not None
            self.runner.stop()
            raise AttributeError("'NoneType' object has no attribute 'read'")

        def close(self) -> None:
            self.closed = True

    stream = StopRaceStream()
    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
        ),
        stream_opener=lambda url: _stream_source(
            stream,
            {"Content-Type": "audio/mpeg", "icy-metaint": "417"},
        ),
    )
    stream.runner = runner

    result = runner.run()

    assert result.stopped is True
    assert stream.closed is True


def test_runner_propagates_read_failure_without_stop(
    tmp_path: Path,
) -> None:
    class FailingStream:
        def read(self, size: int = -1) -> bytes:
            raise RuntimeError("network read failed")

        def close(self) -> None:
            pass

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
        ),
        stream_opener=lambda url: _stream_source(
            FailingStream(),
            {"Content-Type": "audio/mpeg", "icy-metaint": "417"},
        ),
    )

    with pytest.raises(RuntimeError, match="network read failed"):
        runner.run()


def test_runner_shadow_callback_is_parallel_and_optional(tmp_path: Path) -> None:
    frame_length = 417
    frame = b"\xff\xfb\x90\x00" + bytes(frame_length - 4)

    class FakeStream:
        def __init__(self) -> None:
            self._chunks = [frame + b"\x00", b""]
            self.closed = False

        def read(self, size: int = -1) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            self.closed = True

    class FakeObserver:
        def __init__(self) -> None:
            self.calls = 0

        def observe(self, **kwargs: object) -> tuple[object, ...]:
            self.calls += 1
            return ("shadow-analysis",)

    observer = FakeObserver()
    factory_calls: list[str] = []
    callbacks: list[object] = []
    stream = FakeStream()

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
        ),
        stream_opener=lambda url: _stream_source(
            stream,
            {"Content-Type": "audio/mpeg", "icy-metaint": str(frame_length)},
        ),
        shadow_observer_factory=lambda ffmpeg_binary: (
            factory_calls.append(ffmpeg_binary) or observer
        ),
    )

    result = runner.run(on_shadow_analysis=callbacks.append)

    assert result.stopped is False
    assert factory_calls == ["ffmpeg"]
    assert observer.calls == 1
    assert callbacks == ["shadow-analysis"]
    assert stream.closed is True


def test_runner_retains_clean_audio_in_safe_streaming_spool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import brynse.integrations.radio.runner as runner_module

    frame_length = 417
    frame = b"\xff\xfb\x90\x00" + bytes(frame_length - 4)

    class FakeStream:
        def __init__(self) -> None:
            self._chunks = [frame + b"\x00", b""]

        def read(self, size: int = -1) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            pass

    class FakeSpool:
        def __init__(self) -> None:
            self.appended: list[bytes] = []
            self.closed = False

        def append(self, data: bytes) -> tuple[int, int]:
            start = sum(len(chunk) for chunk in self.appended)
            self.appended.append(data)
            return start, start + len(data)

        def close(self) -> None:
            self.closed = True

    spool = FakeSpool()
    spool_directories: list[Path] = []

    def fake_spool_factory(*, directory: Path):
        spool_directories.append(directory)
        return spool

    monkeypatch.setattr(
        runner_module,
        "create_safe_streaming_spool",
        fake_spool_factory,
    )

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
        ),
        stream_opener=lambda url: _stream_source(
            FakeStream(),
            {
                "Content-Type": "audio/mpeg",
                "icy-metaint": str(frame_length),
            },
        ),
    )

    result = runner.run()

    assert result.stopped is False
    assert spool_directories == [tmp_path / ".fluxtuner-spool"]
    assert spool.appended == [frame]
    assert spool.closed is True


def test_runner_discards_spool_before_next_retained_track_offset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import brynse.integrations.radio.runner as runner_module

    class FakeStream:
        def __init__(self) -> None:
            self._chunks = [b"chunk", b""]

        def read(self, size: int = -1) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            pass

    class FakeSpool:
        def __init__(self) -> None:
            self.discarded: list[int] = []

        def append(self, data: bytes) -> tuple[int, int]:
            return 0, len(data)

        def discard_before(self, offset: int) -> None:
            self.discarded.append(offset)

        def close(self) -> None:
            pass

    class FakeIngestResult:
        audio = b"clean-audio"
        timed_metadata_events = ()

    class FakeTransition:
        pass

    class FakeSessionResult:
        ingest = FakeIngestResult()
        transitions = (FakeTransition(),)

    class FakeSession:
        current_track = None

        def feed(self, chunk: bytes) -> FakeSessionResult:
            return FakeSessionResult()

    class FakeOutputWriter:
        retained_start_offset = 4800

        def __init__(self, **kwargs) -> None:
            pass

        def write_transition(self, transition):
            return object()

    spool = FakeSpool()

    monkeypatch.setattr(
        runner_module,
        "create_safe_streaming_spool",
        lambda *, directory: spool,
    )
    monkeypatch.setattr(
        runner_module,
        "SessionOutputWriter",
        FakeOutputWriter,
    )
    monkeypatch.setattr(
        runner_module,
        "RippingSession",
        lambda **kwargs: FakeSession(),
    )

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
        ),
        stream_opener=lambda url: _stream_source(
            FakeStream(),
            {
                "Content-Type": "audio/mpeg",
                "icy-metaint": "1",
            },
        ),
    )

    runner.run()

    assert spool.discarded == [4800]


def test_runner_closes_spool_when_output_writer_construction_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import brynse.integrations.radio.runner as runner_module

    class FakeStream:
        def read(self, size: int = -1) -> bytes:
            return b""

        def close(self) -> None:
            pass

    class FakeSpool:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeSession:
        current_track = None

    spool = FakeSpool()

    monkeypatch.setattr(
        runner_module,
        "create_safe_streaming_spool",
        lambda *, directory: spool,
    )
    monkeypatch.setattr(
        runner_module,
        "RippingSession",
        lambda **kwargs: FakeSession(),
    )

    def fail_output_writer(**kwargs):
        raise RuntimeError("output writer construction failed")

    monkeypatch.setattr(
        runner_module,
        "SessionOutputWriter",
        fail_output_writer,
    )

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
        ),
        stream_opener=lambda url: _stream_source(
            FakeStream(),
            {
                "Content-Type": "audio/mpeg",
                "icy-metaint": "1",
            },
        ),
    )

    with pytest.raises(RuntimeError, match="output writer construction failed"):
        runner.run()

    assert spool.closed is True


def test_runner_closes_spool_when_append_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import brynse.integrations.radio.runner as runner_module

    class FakeStream:
        def __init__(self) -> None:
            self._chunks = [b"chunk", b""]

        def read(self, size: int = -1) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            pass

    class FakeSpool:
        def __init__(self) -> None:
            self.closed = False

        def append(self, data: bytes) -> tuple[int, int]:
            raise RuntimeError("spool append failed")

        def close(self) -> None:
            self.closed = True

    class FakeIngestResult:
        audio = b"clean-audio"
        timed_metadata_events = ()

    class FakeSessionResult:
        ingest = FakeIngestResult()
        transitions = ()

    class FakeSession:
        current_track = None

        def feed(self, chunk: bytes) -> FakeSessionResult:
            return FakeSessionResult()

    class FakeOutputWriter:
        def __init__(self, **kwargs) -> None:
            pass

    spool = FakeSpool()

    monkeypatch.setattr(
        runner_module,
        "create_safe_streaming_spool",
        lambda *, directory: spool,
    )
    monkeypatch.setattr(
        runner_module,
        "RippingSession",
        lambda **kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        runner_module,
        "SessionOutputWriter",
        FakeOutputWriter,
    )

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
        ),
        stream_opener=lambda url: _stream_source(
            FakeStream(),
            {
                "Content-Type": "audio/mpeg",
                "icy-metaint": "1",
            },
        ),
    )

    with pytest.raises(RuntimeError, match="spool append failed"):
        runner.run()

    assert spool.closed is True


def test_runner_does_not_discard_spool_when_track_write_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import brynse.integrations.radio.runner as runner_module

    class FakeStream:
        def __init__(self) -> None:
            self._chunks = [b"chunk", b""]

        def read(self, size: int = -1) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            pass

    class FakeSpool:
        def __init__(self) -> None:
            self.closed = False
            self.discarded: list[int] = []

        def append(self, data: bytes) -> tuple[int, int]:
            return 0, len(data)

        def discard_before(self, offset: int) -> None:
            self.discarded.append(offset)

        def close(self) -> None:
            self.closed = True

    class FakeIngestResult:
        audio = b"clean-audio"
        timed_metadata_events = ()

    class FakeTransition:
        pass

    class FakeSessionResult:
        ingest = FakeIngestResult()
        transitions = (FakeTransition(),)

    class FakeSession:
        current_track = None

        def feed(self, chunk: bytes) -> FakeSessionResult:
            return FakeSessionResult()

    class FakeOutputWriter:
        retained_start_offset = 4800

        def __init__(self, **kwargs) -> None:
            pass

        def write_transition(self, transition):
            raise RuntimeError("track write failed")

    spool = FakeSpool()

    monkeypatch.setattr(
        runner_module,
        "create_safe_streaming_spool",
        lambda *, directory: spool,
    )
    monkeypatch.setattr(
        runner_module,
        "RippingSession",
        lambda **kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        runner_module,
        "SessionOutputWriter",
        FakeOutputWriter,
    )

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
        ),
        stream_opener=lambda url: _stream_source(
            FakeStream(),
            {
                "Content-Type": "audio/mpeg",
                "icy-metaint": "1",
            },
        ),
    )

    with pytest.raises(RuntimeError, match="track write failed"):
        runner.run()

    assert spool.discarded == []
    assert spool.closed is True


def test_runner_closes_spool_when_discard_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import brynse.integrations.radio.runner as runner_module

    class FakeStream:
        def __init__(self) -> None:
            self._chunks = [b"chunk", b""]

        def read(self, size: int = -1) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            pass

    class FakeSpool:
        def __init__(self) -> None:
            self.closed = False

        def append(self, data: bytes) -> tuple[int, int]:
            return 0, len(data)

        def discard_before(self, offset: int) -> None:
            raise RuntimeError("spool discard failed")

        def close(self) -> None:
            self.closed = True

    class FakeIngestResult:
        audio = b"clean-audio"
        timed_metadata_events = ()

    class FakeTransition:
        pass

    class FakeSessionResult:
        ingest = FakeIngestResult()
        transitions = (FakeTransition(),)

    class FakeSession:
        current_track = None

        def feed(self, chunk: bytes) -> FakeSessionResult:
            return FakeSessionResult()

    class FakeOutputWriter:
        retained_start_offset = 4800

        def __init__(self, **kwargs) -> None:
            pass

        def write_transition(self, transition):
            return object()

    spool = FakeSpool()
    written_callbacks: list[object] = []

    monkeypatch.setattr(
        runner_module,
        "create_safe_streaming_spool",
        lambda *, directory: spool,
    )
    monkeypatch.setattr(
        runner_module,
        "RippingSession",
        lambda **kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        runner_module,
        "SessionOutputWriter",
        FakeOutputWriter,
    )

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
        ),
        stream_opener=lambda url: _stream_source(
            FakeStream(),
            {
                "Content-Type": "audio/mpeg",
                "icy-metaint": "1",
            },
        ),
    )

    with pytest.raises(RuntimeError, match="spool discard failed"):
        runner.run(
            on_track_written=written_callbacks.append,
        )

    assert written_callbacks == []
    assert spool.closed is True


def test_open_stream_uses_interoperable_icy_request_headers(
    monkeypatch,
) -> None:
    from io import BytesIO

    import brynse.integrations.radio.runner as runner

    captured_request = None

    class FakeResponse(BytesIO):
        headers = {
            "Content-Type": "audio/mpeg",
            "icy-metaint": "16000",
        }

    def fake_urlopen(request, timeout):
        nonlocal captured_request
        captured_request = request
        assert timeout == 20
        return FakeResponse(b"radio-audio")

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    stream, headers = runner.open_stream("https://example.invalid/stream")

    try:
        assert captured_request is not None
        assert captured_request.get_header("Icy-metadata") == "1"
        assert captured_request.get_header("User-agent") == "Mozilla/5.0"
        assert headers["icy-metaint"] == "16000"
    finally:
        stream.close()


def test_open_stream_wraps_remote_disconnect_as_ripping_run_error(
    monkeypatch,
) -> None:
    import http.client

    import pytest

    import brynse.integrations.radio.runner as runner

    def fake_urlopen(request, timeout):
        raise http.client.RemoteDisconnected("Remote end closed connection without response")

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(
        runner.RippingRunError,
        match="could not open stream",
    ):
        runner.open_stream("https://example.invalid/stream")


def test_open_stream_source_wraps_radio_stream_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from io import BytesIO

    import brynse.integrations.radio.runner as runner

    stream = BytesIO(b"radio-audio")
    headers = {
        "content-type": "audio/mpeg",
        "icy-metaint": "16000",
    }

    monkeypatch.setattr(
        runner,
        "open_stream",
        lambda url: (stream, headers),
    )

    source = runner.open_stream_source(
        "https://example.invalid/stream",
    )

    assert source.read(5) == b"radio"
    assert source.metadata == headers

    source.close()
    assert stream.closed


def test_ripping_runner_consumes_stream_source_directly(tmp_path: Path) -> None:
    from collections.abc import Mapping

    from brynse.integrations.radio.runner import RippingRunConfig, RippingRunner

    class FakeSource:
        def __init__(self) -> None:
            self.closed = False

        @property
        def metadata(self) -> Mapping[str, str]:
            return {
                "Content-Type": "audio/mpeg",
                "icy-metaint": "16000",
            }

        def read(self, max_bytes: int) -> bytes:
            assert max_bytes > 0
            return b""

        def close(self) -> None:
            self.closed = True

    source = FakeSource()

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
        ),
        stream_opener=lambda url: source,
    )

    result = runner.run()

    assert result.codec == "mp3"
    assert result.metaint == 16000
    assert result.stopped is False
    assert source.closed


# --- autonomous boundary promotion wiring ---


def test_runner_does_not_create_shadow_observer_when_autonomous_boundaries_are_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import brynse.integrations.radio.runner as runner_module

    class FakeStream:
        metadata = {
            "Content-Type": "audio/mpeg",
            "icy-metaint": "1",
        }

        def __init__(self) -> None:
            self._chunks = [b"chunk", b""]

        def read(self, size: int = -1) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            pass

    class FakeIngestResult:
        audio = b""
        timed_metadata_events = ()

    class FakeSessionResult:
        ingest = FakeIngestResult()
        transitions = ()

    class FakeSession:
        current_track = None

        def __init__(self, **kwargs) -> None:
            pass

        def feed(self, chunk: bytes) -> FakeSessionResult:
            return FakeSessionResult()

    observer_factory_calls: list[str] = []

    def observer_factory(ffmpeg_binary: str):
        observer_factory_calls.append(ffmpeg_binary)
        raise AssertionError("shadow observer must not be created")

    monkeypatch.setattr(
        runner_module,
        "RippingSession",
        FakeSession,
    )

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
            autonomous_boundaries=False,
        ),
        stream_opener=lambda url: FakeStream(),
        shadow_observer_factory=observer_factory,
    )

    result = runner.run()

    assert result.stopped is False
    assert observer_factory_calls == []


def test_runner_creates_shadow_observer_when_autonomous_boundaries_are_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import brynse.integrations.radio.runner as runner_module

    class FakeStream:
        metadata = {
            "Content-Type": "audio/mpeg",
            "icy-metaint": "1",
        }

        def __init__(self) -> None:
            self._chunks = [b"chunk", b""]

        def read(self, size: int = -1) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            pass

    class FakeIngestResult:
        audio = b""
        timed_metadata_events = ()

    class FakeSessionResult:
        ingest = FakeIngestResult()
        transitions = ()

    class FakeSession:
        current_track = None

        def __init__(self, **kwargs) -> None:
            pass

        def feed(self, chunk: bytes) -> FakeSessionResult:
            return FakeSessionResult()

    class FakeObserver:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def observe(self, **kwargs):
            self.calls.append(kwargs)
            return ()

    observer = FakeObserver()
    observer_factory_calls: list[str] = []

    def observer_factory(ffmpeg_binary: str):
        observer_factory_calls.append(ffmpeg_binary)
        return observer

    monkeypatch.setattr(
        runner_module,
        "RippingSession",
        FakeSession,
    )

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
            ffmpeg_binary="test-ffmpeg",
            autonomous_boundaries=True,
        ),
        stream_opener=lambda url: FakeStream(),
        shadow_observer_factory=observer_factory,
    )

    result = runner.run()

    assert result.stopped is False
    assert observer_factory_calls == ["test-ffmpeg"]
    assert len(observer.calls) == 1
    assert "timeline" in observer.calls[0]
    assert "ring_buffer" in observer.calls[0]


def test_runner_materializes_promoted_autonomous_segment_and_advances_spool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import brynse.integrations.radio.runner as runner_module

    analysis = object()

    class FakeSplit:
        incoming_start = 4000

    class FakeResolution:
        split = FakeSplit()

    resolution = FakeResolution()
    transition = object()
    written_segment = object()

    class FakeStream:
        metadata = {
            "Content-Type": "audio/mpeg",
            "icy-metaint": "1",
        }

        def __init__(self) -> None:
            self._chunks = [b"chunk", b""]

        def read(self, size: int = -1) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            pass

    class FakeSpool:
        def __init__(self) -> None:
            self.appended: list[bytes] = []
            self.discarded: list[int] = []
            self.closed = False

        def append(self, data: bytes):
            self.appended.append(data)
            return 0, len(data)

        def discard_before(self, offset: int) -> None:
            self.discarded.append(offset)

        def close(self) -> None:
            self.closed = True

    spool = FakeSpool()

    class FakeIngestResult:
        audio = b"clean-audio"
        timed_metadata_events = ()

    class FakeSessionResult:
        ingest = FakeIngestResult()
        transitions = ()

    class FakeSession:
        current_track = None

        def __init__(self, **kwargs) -> None:
            pass

        def feed(self, chunk: bytes) -> FakeSessionResult:
            return FakeSessionResult()

    class FakeObserver:
        def observe(self, **kwargs):
            return (analysis,)

    class FakePromoter:
        instances: list["FakePromoter"] = []

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.__class__.instances.append(self)

        def promote(self, **kwargs):
            self.calls.append(kwargs)
            return resolution

    class FakeAutonomousState:
        instances: list["FakeAutonomousState"] = []

        def __init__(self) -> None:
            self.observed = []
            self.transitions = []
            self.__class__.instances.append(self)

        def observe_known_segment(self, segment) -> None:
            self.observed.append(segment)

        def transition(
            self,
            candidate_resolution,
            *,
            minimum_open_seconds: float = 0.0,
        ):
            self.transitions.append(candidate_resolution)
            return transition

    class FakeOutputWriter:
        instances: list["FakeOutputWriter"] = []

        def __init__(self, **kwargs) -> None:
            self.segment_transitions = []
            self.track_transitions = []
            self.retained_start_offset = None
            self.__class__.instances.append(self)

        def can_write_boundary_at(self, incoming_start: int) -> bool:
            return (
                self.retained_start_offset is None
                or incoming_start > self.retained_start_offset
            )

        def write_segment_transition(self, value):
            self.segment_transitions.append(value)
            self.retained_start_offset = 4800
            return written_segment

        def write_transition(self, value):
            self.track_transitions.append(value)
            return object()

    monkeypatch.setattr(
        runner_module,
        "create_safe_streaming_spool",
        lambda *, directory: spool,
    )
    monkeypatch.setattr(
        runner_module,
        "RippingSession",
        FakeSession,
    )
    monkeypatch.setattr(
        runner_module,
        "AutonomousBoundaryPromoter",
        FakePromoter,
    )
    monkeypatch.setattr(
        runner_module,
        "AutonomousSegmentState",
        FakeAutonomousState,
    )
    monkeypatch.setattr(
        runner_module,
        "SessionOutputWriter",
        FakeOutputWriter,
    )

    callback_values: list[object] = []

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
            autonomous_boundaries=True,
        ),
        stream_opener=lambda url: FakeStream(),
        shadow_observer_factory=lambda ffmpeg_binary: FakeObserver(),
    )

    result = runner.run(
        on_segment_written=callback_values.append,
    )

    assert result.stopped is False

    assert spool.appended == [b"clean-audio"]
    assert spool.discarded == [4800]
    assert spool.closed is True

    assert len(FakePromoter.instances) == 1
    assert len(FakePromoter.instances[0].calls) == 1
    assert FakePromoter.instances[0].calls[0]["analysis"] is analysis
    assert "timeline" in FakePromoter.instances[0].calls[0]

    assert len(FakeAutonomousState.instances) == 1
    assert FakeAutonomousState.instances[0].transitions == [resolution]

    assert len(FakeOutputWriter.instances) == 1
    assert FakeOutputWriter.instances[0].segment_transitions == [transition]
    assert FakeOutputWriter.instances[0].track_transitions == []

    assert callback_values == [written_segment]


# --- stale autonomous boundary guard ---


@pytest.mark.parametrize("incoming_start", [4000, 5000])
def test_runner_ignores_promoted_autonomous_boundary_that_does_not_advance_output(
    tmp_path: Path,
    monkeypatch,
    incoming_start: int,
) -> None:
    import brynse.integrations.radio.runner as runner_module

    from brynse.models import (
        BoundaryCandidate,
        SplitDecision,
        SplitKind,
        TemporalSplitDecision,
        TemporalSplitKind,
    )
    from brynse.orchestrator import CandidateResolution

    analysis = object()

    resolution = CandidateResolution(
        candidate=BoundaryCandidate(
            time_seconds=4.0,
            source="autonomous",
        ),
        temporal=TemporalSplitDecision(
            kind=TemporalSplitKind.HARD_CUT,
            incoming_start_seconds=4.0,
            outgoing_end_seconds=4.0,
        ),
        split=SplitDecision(
            kind=SplitKind.HARD_CUT,
            incoming_start=incoming_start,
            outgoing_end=incoming_start,
        ),
    )

    class FakeStream:
        metadata = {
            "Content-Type": "audio/mpeg",
            "icy-metaint": "1",
        }

        def __init__(self) -> None:
            self._chunks = [b"chunk", b""]

        def read(self, size: int = -1) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            pass

    class FakeIngestResult:
        audio = b""
        timed_metadata_events = ()

    class FakeSessionResult:
        ingest = FakeIngestResult()
        transitions = ()

    class FakeSession:
        current_track = None

        def __init__(self, **kwargs) -> None:
            pass

        def feed(self, chunk: bytes) -> FakeSessionResult:
            return FakeSessionResult()

    class FakeObserver:
        def observe(self, **kwargs):
            return (analysis,)

    class FakePromoter:
        def __init__(self) -> None:
            pass

        def promote(self, **kwargs):
            return resolution

    class FakeAutonomousState:
        def __init__(self) -> None:
            pass

        def observe_known_segment(self, segment) -> None:
            pass

        def transition(
            self,
            candidate_resolution,
            *,
            minimum_open_seconds: float = 0.0,
        ):
            raise AssertionError(
                "stale autonomous boundary must be rejected "
                "before segment transition construction"
            )

    class FakeOutputWriter:
        instances: list["FakeOutputWriter"] = []

        def __init__(self, **kwargs) -> None:
            self.segment_writes = []
            self.__class__.instances.append(self)

        @property
        def retained_start_offset(self):
            return 5000

        def can_write_boundary_at(self, incoming_start: int) -> bool:
            return incoming_start > 5000

        def write_segment_transition(self, transition):
            self.segment_writes.append(transition)
            raise AssertionError(
                "stale autonomous boundary must not be materialized"
            )

        def write_transition(self, transition):
            raise AssertionError(
                "semantic transition not expected in this test"
            )

    monkeypatch.setattr(
        runner_module,
        "RippingSession",
        FakeSession,
    )
    monkeypatch.setattr(
        runner_module,
        "AutonomousBoundaryPromoter",
        FakePromoter,
    )
    monkeypatch.setattr(
        runner_module,
        "AutonomousSegmentState",
        FakeAutonomousState,
    )
    monkeypatch.setattr(
        runner_module,
        "SessionOutputWriter",
        FakeOutputWriter,
    )

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
            autonomous_boundaries=True,
        ),
        stream_opener=lambda url: FakeStream(),
        shadow_observer_factory=lambda ffmpeg_binary: FakeObserver(),
    )

    result = runner.run()

    assert result.stopped is False
    assert len(FakeOutputWriter.instances) == 1
    assert FakeOutputWriter.instances[0].segment_writes == []


# --- semantic/autonomous materialization ordering ---


def test_runner_materializes_semantic_transition_before_autonomous_promotion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import brynse.integrations.radio.runner as runner_module

    events: list[str] = []
    analysis = object()
    written_track = object()
    written_segment = object()

    class FakeIncomingTrack:
        title = "Known Track"
        start_offset = 5000
        start_time_seconds = 50.0

    class FakeTemporal:
        incoming_start_seconds = 45.0

    class FakeBoundary:
        temporal = FakeTemporal()

    class FakeTrackTransition:
        incoming = FakeIncomingTrack()
        boundary = FakeBoundary()

    semantic_transition = FakeTrackTransition()

    class FakeSplit:
        incoming_start = 6000

    class FakeResolution:
        split = FakeSplit()

    resolution = FakeResolution()
    autonomous_transition = object()

    class FakeStream:
        metadata = {
            "Content-Type": "audio/mpeg",
            "icy-metaint": "1",
        }

        def __init__(self) -> None:
            self._chunks = [b"chunk", b""]

        def read(self, size: int = -1) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            pass

    class FakeSpool:
        def __init__(self) -> None:
            self.discarded: list[int] = []

        def append(self, data: bytes):
            return 0, len(data)

        def discard_before(self, offset: int) -> None:
            self.discarded.append(offset)

        def close(self) -> None:
            pass

    spool = FakeSpool()

    class FakeIngestResult:
        audio = b"clean-audio"
        timed_metadata_events = ()

    class FakeSessionResult:
        ingest = FakeIngestResult()
        transitions = (semantic_transition,)

    class FakeSession:
        current_track = None

        def __init__(self, **kwargs) -> None:
            pass

        def feed(self, chunk: bytes) -> FakeSessionResult:
            return FakeSessionResult()

    class FakeObserver:
        def observe(self, **kwargs):
            events.append("shadow-observe")
            return (analysis,)

    class FakePromoter:
        def __init__(self) -> None:
            pass

        def promote(self, **kwargs):
            events.append("autonomous-promote")
            return resolution

    class FakeAutonomousState:
        instances: list["FakeAutonomousState"] = []

        def __init__(self) -> None:
            self.materialized = []
            self.__class__.instances.append(self)

        def observe_known_segment(self, segment) -> None:
            pass

        def observe_materialized_segment(self, segment) -> None:
            events.append("semantic-sync")
            self.materialized.append(segment)

        def transition(
            self,
            candidate_resolution,
            *,
            minimum_open_seconds: float = 0.0,
        ):
            events.append("autonomous-transition")
            return autonomous_transition

    class FakeOutputWriter:
        def __init__(self, **kwargs) -> None:
            self.retained_start_offset = None

        def can_write_boundary_at(self, incoming_start: int) -> bool:
            return (
                self.retained_start_offset is None
                or incoming_start > self.retained_start_offset
            )

        def write_transition(self, value):
            events.append("semantic-write")
            self.retained_start_offset = 4500
            return written_track

        def write_segment_transition(self, value):
            events.append("autonomous-write")
            self.retained_start_offset = 6000
            return written_segment

    monkeypatch.setattr(
        runner_module,
        "create_safe_streaming_spool",
        lambda *, directory: spool,
    )
    monkeypatch.setattr(
        runner_module,
        "RippingSession",
        FakeSession,
    )
    monkeypatch.setattr(
        runner_module,
        "AutonomousBoundaryPromoter",
        FakePromoter,
    )
    monkeypatch.setattr(
        runner_module,
        "AutonomousSegmentState",
        FakeAutonomousState,
    )
    monkeypatch.setattr(
        runner_module,
        "SessionOutputWriter",
        FakeOutputWriter,
    )

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
            autonomous_boundaries=True,
        ),
        stream_opener=lambda url: FakeStream(),
        shadow_observer_factory=lambda ffmpeg_binary: FakeObserver(),
    )

    result = runner.run()

    assert result.stopped is False

    assert events == [
        "semantic-write",
        "semantic-sync",
        "shadow-observe",
        "autonomous-promote",
        "autonomous-transition",
        "autonomous-write",
    ]

    assert len(FakeAutonomousState.instances) == 1
    assert FakeAutonomousState.instances[0].materialized == [
        Segment(
            start_offset=4500,
            start_time_seconds=45.0,
            label="Known Track",
        )
    ]

    assert spool.discarded == [4500, 6000]
