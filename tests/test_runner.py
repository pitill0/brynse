from __future__ import annotations

from pathlib import Path

import pytest

from fluxtuner_ripper.integrations.radio.runner import (
    RippingRunConfig,
    RippingRunError,
    RippingRunner,
    resolve_codec,
    resolve_metaint,
)
from fluxtuner_ripper.source import BinaryIOStreamSource


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
    import fluxtuner_ripper.integrations.radio.runner as runner_module

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
    import fluxtuner_ripper.integrations.radio.runner as runner_module

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
    import fluxtuner_ripper.integrations.radio.runner as runner_module

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
    import fluxtuner_ripper.integrations.radio.runner as runner_module

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
    import fluxtuner_ripper.integrations.radio.runner as runner_module

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
    import fluxtuner_ripper.integrations.radio.runner as runner_module

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

    import fluxtuner_ripper.integrations.radio.runner as runner

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

    import fluxtuner_ripper.integrations.radio.runner as runner

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

    import fluxtuner_ripper.integrations.radio.runner as runner

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

    from fluxtuner_ripper.integrations.radio.runner import RippingRunConfig, RippingRunner

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
