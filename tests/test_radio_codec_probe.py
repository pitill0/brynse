"""Bounded codec probing and lossless ICY stream replay."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from threading import Event

import pytest

from brynse.integrations.radio.codec_detection import detect_audio_codec
from brynse.integrations.radio.codec_probe import (
    MAX_PROBE_BYTES,
    PROBE_READ_BYTES,
    probe_stream_codec,
)
from brynse.integrations.radio.runner import (
    RippingRunConfig,
    RippingRunError,
    RippingRunner,
)
from brynse.source import BinaryIOStreamSource


def mp3_frame() -> bytes:
    return b"\xff\xfb\x90\x00" + bytes(413)


def aac_frame(length: int = 64) -> bytes:
    return bytes(
        (
            0xFF,
            0xF1,
            0x50,
            0x80 | ((length >> 11) & 3),
            (length >> 3) & 255,
            ((length & 7) << 5) | 0x1F,
            0xFC,
        )
    ) + bytes(length - 7)


def icy_wire(frame: bytes) -> bytes:
    return (frame + b"\x00") * 4


@pytest.mark.parametrize(
    ("frame", "expected"),
    [(mp3_frame(), "mp3"), (aac_frame(), "aac")],
)
def test_probe_preserves_exact_icy_bytes(frame: bytes, expected: str) -> None:
    wire = icy_wire(frame)
    source = BytesIO(wire)

    codec, prefetched = probe_stream_codec(
        source,
        metaint=len(frame),
        stop_event=Event(),
    )

    assert codec == expected
    assert b"".join(prefetched) + source.read() == wire


def test_probe_bounds_reads_and_rejects_unknown_audio() -> None:
    source = BytesIO(bytes(MAX_PROBE_BYTES * 2))

    codec, prefetched = probe_stream_codec(
        source,
        metaint=16000,
        stop_event=Event(),
    )

    assert codec is None
    assert sum(map(len, prefetched)) == MAX_PROBE_BYTES
    assert source.tell() == MAX_PROBE_BYTES
    assert all(len(chunk) <= PROBE_READ_BYTES for chunk in prefetched)


def test_probe_honors_stop_before_reading() -> None:
    stop = Event()
    stop.set()
    source = BytesIO(mp3_frame() * 4)

    assert probe_stream_codec(
        source,
        metaint=16000,
        stop_event=stop,
    ) == (None, ())
    assert source.tell() == 0


@pytest.mark.parametrize(
    ("frame", "expected"),
    [(mp3_frame(), "mp3"), (aac_frame(), "aac")],
)
def test_runner_recovers_missing_content_type(
    tmp_path: Path,
    frame: bytes,
    expected: str,
) -> None:
    source = BinaryIOStreamSource(
        BytesIO(icy_wire(frame)),
        metadata={"icy-metaint": str(len(frame))},
    )
    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
        ),
        stream_opener=lambda url: source,
    )
    started: list[tuple[str, int]] = []

    def on_started(codec: str, metaint: int) -> None:
        started.append((codec, metaint))
        runner.stop()

    result = runner.run(on_started=on_started)

    assert started == [(expected, len(frame))]
    assert result.codec == expected
    assert result.stopped


def test_runner_replays_probe_bytes_into_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from brynse.integrations.radio import runner as runner_module

    frame = mp3_frame()
    wire = icy_wire(frame)
    source = BinaryIOStreamSource(
        BytesIO(wire),
        metadata={"icy-metaint": str(len(frame))},
    )

    original_feed = runner_module.RippingSession.feed
    delivered: list[bytes] = []

    def recording_feed(self: object, chunk: bytes):
        delivered.append(chunk)
        return original_feed(self, chunk)

    monkeypatch.setattr(
        runner_module.RippingSession,
        "feed",
        recording_feed,
    )

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
        ),
        stream_opener=lambda url: source,
    )

    result = runner.run()

    assert result.codec == "mp3"
    assert not result.stopped
    assert b"".join(delivered) == wire


def test_runner_rejects_unknown_codec(tmp_path: Path) -> None:
    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
        ),
        stream_opener=lambda url: BinaryIOStreamSource(
            BytesIO(bytes(10000)),
            metadata={"icy-metaint": "16000"},
        ),
    )

    with pytest.raises(RippingRunError, match="bounded sample"):
        runner.run()


class FragmentedSource:
    """Return deliberately short reads, regardless of requested size."""

    def __init__(self, content: bytes, fragment: int = 17) -> None:
        self.stream = BytesIO(content)
        self.fragment = fragment

    def read(self, size: int) -> bytes:
        return self.stream.read(min(size, self.fragment))


def test_probe_fragmented_icy_metadata() -> None:
    frame = mp3_frame()
    metadata = b"StreamTitle='Synthetic';"
    padded = metadata.ljust(32, b"\x00")

    # One ICY block after each frame, including nonempty metadata.
    block = frame + bytes([2]) + padded
    wire = block * 4

    source = FragmentedSource(wire, fragment=17)
    codec, prefetched = probe_stream_codec(
        source,
        metaint=len(frame),
        stop_event=Event(),
    )

    assert codec == "mp3"

    remainder = bytearray()
    while chunk := source.read(len(wire)):
        remainder.extend(chunk)

    assert b"".join(prefetched) + bytes(remainder) == wire


def test_probe_accepts_vbr_mp3_frames() -> None:
    # MPEG-1 Layer III, 44.1 kHz, different valid bitrates.
    frames = []
    for bitrate_index, bitrate_kbps in (
        (9, 128),
        (11, 192),
        (5, 64),
        (9, 128),
    ):
        header = bytes(
            (
                0xFF,
                0xFB,
                bitrate_index << 4,
                0x00,
            )
        )
        length = (144000 * bitrate_kbps) // 44100
        frames.append(header + bytes(length - 4))

    assert detect_audio_codec(b"".join(frames)) == "mp3"


def test_probe_stops_after_interrupted_read() -> None:
    stop_event = Event()

    class InterruptedSource:
        def read(self, size: int) -> bytes:
            stop_event.set()
            raise OSError("read interrupted by stop")

    codec, chunks = probe_stream_codec(
        InterruptedSource(),
        metaint=16000,
        stop_event=stop_event,
    )

    assert codec is None
    assert chunks == ()


def test_runner_stop_during_successful_probe(
    tmp_path: Path,
) -> None:
    """Cancel at the probe/read boundary without reporting a new run."""

    class StopDuringProbe:
        metadata = {"icy-metaint": "16000"}

        def __init__(self) -> None:
            self.runner: RippingRunner | None = None
            self.read_calls = 0
            self.closed = False

        def read(self, size: int) -> bytes:
            self.read_calls += 1
            assert self.runner is not None
            self.runner.stop()
            return mp3_frame() * 4

        def close(self) -> None:
            self.closed = True

    source = StopDuringProbe()
    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
        ),
        stream_opener=lambda url: source,
    )
    source.runner = runner

    started: list[tuple[str, int]] = []

    result = runner.run(on_started=lambda codec, metaint: started.append((codec, metaint)))

    assert result.stopped is True
    assert started == [], "Cancelled probe must not announce a started run"
    assert source.closed is True
    assert source.read_calls == 1


def test_runner_concurrent_stop_during_probe(tmp_path: Path) -> None:
    """A concurrent stop must unblock probing without a start callback."""
    from threading import Thread

    entered = Event()
    released = Event()
    closed = Event()

    class BlockingSource:
        metadata = {"icy-metaint": "16000"}

        def read(self, size: int) -> bytes:
            entered.set()
            if not released.wait(timeout=5):
                raise AssertionError("Probe remained blocked")
            raise OSError("stream closed during read")

        def close(self) -> None:
            closed.set()
            released.set()

    source = BlockingSource()
    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
        ),
        stream_opener=lambda url: source,
    )

    results: list[object] = []
    errors: list[Exception] = []
    started: list[tuple[str, int]] = []

    def worker() -> None:
        try:
            results.append(
                runner.run(on_started=lambda codec, metaint: started.append((codec, metaint)))
            )
        except Exception as exc:
            errors.append(exc)

    thread = Thread(target=worker, daemon=True)
    thread.start()

    try:
        assert entered.wait(timeout=5), "Probe never began reading"

        runner.stop()
        thread.join(timeout=5)

        assert not thread.is_alive(), "Stop did not unblock probing"
        assert closed.is_set()
        assert errors == []
        assert len(results) == 1
        assert results[0].stopped is True
        assert started == []
    finally:
        released.set()
        runner.stop()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "content_type",
    ["", "application/octet-stream", "text/plain"],
)
def test_runner_falls_back_for_unusable_mime(
    tmp_path: Path,
    content_type: str,
) -> None:
    source = BinaryIOStreamSource(
        BytesIO(icy_wire(mp3_frame())),
        metadata={
            "Content-Type": content_type,
            "icy-metaint": str(len(mp3_frame())),
        },
    )
    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
        ),
        stream_opener=lambda url: source,
    )

    observed: list[str] = []

    def on_started(codec: str, metaint: int) -> None:
        observed.append(codec)
        runner.stop()

    result = runner.run(on_started=on_started)

    assert observed == ["mp3"]
    assert result.codec == "mp3"
    assert result.stopped


def test_runner_supported_mime_does_not_probe_audio(
    tmp_path: Path,
) -> None:
    class HeaderOnlySource:
        metadata = {
            "Content-Type": "audio/mpeg",
            "icy-metaint": "16000",
        }

        def __init__(self) -> None:
            self.read_calls = 0
            self.closed = False

        def read(self, size: int) -> bytes:
            self.read_calls += 1
            raise AssertionError("Unexpected read before start callback")

        def close(self) -> None:
            self.closed = True

    source = HeaderOnlySource()
    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
        ),
        stream_opener=lambda url: source,
    )

    observed: list[str] = []

    def on_started(codec: str, metaint: int) -> None:
        observed.append(codec)
        runner.stop()

    result = runner.run(on_started=on_started)

    assert observed == ["mp3"]
    assert result.stopped
    assert source.read_calls == 0
    assert source.closed


@pytest.mark.parametrize("frame", [mp3_frame(), aac_frame()])
def test_probe_rejects_incomplete_final_frame(frame: bytes) -> None:
    """Two complete frames and one truncated frame are insufficient."""
    audio = frame * 2 + frame[: max(1, len(frame) // 2)]

    assert detect_audio_codec(audio) is None

    source = BytesIO(audio)
    codec, prefetched = probe_stream_codec(
        source,
        metaint=16000,
        stop_event=Event(),
    )

    assert codec is None
    assert b"".join(prefetched) == audio


@pytest.mark.parametrize("frame", [mp3_frame(), aac_frame()])
def test_probe_rejects_discontinuous_frame_signatures(
    frame: bytes,
) -> None:
    """Three isolated valid frames do not constitute a continuous run."""
    audio = frame + b"broken" + frame + b"broken" + frame

    assert detect_audio_codec(audio) is None


@pytest.mark.parametrize("frame", [mp3_frame(), aac_frame()])
def test_probe_rejects_partial_sync_headers(frame: bytes) -> None:
    """A sample containing repeated but incomplete headers is not audio evidence."""
    audio = (frame[:4] + bytes(20)) * 20

    assert detect_audio_codec(audio) is None


def test_runner_recovers_after_failed_codec_probe(
    tmp_path: Path,
) -> None:
    """A failed automatic detection must not poison the next job."""
    first_source = BinaryIOStreamSource(
        BytesIO(bytes(10000)),
        metadata={"icy-metaint": "16000"},
    )
    second_source = BinaryIOStreamSource(
        BytesIO(icy_wire(mp3_frame())),
        metadata={"icy-metaint": str(len(mp3_frame()))},
    )
    sources = iter((first_source, second_source))

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
        ),
        stream_opener=lambda url: next(sources),
    )

    with pytest.raises(RippingRunError, match="bounded sample"):
        runner.run()

    started: list[tuple[str, int]] = []

    def on_started(codec: str, metaint: int) -> None:
        started.append((codec, metaint))
        runner.stop()

    result = runner.run(on_started=on_started)

    assert started == [("mp3", len(mp3_frame()))]
    assert result.codec == "mp3"
    assert result.stopped is True
