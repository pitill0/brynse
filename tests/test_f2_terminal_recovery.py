"""F-2: normal EOF and requested stop must not silently destroy audio."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from brynse.integrations.radio.runner import RippingRunConfig, RippingRunError, RippingRunner
from brynse.source import BinaryIOStreamSource

# Valid MP3 frame used by existing runner tests.
FRAME_LENGTH = 417
FRAME = b"\xff\xfb\x90\x00" + bytes(FRAME_LENGTH - 4)


@pytest.mark.parametrize("termination", ("eof", "stop"))
def test_terminal_audio_survives_normal_runner_exit(
    tmp_path: Path,
    termination: str,
) -> None:
    """Real spool bytes must remain recoverable when no track was published."""

    class Stream:
        def __init__(self) -> None:
            self.calls = 0
            self.runner: RippingRunner | None = None
            self.closed = False

        def read(self, size: int = -1) -> bytes:
            self.calls += 1
            if self.calls == 1:
                # One ICY interval followed by a zero-length metadata block.
                return FRAME + b"\x00"

            if termination == "stop":
                assert self.runner is not None
                self.runner.stop()
                # Model a blocking read interrupted by stop().
                raise OSError("read interrupted by requested stop")

            return b""

        def close(self) -> None:
            self.closed = True

    stream = Stream()
    output = tmp_path / termination

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/f2-terminal",
            output_directory=output,
            codec="mp3",
        ),
        stream_opener=lambda url: BinaryIOStreamSource(
            stream,
            metadata={
                "Content-Type": "audio/mpeg",
                "icy-metaint": str(FRAME_LENGTH),
            },
        ),
    )
    stream.runner = runner

    result = runner.run()

    assert result.stopped is (termination == "stop")
    assert stream.calls >= 2
    assert stream.closed

    published = tuple(output.glob("*.mp3"))
    assert not published, "No resolved track should be published"

    # The exact encoded tail is now durably published as recoverable audio.
    # No incomplete song is presented as a finalized MP3 track.
    terminal_audio = list(output.glob("brynse-terminal-audio-*.bin"))
    assert len(terminal_audio) == 1

    recovered = terminal_audio[0].read_bytes()
    assert recovered == FRAME
    assert hashlib.sha256(recovered).digest() == hashlib.sha256(FRAME).digest()

    spool_dir = output / ".brynse-spool"
    chunks = list(spool_dir.glob("*.chunk"))
    assert not chunks, "confirmed terminal publication must release spool chunks"

    terminal_records = list(output.glob("brynse-terminal-*.json"))
    assert len(terminal_records) == 1
    record = json.loads(terminal_records[0].read_text(encoding="utf-8"))
    assert record["record_type"] == "terminal_coverage"
    assert record["source_start_offset"] == 0
    assert record["source_end_offset"] == len(FRAME)
    assert record["spool_start_offset"] == 0
    assert record["spool_end_offset"] == len(FRAME)
    assert record["retained_tail"] is None
    assert len(record["published"]) == 1
    published_tail = record["published"][0]
    assert published_tail["kind"] == "unowned"
    assert published_tail["start_offset"] == 0
    assert published_tail["end_offset"] == len(FRAME)
    assert published_tail["file"] == terminal_audio[0].name
    assert record["spool_chunks"]


def test_terminal_audio_publication_failure_preserves_spool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed terminal publication must never discard encoded source audio."""
    import brynse.publication_provenance as provenance

    class Stream:
        def __init__(self) -> None:
            self.chunks = [FRAME + b"\\x00", b""]

        def read(self, size: int = -1) -> bytes:
            return self.chunks.pop(0)

        def close(self) -> None:
            pass

    def fail_publication(**kwargs: object) -> Path:
        raise OSError("injected terminal audio publication failure")

    monkeypatch.setattr(
        provenance,
        "publish_terminal_audio",
        fail_publication,
    )

    output = tmp_path / "failed-terminal-publication"

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/f2-terminal-failure",
            output_directory=output,
            codec="mp3",
        ),
        stream_opener=lambda url: BinaryIOStreamSource(
            Stream(),
            metadata={
                "Content-Type": "audio/mpeg",
                "icy-metaint": str(FRAME_LENGTH),
            },
        ),
    )

    with pytest.raises(
        RippingRunError,
        match="injected terminal audio publication failure",
    ):
        runner.run()

    chunks = sorted((output / ".brynse-spool").glob("*.chunk"))
    assert chunks, "source audio was discarded after publication failure"

    recovered = b"".join(chunk.read_bytes() for chunk in chunks)
    assert recovered == FRAME
    assert hashlib.sha256(recovered).digest() == hashlib.sha256(FRAME).digest()

    assert not list(output.glob("brynse-terminal-audio-*.bin"))
    assert not list(output.glob("brynse-terminal-*.json"))
