"""F-2: normal EOF and requested stop must not silently destroy audio."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from brynse.integrations.radio.runner import RippingRunConfig, RippingRunner
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

    spool_dir = output / ".brynse-spool"
    chunks = sorted(p for p in spool_dir.iterdir() if p.is_file() and p.name.endswith(".chunk"))
    assert chunks, f"{termination}: pending MP3 was deleted by normal runner cleanup"

    recovered = b"".join(path.read_bytes() for path in chunks)
    assert recovered == FRAME
    assert hashlib.sha256(recovered).digest() == hashlib.sha256(FRAME).digest()


    terminal_records = list(output.glob("brynse-terminal-*.json"))
    assert len(terminal_records) == 1
    record = json.loads(terminal_records[0].read_text(encoding="utf-8"))
    assert record["record_type"] == "terminal_coverage"
    assert record["source_start_offset"] == 0
    assert record["source_end_offset"] == len(FRAME)
    assert record["spool_start_offset"] == 0
    assert record["spool_end_offset"] == len(FRAME)
    assert record["retained_tail"] == {"start_offset": 0, "end_offset": len(FRAME)}
    assert record["published"] == []
    assert record["spool_chunks"]
