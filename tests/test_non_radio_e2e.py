from __future__ import annotations

import shutil
import subprocess
from io import BytesIO

import pytest

from brynse.generic_cli import _run_generic_pipeline
from brynse.source import BinaryIOStreamSource


def test_non_radio_stream_source_materializes_fixed_interval_segments(
    tmp_path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    generated = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-f",
            "mp3",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout

    source = BinaryIOStreamSource(
        BytesIO(generated),
        metadata={"origin": "synthetic-non-radio"},
    )

    def chunks():
        try:
            while True:
                chunk = source.read(4096)
                if not chunk:
                    break
                yield chunk
        finally:
            source.close()

    output_directory = tmp_path / "segments"

    payload = _run_generic_pipeline(
        chunks=chunks(),
        codec="mp3",
        provider_name="fixed",
        interval_seconds=1.0,
        output_directory=output_directory,
        minimum_tail_seconds=0.1,
    )

    assert payload["bytes_ingested"] == len(generated)
    assert payload["provider"] == "fixed_interval"

    boundaries = payload["boundaries"]
    segments = payload["segments"]

    assert isinstance(boundaries, list)
    assert isinstance(segments, list)

    assert len(boundaries) >= 2
    assert len(segments) >= 3

    for segment in segments:
        path = segment["path"]
        assert path

        segment_path = output_directory / path.split("/")[-1]
        assert segment_path.exists()
        assert segment_path.stat().st_size > 0

    assert source._stream.closed is True
