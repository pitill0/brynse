"""Recorded-session regression for missing musical introductions.

This test examines an existing replay output. It never launches a
replay or modifies any recorded session.
"""

import json
import os
from pathlib import Path

import pytest


def _audio_without_icy_metadata(
    stream_path: Path,
    metaint: int,
) -> bytes:
    raw = stream_path.read_bytes()
    audio = bytearray()
    cursor = 0

    while cursor < len(raw):
        take = min(metaint, len(raw) - cursor)
        audio.extend(raw[cursor : cursor + take])
        cursor += take

        if take < metaint or cursor == len(raw):
            break

        metadata_length = raw[cursor] * 16
        cursor += 1

        assert cursor + metadata_length <= len(raw)
        cursor += metadata_length

    assert cursor == len(raw)
    return bytes(audio)


def test_lisa_ekdahl_replay_preserves_musical_intro() -> None:
    capture_env = os.environ.get("BRYNSE_LISA_CAPTURE")
    track_env = os.environ.get("BRYNSE_LISA_TRACK")

    if not capture_env or not track_env:
        pytest.skip("External recorded-session fixtures not configured")

    capture = Path(capture_env).resolve(strict=True)
    track = Path(track_env).resolve(strict=True)

    assert capture.is_dir()
    assert track.is_file()

    title = "Lisa Ekdahl - I Know You Love Me"

    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["codec"] == "mp3"
    assert manifest["metaint"] == 32768
    assert manifest["source_metadata"]["icy-br"] == "192"

    recorded_events = [
        json.loads(line)
        for line in (capture / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    matching = [event for event in recorded_events if event.get("title") == title]
    assert len(matching) == 1
    event = matching[0]

    audio = _audio_without_icy_metadata(
        capture / "stream.icy",
        manifest["metaint"],
    )
    saved = track.read_bytes()

    # Skip the MP3 output's independently added ID3 header.
    probe_offset = 32768
    probe_length = 16384
    assert len(saved) >= probe_offset + probe_length

    probe = saved[probe_offset : probe_offset + probe_length]

    match = audio.find(probe)
    assert match >= 0, "Track must match captured audio"
    assert audio.find(probe, match + 1) == -1

    source_start = match - probe_offset
    assert source_start >= 0

    middle = len(saved) // 2
    assert (
        saved[middle : middle + probe_length]
        == audio[source_start + middle : source_start + middle + probe_length]
    ), "Track interior must match the captured stream"

    bytes_per_second = 192000 / 8
    delay = (source_start - event["audio_offset"]) / bytes_per_second

    # Specific to the manually audited Lisa Ekdahl recording.
    # Not a general rule for radio segmentation.
    assert -2.0 <= delay <= 2.0, (
        "INTRO_LOSS_REGRESSION: "
        f"song starts {delay:+.6f}s after ICY reference; "
        "the confirmed musical introduction was discarded"
    )
