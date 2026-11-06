from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from brynse.integrations.radio.recurrence_scan import annotate_clips, main
from brynse.recurrence import RecurrenceOccurrence, RecurringClip

RATE = 22050


def test_annotation_flags_clips_heard_at_title_changes() -> None:
    jingle = RecurringClip(
        clip_id=1,
        duration_seconds=3.0,
        occurrences=(
            RecurrenceOccurrence(100.0, 103.0, 0.99),
            RecurrenceOccurrence(300.0, 303.0, 0.99),
        ),
    )
    chorus = RecurringClip(
        clip_id=2,
        duration_seconds=8.0,
        occurrences=(
            RecurrenceOccurrence(150.0, 158.0, 0.97),
            RecurrenceOccurrence(230.0, 238.0, 0.97),
        ),
    )
    changes = [(0.0, "A - One"), (102.0, "B - Two"), (303.5, "C - Three")]

    annotated = annotate_clips([chorus, jingle], changes, near_transition_seconds=5.0)

    assert [clip.clip_id for clip in annotated] == [1, 2]
    assert annotated[0].transition_bound
    assert annotated[0].occurrences[0].distance_to_title_change_seconds == 0.0
    assert annotated[0].occurrences[1].nearest_title == "C - Three"
    assert not annotated[1].transition_bound


def _music(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    out = np.zeros(n)
    base = rng.uniform(150, 500)
    for index in range(int(seconds * 4)):
        start = index * RATE // 4
        frequency = base * 2 ** (rng.integers(0, 24) / 12)
        t = np.arange(min(RATE // 4, n - start)) / RATE
        out[start : start + len(t)] += np.sin(2 * np.pi * frequency * t) * np.exp(-t * 6)
    return 0.3 * out + 0.02 * rng.standard_normal(n)


def _jingle() -> np.ndarray:
    t = np.arange(int(3.0 * RATE)) / RATE
    noise = np.random.default_rng(99).standard_normal(len(t))
    return 0.5 * np.sin(2 * np.pi * (500 + 700 * t) * t) + 0.1 * noise


def _mp3(signal: np.ndarray) -> bytes:
    raw = (np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes()
    return subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(RATE),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            "-write_xing",
            "0",
            "-id3v2_version",
            "0",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=raw,
        capture_output=True,
        check=True,
    ).stdout


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_scan_finds_jingle_in_captured_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jingle = _jingle()
    pieces = [_music(70, 1), jingle, _music(70, 2), jingle, _music(15, 3)]
    starts = [70.0, 143.0]
    audio = _mp3(np.concatenate(pieces))

    metaint = 16000
    stream = bytearray()
    for index in range(0, len(audio), metaint):
        stream += audio[index : index + metaint]
        if len(audio[index : index + metaint]) == metaint:
            stream += b"\x00"

    session = tmp_path / "session"
    session.mkdir()
    (session / "stream.icy").write_bytes(bytes(stream))
    events = [
        {"title": "A - One", "audio_offset": 0, "audio_time_seconds": 0.0},
        {"title": "B - Two", "audio_offset": 1, "audio_time_seconds": 70.5},
        {"title": "C - Three", "audio_offset": 2, "audio_time_seconds": 143.4},
    ]
    (session / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "codec": "mp3",
                "metaint": metaint,
                "metadata_threshold_seconds": 8.0,
                "transient_exclusion": True,
                "autonomous_boundaries": False,
                "source_url": "https://example.test/stream",
                "source_metadata": {"content-type": "audio/mpeg", "icy-metaint": str(metaint)},
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "recurrence.json"
    assert main([str(session), "--json", str(output)]) == 0

    document = json.loads(output.read_text())
    assert len(document["clips"]) == 1
    clip = document["clips"][0]
    assert clip["transition_bound"]
    assert clip["duration_seconds"] == pytest.approx(3.0, abs=0.1)
    found = [item["start_seconds"] for item in clip["occurrences"]]
    # MP3 encoder/decoder delay shifts every time by the same few ms.
    assert found == pytest.approx(starts, abs=0.1)
    assert "AT TRANSITIONS" in capsys.readouterr().out


def test_annotation_rejects_invalid_transition_window() -> None:
    with pytest.raises(ValueError, match="near_transition_seconds"):
        annotate_clips([], [], near_transition_seconds=-1.0)
    with pytest.raises(ValueError, match="near_transition_seconds"):
        annotate_clips([], [], near_transition_seconds=float("nan"))


def _fake_decode_session(tmp_path, monkeypatch, payload: bytes):
    """Prepare an isolated decoder input without invoking actual FFmpeg."""
    from types import SimpleNamespace

    from brynse.integrations.radio import recurrence_scan

    stream = tmp_path / "stream.icy"
    stream.write_bytes(payload)

    monkeypatch.setattr(
        recurrence_scan,
        "load_replay_session",
        lambda _directory: SimpleNamespace(
            stream_path=stream,
            metaint=16000,
            codec="mp3",
        ),
    )

    class FakeParser:
        def __init__(self, *, metaint):
            assert metaint == 16000

        def feed(self, source):
            return SimpleNamespace(audio=source)

    monkeypatch.setattr(recurrence_scan, "IcyStreamParser", FakeParser)
    return recurrence_scan


def test_decode_rejects_oversized_capture(tmp_path, monkeypatch) -> None:
    from brynse.integrations.radio.replay import ReplaySessionError

    module = _fake_decode_session(tmp_path, monkeypatch, bytes(32))
    monkeypatch.setattr(module, "_MAX_CAPTURE_BYTES", 16)

    def unexpected_ffmpeg(*args, **kwargs):
        raise AssertionError("FFmpeg must not start")

    monkeypatch.setattr(
        module.subprocess,
        "run",
        unexpected_ffmpeg,
    )

    with pytest.raises(ReplaySessionError, match="captured stream exceeds"):
        module.decode_session_audio(tmp_path)


def test_decode_rejects_truncated_pcm(tmp_path, monkeypatch) -> None:
    from brynse.integrations.radio.replay import ReplaySessionError

    module = _fake_decode_session(tmp_path, monkeypatch, bytes(32))

    monkeypatch.setattr(module, "_MAX_PCM_BYTES", 1024)
    monkeypatch.setattr(module, "_PCM_LIMIT_GUARD_BYTES", 64)

    def bounded_ffmpeg(command, *, input, stdout, stderr, check, timeout):
        assert input == bytes(32)
        assert stderr == subprocess.PIPE
        assert check is False
        assert timeout == 600.0
        assert command[command.index("-fs") + 1] == "1024"
        stdout.write(bytes(1024))
        return subprocess.CompletedProcess(command, 0, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", bounded_ffmpeg)

    with pytest.raises(ReplaySessionError, match="PCM reached size limit"):
        module.decode_session_audio(tmp_path)


def test_decode_preserves_valid_pcm(tmp_path, monkeypatch) -> None:
    module = _fake_decode_session(tmp_path, monkeypatch, bytes(32))

    monkeypatch.setattr(module, "_MAX_PCM_BYTES", 1024)
    monkeypatch.setattr(module, "_PCM_LIMIT_GUARD_BYTES", 64)

    expected = bytes(range(256))

    def successful_ffmpeg(command, *, input, stdout, stderr, check, timeout):
        assert input == bytes(32)
        assert command[command.index("-fs") + 1] == "1024"
        stdout.write(expected)
        return subprocess.CompletedProcess(command, 0, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", successful_ffmpeg)

    result = module.decode_session_audio(tmp_path)

    assert result.sample_rate == 8000
    assert result.channels == 1
    assert result.sample_width_bytes == 2
    assert result.data == expected
