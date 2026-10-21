from __future__ import annotations

import json
from pathlib import Path

import pytest

from brynse.integrations.radio.capture import RadioSessionCapture
from brynse.integrations.radio.cli import _build_parser
from brynse.integrations.radio.models import TimedMetadataEvent
from brynse.integrations.radio.runner import RippingRunConfig


def _capture(tmp_path: Path) -> RadioSessionCapture:
    return RadioSessionCapture(
        directory=tmp_path / "session",
        codec="mp3",
        metaint=16000,
        metadata_threshold_seconds=8.0,
        transient_exclusion=True,
        autonomous_boundaries=False,
        source_url="https://example.test/stream",
        source_metadata={
            "Content-Type": "audio/mpeg",
            "icy-metaint": "16000",
        },
    )


def test_session_capture_preserves_exact_raw_stream_bytes(
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path)

    capture.write_stream_chunk(b"audio-one\\x00metadata")
    capture.write_stream_chunk(b"audio-two")
    capture.close()

    assert capture.stream_path.read_bytes() == b"audio-one\\x00metadataaudio-two"


def test_session_capture_records_resolved_metadata_events(
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path)

    capture.write_metadata_event(
        TimedMetadataEvent(
            title="Artist - Track",
            audio_offset=123456,
            audio_time_seconds=42.75,
        )
    )
    capture.close()

    records = [
        json.loads(line)
        for line in capture.events_path.read_text(
            encoding="utf-8",
        ).splitlines()
    ]

    assert records == [
        {
            "title": "Artist - Track",
            "audio_offset": 123456,
            "audio_time_seconds": 42.75,
        }
    ]


def test_session_capture_manifest_describes_replay_contract(
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path)
    capture.close()

    manifest = json.loads(
        capture.manifest_path.read_text(
            encoding="utf-8",
        )
    )

    assert manifest == {
        "schema_version": 1,
        "codec": "mp3",
        "metaint": 16000,
        "metadata_threshold_seconds": 8.0,
        "transient_exclusion": True,
        "autonomous_boundaries": False,
        "source_url": "https://example.test/stream",
        "source_metadata": {
            "Content-Type": "audio/mpeg",
            "icy-metaint": "16000",
        },
    }


def test_session_capture_refuses_to_overwrite_existing_session(
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path)
    capture.close()

    with pytest.raises(
        FileExistsError,
        match="session capture already contains",
    ):
        _capture(tmp_path)


def test_session_capture_close_is_idempotent(
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path)

    capture.close()
    capture.close()


def test_radio_cli_accepts_session_capture_path() -> None:
    args = _build_parser().parse_args(
        [
            "https://example.test/stream",
            "--output",
            "/tmp/tracks",
            "--session-capture",
            "/tmp/session",
        ]
    )

    assert args.session_capture == Path("/tmp/session")


def test_ripping_run_config_disables_capture_by_default(
    tmp_path: Path,
) -> None:
    config = RippingRunConfig(
        url="https://example.test/stream",
        output_directory=tmp_path,
    )

    assert config.session_capture_directory is None
