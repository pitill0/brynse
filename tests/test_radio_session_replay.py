from __future__ import annotations

import json
from pathlib import Path

import pytest

from brynse.integrations.radio.models import TimedMetadataEvent
from brynse.integrations.radio.replay import (
    ReplaySessionError,
    load_captured_events,
    load_replay_session,
    open_replay_source,
    replay_event_record,
    validate_replayed_events,
)


def _write_session(
    tmp_path: Path,
) -> Path:
    session = tmp_path / "session"
    session.mkdir()

    (session / "stream.icy").write_bytes(
        b"raw-audio\x00icy-metadata"
    )

    (session / "events.jsonl").write_text(
        json.dumps(
            {
                "title": "Artist - Track",
                "audio_offset": 32768,
                "audio_time_seconds": 1.25,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    (session / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "codec": "mp3",
                "metaint": 32768,
                "metadata_threshold_seconds": 8.0,
                "transient_exclusion": True,
                "autonomous_boundaries": False,
                "source_url": (
                    "https://example.test/stream"
                ),
                "source_metadata": {
                    "content-type": "audio/mpeg",
                    "icy-metaint": "32768",
                },
            }
        ),
        encoding="utf-8",
    )

    return session


def test_load_replay_session_restores_capture_contract(
    tmp_path: Path,
) -> None:
    directory = _write_session(tmp_path)

    session = load_replay_session(
        directory
    )

    assert session.codec == "mp3"
    assert session.metaint == 32768
    assert (
        session.metadata_threshold_seconds
        == 8.0
    )
    assert session.transient_exclusion is True
    assert (
        session.autonomous_boundaries
        is False
    )
    assert (
        session.source_metadata[
            "icy-metaint"
        ]
        == "32768"
    )


def test_open_replay_source_reads_exact_captured_bytes(
    tmp_path: Path,
) -> None:
    directory = _write_session(tmp_path)
    session = load_replay_session(
        directory
    )

    source = open_replay_source(session)

    try:
        assert (
            source.read(64 * 1024)
            == b"raw-audio\x00icy-metadata"
        )
        assert (
            source.metadata["icy-metaint"]
            == "32768"
        )
    finally:
        source.close()


def test_captured_events_are_validation_oracle(
    tmp_path: Path,
) -> None:
    directory = _write_session(tmp_path)
    session = load_replay_session(
        directory
    )

    assert load_captured_events(
        session
    ) == (
        {
            "title": "Artist - Track",
            "audio_offset": 32768,
            "audio_time_seconds": 1.25,
        },
    )


def test_replay_event_record_matches_capture_schema() -> None:
    event = TimedMetadataEvent(
        title="Artist - Track",
        audio_offset=32768,
        audio_time_seconds=1.25,
    )

    assert replay_event_record(event) == {
        "title": "Artist - Track",
        "audio_offset": 32768,
        "audio_time_seconds": 1.25,
    }


def test_validate_replayed_events_accepts_exact_match() -> None:
    records = (
        {
            "title": "Artist - Track",
            "audio_offset": 32768,
            "audio_time_seconds": 1.25,
        },
    )

    validate_replayed_events(
        captured=records,
        replayed=records,
    )


def test_validate_replayed_events_rejects_divergence() -> None:
    with pytest.raises(
        ReplaySessionError,
        match=(
            "replayed metadata timeline "
            "does not match capture"
        ),
    ):
        validate_replayed_events(
            captured=(
                {
                    "title": "A",
                    "audio_offset": 1,
                    "audio_time_seconds": 1.0,
                },
            ),
            replayed=(
                {
                    "title": "B",
                    "audio_offset": 1,
                    "audio_time_seconds": 1.0,
                },
            ),
        )


def test_replay_session_rejects_metaint_mismatch(
    tmp_path: Path,
) -> None:
    directory = _write_session(tmp_path)
    manifest_path = (
        directory / "manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )
    manifest["metaint"] = 16384

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        ReplaySessionError,
        match="metaint does not match",
    ):
        load_replay_session(directory)


def test_replay_session_requires_complete_bundle(
    tmp_path: Path,
) -> None:
    directory = _write_session(tmp_path)
    (
        directory / "events.jsonl"
    ).unlink()

    with pytest.raises(
        ReplaySessionError,
        match="missing events.jsonl",
    ):
        load_replay_session(directory)
