from __future__ import annotations

import io
from pathlib import Path

import pytest

from fluxtuner_ripper.cli import (
    CliError,
    _build_parser,
    _resolve_codec,
    _resolve_metaint,
    main,
)


def test_cli_resolves_mp3_codec_from_content_type() -> None:
    assert _resolve_codec("auto", {"Content-Type": "audio/mpeg"}) == "mp3"


def test_cli_resolves_aac_codec_from_content_type_with_parameters() -> None:
    assert (
        _resolve_codec(
            "auto",
            {"Content-Type": "audio/aacp; charset=binary"},
        )
        == "aac"
    )


def test_cli_codec_override_does_not_require_content_type() -> None:
    assert _resolve_codec("aac", {}) == "aac"


def test_cli_rejects_unknown_auto_codec() -> None:
    with pytest.raises(CliError, match="could not detect stream codec"):
        _resolve_codec("auto", {"Content-Type": "application/octet-stream"})


def test_cli_parses_positive_icy_metaint() -> None:
    assert _resolve_metaint({"icy-metaint": "16000"}) == 16000


@pytest.mark.parametrize("value", [None, "0", "-1", "broken"])
def test_cli_rejects_invalid_icy_metaint(value: str | None) -> None:
    headers = {} if value is None else {"icy-metaint": value}
    with pytest.raises(CliError):
        _resolve_metaint(headers)


def test_cli_main_ingests_mock_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import fluxtuner_ripper.cli as cli

    class FakeStream(io.BytesIO):
        pass

    # MPEG-1 Layer III, 128 kbps, 44.1 kHz: 417-byte frame.
    frame_length = 417
    frame = b"\xff\xfb\x90\x00" + bytes(frame_length - 4)

    metadata = b"StreamTitle='Artist - Track';"
    blocks = (len(metadata) + 15) // 16
    padded = metadata.ljust(blocks * 16, b"\x00")

    # Metadata lands exactly at the next-frame boundary. The ingestor keeps
    # it pending until the following complete frame establishes its timestamp.
    raw = frame + bytes([blocks]) + padded + frame

    stream = FakeStream(raw)
    headers = {
        "Content-Type": "audio/mpeg",
        "icy-metaint": str(frame_length),
    }

    monkeypatch.setattr(cli, "_open_stream", lambda url: (stream, headers))

    result = main(
        [
            "https://example.invalid/stream",
            "--output",
            str(tmp_path / "tracks"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "codec: mp3" in captured.out
    assert "icy-metaint: 4" in captured.out
    assert "Artist - Track" in captured.out
    assert (tmp_path / "tracks").is_dir()


def test_cli_requires_output_argument() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["https://example.invalid/stream"])

    assert excinfo.value.code == 2


def test_cli_reports_unfinished_track_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import fluxtuner_ripper.cli as cli

    class FakeRunner:
        current_track_title = "Artist - Open Track"

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> object:
            raise KeyboardInterrupt

        def stop(self) -> None:
            pass

    monkeypatch.setattr(cli, "RippingRunner", FakeRunner)

    result = main(
        [
            "https://example.invalid/stream",
            "--output",
            str(tmp_path / "tracks"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 130
    assert "incomplete track not finalized: Artist - Open Track" in captured.err
    assert "stopped" in captured.err


def test_cli_enables_transient_exclusion_by_default() -> None:
    args = _build_parser().parse_args(["https://example.invalid/stream", "--output", "/tmp/tracks"])

    assert args.transient_exclusion is True


def test_cli_can_disable_transient_exclusion() -> None:
    args = _build_parser().parse_args(
        [
            "https://example.invalid/stream",
            "--output",
            "/tmp/tracks",
            "--no-transient-exclusion",
        ]
    )

    assert args.transient_exclusion is False


def test_cli_reports_runtime_ripping_failure_as_operational_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import fluxtuner_ripper.cli as cli

    class FakeRunner:
        current_track_title = None

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> object:
            raise cli.RippingRunError("stream read failed")

        def stop(self) -> None:
            pass

    monkeypatch.setattr(cli, "RippingRunner", FakeRunner)

    result = cli.main(
        [
            "https://example.invalid/stream",
            "--output",
            str(tmp_path / "tracks"),
        ]
    )

    captured = capsys.readouterr()

    assert result == 1
    assert "stream read failed" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_radio_cli_handles_keyboard_interrupt_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import fluxtuner_ripper.cli as cli

    class FakeRunner:
        current_track_title = None

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> object:
            raise KeyboardInterrupt

        def stop(self) -> None:
            pass

    monkeypatch.setattr(cli, "RippingRunner", FakeRunner)

    result = cli.main(
        [
            "https://example.invalid/stream",
            "--output",
            str(tmp_path / "tracks"),
        ]
    )

    captured = capsys.readouterr()

    assert result == 130
    assert "stopped" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert captured.out == ""
