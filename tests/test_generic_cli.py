from pathlib import Path

import pytest

from brynse.generic_cli import GenericCliError, _build_parser, _validate_args


def test_generic_cli_parses_file_input() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            "input.aac",
            "--codec",
            "aac",
            "--interval",
            "30",
        ]
    )

    assert args.input == "input.aac"
    assert args.codec == "aac"
    assert args.interval == 30.0


def test_generic_cli_accepts_stdin() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            "-",
            "--codec",
            "mp3",
            "--interval",
            "10",
        ]
    )

    _validate_args(args)


@pytest.mark.parametrize("interval", ["0", "-1"])
def test_generic_cli_rejects_invalid_interval(interval: str) -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            "-",
            "--codec",
            "aac",
            "--interval",
            interval,
        ]
    )

    with pytest.raises(
        GenericCliError,
        match="--interval must be greater than zero",
    ):
        _validate_args(args)


def test_generic_cli_rejects_missing_file(tmp_path: Path) -> None:
    parser = _build_parser()

    missing = tmp_path / "missing.aac"

    args = parser.parse_args(
        [
            str(missing),
            "--codec",
            "aac",
            "--interval",
            "10",
        ]
    )

    with pytest.raises(
        GenericCliError,
        match="input file does not exist",
    ):
        _validate_args(args)


def test_external_provider_requires_boundaries_file(
    tmp_path: Path,
) -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            str(tmp_path / "input.mp3"),
            "--codec",
            "mp3",
            "--provider",
            "external",
        ]
    )

    with pytest.raises(
        GenericCliError,
        match="--boundaries-file is required when --provider=external",
    ):
        _validate_args(args)


def test_generic_cli_opens_file_as_stream_source(tmp_path: Path) -> None:
    import brynse.generic_cli as generic_cli

    input_path = tmp_path / "input.mp3"
    input_path.write_bytes(b"encoded-data")

    source = generic_cli._open_input_source(str(input_path))

    assert source.read(7) == b"encoded"
    assert source.metadata == {}

    source.close()


def test_generic_cli_opens_stdin_as_non_closing_stream_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from io import BytesIO
    from types import SimpleNamespace

    import brynse.generic_cli as generic_cli

    stream = BytesIO(b"stdin-data")

    monkeypatch.setattr(
        generic_cli.sys,
        "stdin",
        SimpleNamespace(buffer=stream),
    )

    source = generic_cli._open_input_source("-")

    assert source.read(5) == b"stdin"
    assert source.metadata == {}

    source.close()

    assert stream.closed is False


def test_iter_input_reads_through_stream_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import brynse.generic_cli as generic_cli

    class FakeSource:
        def __init__(self) -> None:
            self._chunks = [b"abc", b"def", b""]
            self.closed = False

        @property
        def metadata(self) -> dict[str, str]:
            return {}

        def read(self, max_bytes: int) -> bytes:
            assert max_bytes == 3
            return self._chunks.pop(0)

        def close(self) -> None:
            self.closed = True

    source = FakeSource()

    monkeypatch.setattr(
        generic_cli,
        "_open_input_source",
        lambda input_value: source,
    )

    chunks = list(
        generic_cli._iter_input(
            "ignored-input",
            chunk_size=3,
        )
    )

    assert chunks == [b"abc", b"def"]
    assert source.closed is True
