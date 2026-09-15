from pathlib import Path

import pytest

from fluxtuner_ripper.generic_cli import GenericCliError, _build_parser, _validate_args


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
