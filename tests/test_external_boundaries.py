from pathlib import Path

import pytest

from fluxtuner_ripper.external_boundaries import (
    ExternalBoundaryParseError,
    load_external_boundaries,
    parse_external_boundaries_json,
    parse_external_boundaries_jsonl,
)


def test_parse_external_boundaries_json() -> None:
    boundaries = parse_external_boundaries_json(
        """
        [
          {
            "time_seconds": 12.5,
            "source": "agent"
          },
          {
            "time_seconds": 30,
            "source": "vad",
            "reference_offset": 720000
          }
        ]
        """
    )

    assert len(boundaries) == 2

    assert boundaries[0].time_seconds == 12.5
    assert boundaries[0].source == "agent"
    assert boundaries[0].reference_offset is None

    assert boundaries[1].time_seconds == 30.0
    assert boundaries[1].source == "vad"
    assert boundaries[1].reference_offset == 720000


def test_parse_external_boundaries_jsonl() -> None:
    boundaries = parse_external_boundaries_jsonl(
        """
        {"time_seconds": 10.0, "source": "agent"}

        {"time_seconds": 20.0}
        """
    )

    assert [boundary.time_seconds for boundary in boundaries] == [
        10.0,
        20.0,
    ]
    assert [boundary.source for boundary in boundaries] == [
        "agent",
        "external",
    ]


def test_parse_external_boundaries_rejects_invalid_shape() -> None:
    with pytest.raises(
        ExternalBoundaryParseError,
        match="must contain an array",
    ):
        parse_external_boundaries_json('{"time_seconds": 10.0}')


def test_parse_external_boundaries_rejects_missing_time() -> None:
    with pytest.raises(
        ExternalBoundaryParseError,
        match="missing time_seconds",
    ):
        parse_external_boundaries_json('[{"source": "agent"}]')


def test_parse_external_boundaries_rejects_boolean_time() -> None:
    with pytest.raises(
        ExternalBoundaryParseError,
        match="time_seconds must be numeric",
    ):
        parse_external_boundaries_json('[{"time_seconds": true}]')


def test_load_external_boundaries_jsonl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "boundaries.jsonl"
    path.write_text(
        '{"time_seconds": 12.5, "source": "agent"}\n',
        encoding="utf-8",
    )

    boundaries = load_external_boundaries(path)

    assert len(boundaries) == 1
    assert boundaries[0].time_seconds == 12.5
    assert boundaries[0].source == "agent"


def test_load_external_boundaries_rejects_unknown_extension(
    tmp_path: Path,
) -> None:
    path = tmp_path / "boundaries.txt"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ExternalBoundaryParseError,
        match=r"\.json or \.jsonl",
    ):
        load_external_boundaries(path)


def test_load_external_boundaries_from_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import sys

    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            '{"time_seconds": 12.5, "source": "agent"}\n{"time_seconds": 30.0, "source": "vad"}\n'
        ),
    )

    boundaries = load_external_boundaries(Path("-"))

    assert [boundary.time_seconds for boundary in boundaries] == [
        12.5,
        30.0,
    ]
    assert [boundary.source for boundary in boundaries] == [
        "agent",
        "vad",
    ]
