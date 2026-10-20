from __future__ import annotations

from pathlib import Path

import pytest

from brynse.integrations.radio.session_output import safe_track_stem
from brynse.output import SegmentFileWriter


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("ДДТ - Осень", "ДДТ - Осень"),
        ("Rosalía - Despechá", "Rosalía - Despechá"),
        ("東京 - 夜", "東京 - 夜"),
    ],
)
def test_safe_track_stem_preserves_unicode(
    title: str,
    expected: str,
) -> None:
    assert safe_track_stem(title) == expected


@pytest.mark.parametrize("codec", ["mp3", "aac"])
def test_repeated_titles_preserve_both_recordings(
    tmp_path: Path,
    codec: str,
) -> None:
    writer = SegmentFileWriter()

    first = writer.write(
        directory=tmp_path,
        stem="Cadena SER",
        codec=codec,
        data=b"FIRST_RECORDING",
    )
    second = writer.write(
        directory=tmp_path,
        stem="Cadena SER",
        codec=codec,
        data=b"SECOND_RECORDING",
    )

    assert first != second
    assert first.read_bytes() == b"FIRST_RECORDING"
    assert second.read_bytes() == b"SECOND_RECORDING"
    assert first.name == f"Cadena SER.{codec}"
    assert second.name == f"Cadena SER-2.{codec}"


def test_sanitized_title_collisions_preserve_both_recordings(
    tmp_path: Path,
) -> None:
    writer = SegmentFileWriter()

    first_stem = safe_track_stem("Artist / Track")
    second_stem = safe_track_stem("Artist : Track")

    assert first_stem == second_stem

    first = writer.write(
        directory=tmp_path,
        stem=first_stem,
        codec="mp3",
        data=b"FIRST",
    )
    second = writer.write(
        directory=tmp_path,
        stem=second_stem,
        codec="mp3",
        data=b"SECOND",
    )

    assert first != second
    assert first.read_bytes() == b"FIRST"
    assert second.read_bytes() == b"SECOND"


@pytest.mark.parametrize("title", ["A" * 300, "界" * 150])
def test_long_titles_fit_filesystem_filename_limit(
    tmp_path: Path,
    title: str,
) -> None:
    writer = SegmentFileWriter()

    path = writer.write(
        directory=tmp_path,
        stem=safe_track_stem(title),
        codec="mp3",
        data=b"LONG_RECORDING",
    )

    assert len(path.name.encode("utf-8")) <= 255
    assert path.read_bytes() == b"LONG_RECORDING"

    second = writer.write(
        directory=tmp_path,
        stem=safe_track_stem(title),
        codec="mp3",
        data=b"SECOND_LONG_RECORDING",
    )

    assert second != path
    assert len(second.name.encode("utf-8")) <= 255
    assert path.read_bytes() == b"LONG_RECORDING"
    assert second.read_bytes() == b"SECOND_LONG_RECORDING"


def test_m4a_fallback_preserves_existing_recording(
    tmp_path: Path,
) -> None:
    from brynse.output import SegmentOutputService

    original = tmp_path / "Cadena SER.m4a"
    original.write_bytes(b"ORIGINAL")

    service = SegmentOutputService()
    path = service._write_m4a(
        directory=tmp_path,
        stem="Cadena SER",
        data=b"NEW",
    )

    assert original.read_bytes() == b"ORIGINAL"
    assert path == tmp_path / "Cadena SER-2.m4a"
    assert path.read_bytes() == b"NEW"
    assert not list(tmp_path.glob("*.tmp"))


def test_m4a_fallback_handles_long_unicode_stem(
    tmp_path: Path,
) -> None:
    from brynse.output import SegmentOutputService

    service = SegmentOutputService()
    stem = "界" * 150

    first = service._write_m4a(
        directory=tmp_path,
        stem=stem,
        data=b"FIRST",
    )
    second = service._write_m4a(
        directory=tmp_path,
        stem=stem,
        data=b"SECOND",
    )

    assert first != second
    assert len(first.name.encode("utf-8")) <= 240
    assert len(second.name.encode("utf-8")) <= 240
    assert first.read_bytes() == b"FIRST"
    assert second.read_bytes() == b"SECOND"
    assert not list(tmp_path.glob("*.tmp"))
