from __future__ import annotations

from pathlib import Path

import pytest

from brynse.models import SegmentByteRange
from brynse.output import SegmentOutputService


class _Source:
    def __init__(self, data: bytes) -> None:
        self.data = data

    @property
    def end_offset(self) -> int:
        return len(self.data)

    def contains(self, start: int, end: int) -> bool:
        return 0 <= start <= end <= len(self.data)

    def read(self, start: int, end: int) -> bytes:
        return self.data[start:end]


class _StreamingFinalizer:
    def finalize_stream(
        self,
        *,
        chunks,
        output_path: Path,
        timeout_seconds: float | None = None,
    ) -> None:
        with output_path.open("wb") as handle:
            for chunk in chunks:
                handle.write(chunk)


class _FailingStreamingFinalizer:
    def finalize_stream(
        self,
        *,
        chunks,
        output_path: Path,
        timeout_seconds: float | None = None,
    ) -> None:
        output_path.write_bytes(b"INCOMPLETE")
        raise RuntimeError("simulated finalization failure")


def _write(
    service: SegmentOutputService,
    directory: Path,
    stem: str,
    codec: str,
    payload: bytes,
) -> Path:
    return service.write_segment(
        source=_Source(payload),
        byte_range=SegmentByteRange(
            start_offset=0,
            end_offset=len(payload),
        ),
        directory=directory,
        stem=stem,
        codec=codec,
        chunk_size=4,
    )


@pytest.mark.parametrize(
    ("codec", "extension"),
    [
        ("mp3", ".mp3"),
        ("aac", ".m4a"),
    ],
)
def test_streaming_repeated_titles_preserve_recordings(
    tmp_path: Path,
    codec: str,
    extension: str,
) -> None:
    finalizer = _StreamingFinalizer()
    service = SegmentOutputService(
        mp3_finalizer=finalizer,
        aac_finalizer=finalizer,
    )

    first = _write(service, tmp_path, "Cadena SER", codec, b"FIRST")
    second = _write(service, tmp_path, "Cadena SER", codec, b"SECOND")

    assert first == tmp_path / f"Cadena SER{extension}"
    assert second == tmp_path / f"Cadena SER-2{extension}"
    assert first.read_bytes() == b"FIRST"
    assert second.read_bytes() == b"SECOND"


@pytest.mark.parametrize("codec", ["mp3", "aac"])
def test_streaming_long_unicode_stem_preserves_recordings(
    tmp_path: Path,
    codec: str,
) -> None:
    finalizer = _StreamingFinalizer()
    service = SegmentOutputService(
        mp3_finalizer=finalizer,
        aac_finalizer=finalizer,
    )

    stem = "界" * 150

    first = _write(service, tmp_path, stem, codec, b"FIRST")
    second = _write(service, tmp_path, stem, codec, b"SECOND")

    assert first != second
    assert len(first.name.encode("utf-8")) <= 240
    assert len(second.name.encode("utf-8")) <= 240
    assert first.read_bytes() == b"FIRST"
    assert second.read_bytes() == b"SECOND"


@pytest.mark.parametrize(
    ("codec", "extension"),
    [
        ("mp3", ".mp3"),
        ("aac", ".m4a"),
    ],
)
def test_streaming_failure_preserves_existing_recording(
    tmp_path: Path,
    codec: str,
    extension: str,
) -> None:
    existing = tmp_path / f"Cadena SER{extension}"
    existing.write_bytes(b"ORIGINAL")

    finalizer = _FailingStreamingFinalizer()
    service = SegmentOutputService(
        mp3_finalizer=finalizer,
        aac_finalizer=finalizer,
    )

    with pytest.raises(RuntimeError, match="simulated finalization failure"):
        _write(service, tmp_path, "Cadena SER", codec, b"NEW")

    assert existing.read_bytes() == b"ORIGINAL"
    assert sorted(path.name for path in tmp_path.iterdir()) == [existing.name]


@pytest.mark.parametrize(
    ("codec", "extension"),
    [
        ("mp3", ".mp3"),
        ("aac", ".m4a"),
    ],
)
def test_streaming_publication_failure_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    codec: str,
    extension: str,
) -> None:
    import errno

    import brynse._filename as filename_module
    from brynse.output import SegmentFileWriteError

    existing = tmp_path / f"Cadena SER{extension}"
    existing.write_bytes(b"ORIGINAL")

    def reject_publication(*, temp_path, directory, stem, extension):
        raise OSError(errno.EPERM, "publication denied")

    monkeypatch.setattr(
        filename_module,
        "publish_unique",
        reject_publication,
    )

    finalizer = _StreamingFinalizer()
    service = SegmentOutputService(
        mp3_finalizer=finalizer,
        aac_finalizer=finalizer,
    )

    with pytest.raises(SegmentFileWriteError, match="failed to publish"):
        _write(service, tmp_path, "Cadena SER", codec, b"NEW")

    assert existing.read_bytes() == b"ORIGINAL"
    assert sorted(path.name for path in tmp_path.iterdir()) == [existing.name]
