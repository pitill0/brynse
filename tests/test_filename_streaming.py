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
    """Incomplete finalizer output is removed, never published."""
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
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_file()) == [existing.name]

    private = tmp_path / ".brynse-recovery"
    assert private.is_dir()
    assert list(private.iterdir()) == []


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
    """A complete stream remains recoverable if final linking fails."""
    import errno

    import brynse._recoverable_publication as publication
    from brynse.output import SegmentFileWriteError

    existing = tmp_path / f"Cadena SER{extension}"
    existing.write_bytes(b"ORIGINAL")

    original_link = publication._link_fd
    final_attempts = 0

    def deny_final_link(
        source_fd: int,
        directory_fd: int,
        name: str,
    ) -> None:
        nonlocal final_attempts
        if not name.startswith("recovery-"):
            final_attempts += 1
            raise OSError(errno.EPERM, "publication denied")
        original_link(source_fd, directory_fd, name)

    monkeypatch.setattr(publication, "_link_fd", deny_final_link)

    finalizer = _StreamingFinalizer()
    service = SegmentOutputService(
        mp3_finalizer=finalizer,
        aac_finalizer=finalizer,
    )

    with pytest.raises(SegmentFileWriteError, match="failed to publish"):
        _write(service, tmp_path, "Cadena SER", codec, b"NEW")

    assert final_attempts == 1
    assert existing.read_bytes() == b"ORIGINAL"
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_file()) == [existing.name]

    private = tmp_path / ".brynse-recovery"
    staging = list(private.glob(".brynse-stream-*"))
    backups = list(private.glob(f"recovery-*{extension}"))
    assert len(staging) == 1
    assert len(backups) == 1
    assert staging[0].read_bytes() == b"NEW"
    assert backups[0].read_bytes() == b"NEW"
