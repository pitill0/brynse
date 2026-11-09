"""Permanent B1-B4 acceptance contracts for descriptor-based publication."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from brynse import _recoverable_publication as publication
from brynse.integrations.radio.runner import RippingRunConfig, RippingRunError, RippingRunner
from brynse.models import SegmentByteRange
from brynse.output import SegmentFileWriteError, SegmentFileWriter, SegmentOutputService


class _Source:
    def __init__(self, content: bytes) -> None:
        self.content = content

    @property
    def end_offset(self) -> int:
        return len(self.content)

    def contains(self, start: int, end: int) -> bool:
        return 0 <= start <= end <= len(self.content)

    def read(self, start: int, end: int) -> bytes:
        return self.content[start:end]


class _Finalizer:
    def finalize_stream(self, *, chunks, output_path: Path, timeout_seconds=None) -> None:
        with output_path.open("wb") as output:
            for chunk in chunks:
                output.write(chunk)


def test_b1_streaming_symlink_output_publishes_into_real_directory(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "output-link"
    link.symlink_to(real, target_is_directory=True)
    finalizer = _Finalizer()
    service = SegmentOutputService(mp3_finalizer=finalizer, aac_finalizer=finalizer)
    payload = b"FULL_COMPLETED_AUDIO"

    result = service.write_segment(
        source=_Source(payload),
        byte_range=SegmentByteRange(start_offset=0, end_offset=len(payload)),
        directory=link,
        stem="Artist - Track",
        codec="mp3",
        chunk_size=4,
    )
    assert result.read_bytes() == payload
    assert result.resolve().parent == real
    assert not list((real / ".brynse-recovery").iterdir())


@pytest.mark.parametrize("failure", ["invalid-mode", "no-hardlink"])
def test_b4_preflight_failure_never_calls_stream_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    if failure == "invalid-mode":
        recovery = tmp_path / ".brynse-recovery"
        recovery.mkdir(mode=0o755)
        recovery.chmod(0o755)
    else:

        def deny_link(source_fd: int, directory_fd: int, name: str) -> None:
            raise OSError(errno.EPERM, "injected denied hardlink")

        monkeypatch.setattr(publication, "_link_fd", deny_link)

    opened: list[str] = []

    def opener(url: str):
        opened.append(url)
        raise AssertionError("stream opened before successful publication preflight")

    runner = RippingRunner(
        RippingRunConfig(url="https://example.invalid/stream", output_directory=tmp_path),
        stream_opener=opener,
    )
    with pytest.raises(RippingRunError, match="output storage incompatible"):
        runner.run()
    assert not opened
    assert not list(tmp_path.glob("*.mp3"))


def test_b3_proc_fd_link_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"completed")
    seen: dict[str, object] = {}

    def observe(src: str, dst: str, **kwargs: object) -> None:
        seen.update(src=src, dst=dst, kwargs=kwargs)

    with source.open("rb") as handle, os.scandir(tmp_path):
        output_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with patch.object(publication.os, "link", side_effect=observe):
                publication._link_fd(handle.fileno(), output_fd, "recorded.mp3")
            assert seen["src"] == f"/proc/self/fd/{handle.fileno()}"
            assert seen["dst"] == "recorded.mp3"
            assert seen["kwargs"] == {
                "dst_dir_fd": output_fd,
                "follow_symlinks": True,
            }
        finally:
            os.close(output_fd)


def test_b3_missing_proc_fd_is_reported_as_enotsup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"completed")
    with source.open("rb") as handle:
        output_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with (
                patch.object(
                    publication.os, "link", side_effect=FileNotFoundError(errno.ENOENT, "gone")
                ),
                patch.object(publication.Path, "exists", return_value=False),
                pytest.raises(OSError) as caught,
            ):
                publication._link_fd(handle.fileno(), output_fd, "recorded.mp3")
            assert caught.value.errno == errno.ENOTSUP
        finally:
            os.close(output_fd)


@pytest.mark.parametrize("route", ["writer", "m4a", "streaming"])
def test_b2_bad_recovery_mode_rejected_by_all_output_routes(tmp_path: Path, route: str) -> None:
    recovery = tmp_path / ".brynse-recovery"
    recovery.mkdir(mode=0o755)
    recovery.chmod(0o755)
    with pytest.raises(SegmentFileWriteError, match="recovery directory must have mode 0700"):
        if route == "writer":
            SegmentFileWriter().write(directory=tmp_path, stem="Radio", codec="mp3", data=b"audio")
        elif route == "m4a":
            SegmentOutputService()._write_m4a(directory=tmp_path, stem="Radio", data=b"audio")
        else:
            finalizer = _Finalizer()
            service = SegmentOutputService(mp3_finalizer=finalizer, aac_finalizer=finalizer)
            service.write_segment(
                source=_Source(b"audio"),
                byte_range=SegmentByteRange(start_offset=0, end_offset=5),
                directory=tmp_path,
                stem="Radio",
                codec="mp3",
                chunk_size=2,
            )
    assert not list(tmp_path.glob("*.mp3"))
