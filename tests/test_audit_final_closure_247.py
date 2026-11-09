"""Adversarial closure checks for the final review."""

from __future__ import annotations

import errno
import os
import threading
from pathlib import Path

import pytest

from brynse import _recoverable_publication as publication
from brynse.generic_cli import GenericCliError, _run_generic_pipeline
from brynse.streaming_spool import StreamingSpool, inspect_retained_spool_chunks


def test_r2_concurrent_directory_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "new" / "output"
    barrier = threading.Barrier(2)
    original = Path.mkdir

    def concurrent(path: Path, *args: object, **kwargs: object) -> None:
        if path == target:
            barrier.wait(timeout=5)
        original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", concurrent)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            publication.mkdir_durably(target)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert target.is_dir()


def test_r1_generic_retains_consumed_audio_after_directory_swap(tmp_path: Path) -> None:
    out = tmp_path / "out"
    moved = tmp_path / "out-old"
    encoded = b"\xff\xfb\x90\x64" + b"\0" * 413

    def chunks():
        yield encoded * 4
        out.rename(moved)
        out.mkdir()
        raise OSError(errno.ESTALE, "simulated output swap")

    with pytest.raises(GenericCliError, match="retained encoded spool") as caught:
        _run_generic_pipeline(
            chunks=chunks(),
            codec="mp3",
            interval_seconds=8,
            output_directory=out,
        )
    assert "PosixPath" not in str(caught.value)
    retained = inspect_retained_spool_chunks(moved)
    assert retained
    assert b"".join(path.read_bytes() for path in retained) == encoded * 4
    assert list(out.iterdir()) == []


def test_o1_inventory_ignores_symlinks(tmp_path: Path) -> None:
    spool = StreamingSpool(directory=tmp_path)
    spool.append(b"hello")
    paths = spool.recovery_paths
    spool.close(preserve=True)
    (tmp_path / ".brynse-spool-evil.chunk").symlink_to(paths[0])
    assert inspect_retained_spool_chunks(tmp_path) == paths


def test_o2_reconcile_verified_stream_twin(tmp_path: Path) -> None:
    private = publication.ensure_private_recovery_directory(tmp_path)
    published = tmp_path / "Track.mp3"
    published.write_bytes(b"finished")
    twin = private / ".brynse-stream-abcdefgh.mp3"
    os.link(published, twin)
    report = publication.reconcile_recovery_directory(tmp_path)
    assert twin.name in report.removed
    assert not twin.exists()
    assert published.read_bytes() == b"finished"


def test_o4_skip_output_scan_when_nothing_reclaimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = publication.ensure_private_recovery_directory(tmp_path)
    (private / "unknown.txt").write_bytes(b"recovery")
    original = os.listdir
    output_inode = tmp_path.stat().st_ino

    def listdir(target: object) -> list[str]:
        if isinstance(target, int) and os.fstat(target).st_ino == output_inode:
            raise AssertionError("unnecessary output directory indexing")
        return original(target)

    monkeypatch.setattr(publication.os, "listdir", listdir)
    assert publication.reconcile_recovery_directory(tmp_path).retained == ("unknown.txt",)
