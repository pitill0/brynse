"""Only proved duplicate publication links may be reclaimed on restart."""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

import pytest

from brynse import _recoverable_publication as publication


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    recovery = publication.ensure_private_recovery_directory(tmp_path)
    output = tmp_path / "Radio.mp3"
    output.write_bytes(b"recording")
    backup = recovery / ("recovery-" + "a" * 32 + ".mp3")
    os.link(output, backup)
    return output, backup


def test_reconcile_removes_only_proven_duplicate(tmp_path: Path) -> None:
    output, backup = _setup(tmp_path)
    report = publication.reconcile_recovery_directory(tmp_path)
    assert report.removed == (backup.name,)
    assert not report.retained
    assert output.read_bytes() == b"recording"
    assert not backup.exists()
    assert output.stat().st_nlink == 1
    assert publication.reconcile_recovery_directory(tmp_path).removed == ()


def test_reconcile_preserves_unknown_and_recovery_only(tmp_path: Path) -> None:
    private = publication.ensure_private_recovery_directory(tmp_path)
    orphan = private / ("recovery-" + "b" * 32 + ".mp3")
    orphan.write_bytes(b"unpublished")
    artifact = private / ".brynse-file-unknown.tmp"
    artifact.write_bytes(b"partial")
    quarantine = private / "cleanup-abc"
    quarantine.write_bytes(b"other")
    symlink = private / ("recovery-" + "c" * 32 + ".mp3")
    symlink.symlink_to(orphan)
    report = publication.reconcile_recovery_directory(tmp_path)
    assert not report.removed
    assert set(report.retained) == {orphan.name, artifact.name, quarantine.name, symlink.name}
    assert orphan.read_bytes() == b"unpublished"
    assert symlink.is_symlink()


def test_reconcile_preserves_link_if_published_symlink(tmp_path: Path) -> None:
    recovery = publication.ensure_private_recovery_directory(tmp_path)
    foreign = tmp_path.parent / (tmp_path.name + "-foreign")
    foreign.write_bytes(b"foreign")
    backup = recovery / ("recovery-" + "d" * 32 + ".mp3")
    os.link(foreign, backup)
    (tmp_path / "Radio.mp3").symlink_to(foreign)
    report = publication.reconcile_recovery_directory(tmp_path)
    assert report.removed == ()
    assert backup.exists()
    assert foreign.read_bytes() == b"foreign"


def test_reconcile_retains_when_output_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, backup = _setup(tmp_path)
    original_fsync = publication.os.fsync

    def fail_output(fd: int) -> None:
        if os.fstat(fd).st_ino == tmp_path.stat().st_ino:
            raise OSError(errno.EIO, "injected output fsync error")
        original_fsync(fd)

    monkeypatch.setattr(publication.os, "fsync", fail_output)
    report = publication.reconcile_recovery_directory(tmp_path)
    assert not report.removed
    assert backup.exists()


def test_reconcile_retains_mismatched_inode_on_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _, backup = _setup(tmp_path)
    original = publication._cleanup_verified_link

    def replaced(fd: int, name: str, identity: tuple[int, int]) -> bool:
        backup.unlink()
        backup.write_bytes(b"foreign replacement")
        return original(fd, name, identity)

    monkeypatch.setattr(publication, "_cleanup_verified_link", replaced)
    with caplog.at_level(logging.WARNING, logger=publication.__name__):
        report = publication.reconcile_recovery_directory(tmp_path)
    assert report.removed == ()
    assert report.retained == (backup.name,)
    assert any(p.read_bytes() == b"foreign replacement" for p in backup.parent.glob("cleanup-*"))
