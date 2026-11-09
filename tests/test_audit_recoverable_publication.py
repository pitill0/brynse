"""Invariants for fd-based, recoverable Linux publication."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

import brynse._recoverable_publication as publication


@pytest.fixture
def completed_fd(tmp_path: Path):
    fd = os.open(
        tmp_path,
        os.O_TMPFILE | os.O_RDWR | os.O_CLOEXEC,
        0o600,
    )
    try:
        os.write(fd, b"COMPLETE_RECORDING")
        yield fd
    finally:
        os.close(fd)


def test_publishes_same_inode_with_private_recovery(tmp_path: Path, completed_fd: int) -> None:
    receipt = publication.publish_fd_recoverably(
        source_fd=completed_fd,
        directory=tmp_path,
        stem="Radio",
        extension=".mp3",
    )
    assert receipt.published_path.read_bytes() == b"COMPLETE_RECORDING"
    assert receipt.recovery_path.read_bytes() == b"COMPLETE_RECORDING"
    assert receipt.published_path.stat().st_ino == os.fstat(completed_fd).st_ino
    assert receipt.recovery_path.stat().st_ino == os.fstat(completed_fd).st_ino
    assert stat.S_IMODE(receipt.recovery_path.parent.stat().st_mode) == 0o700


def test_existing_target_gets_exclusive_suffix(tmp_path: Path, completed_fd: int) -> None:
    original = tmp_path / "Radio.mp3"
    original.write_bytes(b"ORIGINAL")
    receipt = publication.publish_fd_recoverably(
        source_fd=completed_fd,
        directory=tmp_path,
        stem="Radio",
        extension=".mp3",
    )
    assert original.read_bytes() == b"ORIGINAL"
    assert receipt.published_path.name == "Radio-2.mp3"
    assert receipt.published_path.read_bytes() == b"COMPLETE_RECORDING"


def test_publication_denied_preserves_private_recovery(
    tmp_path: Path,
    completed_fd: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_link = publication._link_fd

    def reject_final(source_fd: int, dir_fd: int, name: str) -> None:
        if name.endswith(".mp3") and not name.startswith("recovery-"):
            raise OSError(errno.EPERM, "final link denied")
        real_link(source_fd, dir_fd, name)

    monkeypatch.setattr(publication, "_link_fd", reject_final)

    with pytest.raises(publication.RecoverablePublicationError) as caught:
        publication.publish_fd_recoverably(
            source_fd=completed_fd,
            directory=tmp_path,
            stem="Radio",
            extension=".mp3",
        )

    recovery = caught.value.recovery_path
    assert recovery.read_bytes() == b"COMPLETE_RECORDING"
    assert not (tmp_path / "Radio.mp3").exists()
    assert stat.S_IMODE(recovery.parent.stat().st_mode) == 0o700


def test_recovery_directory_symlink_is_rejected(tmp_path: Path, completed_fd: int) -> None:
    outside = tmp_path.parent / "foreign-recovery-target"
    outside.mkdir(exist_ok=True)
    (tmp_path / ".brynse-recovery").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        publication.publish_fd_recoverably(
            source_fd=completed_fd,
            directory=tmp_path,
            stem="Radio",
            extension=".mp3",
        )

    assert not list(outside.iterdir())


def test_no_name_based_cleanup_during_publication(
    tmp_path: Path,
    completed_fd: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_unlink(*args: object, **kwargs: object) -> None:
        raise AssertionError("untrusted name-based cleanup attempted")

    monkeypatch.setattr(publication.os, "unlink", forbidden_unlink)

    receipt = publication.publish_fd_recoverably(
        source_fd=completed_fd,
        directory=tmp_path,
        stem="Radio",
        extension=".m4a",
    )
    assert receipt.published_path.read_bytes() == b"COMPLETE_RECORDING"


def test_failed_output_directory_fsync_reports_existing_publication(
    tmp_path: Path,
    completed_fd: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed directory sync must not invite a duplicate publication."""
    original_fsync = os.fsync
    directory_syncs = 0

    def fail_output_sync(fd: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_syncs += 1
            if directory_syncs == 3:
                raise OSError(errno.EIO, "injected output directory fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(publication.os, "fsync", fail_output_sync)

    with pytest.raises(publication.PublicationDurabilityError) as caught:
        publication.publish_fd_recoverably(
            source_fd=completed_fd,
            directory=tmp_path,
            stem="Radio",
            extension=".mp3",
        )

    failure = caught.value
    assert directory_syncs == 3
    assert failure.errno == errno.EIO
    assert failure.published_path == tmp_path / "Radio.mp3"
    assert failure.published_path.read_bytes() == b"COMPLETE_RECORDING"
    assert failure.recovery_path.read_bytes() == b"COMPLETE_RECORDING"
    assert not (tmp_path / "Radio-2.mp3").exists()


def test_recovery_directory_sync_failure_blocks_final_publication(
    tmp_path: Path,
    completed_fd: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed recovery directory sync must block final publication."""
    original_fsync = os.fsync
    directory_syncs = 0

    def fail_first_directory_sync(fd: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_syncs += 1
            if directory_syncs == 2:
                raise OSError(errno.EIO, "recovery directory sync denied")
        original_fsync(fd)

    monkeypatch.setattr(publication.os, "fsync", fail_first_directory_sync)

    with pytest.raises(publication.RecoveryDurabilityError) as caught:
        publication.publish_fd_recoverably(
            source_fd=completed_fd,
            directory=tmp_path,
            stem="Radio",
            extension=".mp3",
        )

    assert directory_syncs == 2
    assert caught.value.errno == errno.EIO
    assert caught.value.recovery_path.read_bytes() == b"COMPLETE_RECORDING"
    assert not (tmp_path / "Radio.mp3").exists()


def test_failed_final_link_retains_final_file_mode(
    tmp_path: Path,
    completed_fd: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery is finalized before attempting the destination link."""
    real_link = publication._link_fd

    def fail_final_link(
        source_fd: int,
        directory_fd: int,
        name: str,
    ) -> None:
        if not name.startswith("recovery-"):
            raise OSError(errno.EPERM, "final publication denied")
        real_link(source_fd, directory_fd, name)

    monkeypatch.setattr(publication, "_link_fd", fail_final_link)

    with pytest.raises(publication.RecoverablePublicationError) as caught:
        publication.publish_fd_recoverably(
            source_fd=completed_fd,
            directory=tmp_path,
            stem="Radio",
            extension=".mp3",
        )

    recovered = caught.value.recovery_path
    assert recovered.read_bytes() == b"COMPLETE_RECORDING"
    assert stat.S_IMODE(recovered.stat().st_mode) == stat.S_IMODE(os.fstat(completed_fd).st_mode)
    assert not (tmp_path / "Radio.mp3").exists()


def test_successful_private_publication_has_no_retained_links(
    tmp_path: Path,
) -> None:
    """Normal success must not accumulate private recovery links."""
    from brynse.output import SegmentFileWriter

    for index in range(5):
        path = SegmentFileWriter().write(
            directory=tmp_path,
            stem="Radio",
            codec="mp3",
            data=f"COMPLETE_{index}".encode(),
        )
        assert path.read_bytes() == f"COMPLETE_{index}".encode()

    private = tmp_path / ".brynse-recovery"
    assert private.is_dir()
    assert list(private.iterdir()) == []


def test_unsupported_linkat_retains_completed_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported filesystem capability fails without losing audio."""
    from brynse.output import SegmentFileWriteError, SegmentFileWriter

    def unavailable(
        source_fd: int,
        directory_fd: int,
        name: str,
    ) -> None:
        raise OSError(errno.ENOTSUP, "linkat unavailable")

    monkeypatch.setattr(publication, "_link_fd", unavailable)

    with pytest.raises(SegmentFileWriteError):
        SegmentFileWriter().write(
            directory=tmp_path,
            stem="Radio",
            codec="mp3",
            data=b"RECOVERABLE_BYTES",
        )

    private = tmp_path / ".brynse-recovery"
    staged = list(private.glob(".brynse-file-*"))
    assert len(staged) == 1
    assert staged[0].read_bytes() == b"RECOVERABLE_BYTES"
    assert not list(tmp_path.glob("*.mp3"))


def test_parent_directory_sync_failure_blocks_recovery_claim(
    tmp_path: Path,
    completed_fd: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newly created recovery directory must first be durable."""
    original = os.fsync
    count = 0

    def reject_parent(fd: int) -> None:
        nonlocal count
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            count += 1
            if count == 1:
                raise OSError(errno.EIO, "parent directory sync failed")
        original(fd)

    monkeypatch.setattr(publication.os, "fsync", reject_parent)

    with pytest.raises(OSError, match="parent directory sync failed"):
        publication.publish_fd_recoverably(
            source_fd=completed_fd,
            directory=tmp_path,
            stem="Radio",
            extension=".mp3",
        )

    assert count == 1
    assert not list(tmp_path.glob("*.mp3"))
    assert list((tmp_path / ".brynse-recovery").iterdir()) == []


def test_foreign_staging_replacement_is_never_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Substitution at quarantine time must not erase foreign data."""
    from brynse.output import SegmentFileWriter

    original_rename = publication.os.rename
    swapped = False
    foreign = b"FOREIGN_INODE_DO_NOT_DELETE"

    def swap_before_quarantine(
        source: object,
        destination: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal swapped
        if (
            not swapped
            and isinstance(source, str)
            and source.startswith(".brynse-file-")
            and kwargs.get("src_dir_fd") is not None
        ):
            swapped = True
            fd = kwargs["src_dir_fd"]
            os.rename(source, ".original-staging", src_dir_fd=fd, dst_dir_fd=fd)
            replacement = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=fd,
            )
            try:
                os.write(replacement, foreign)
            finally:
                os.close(replacement)
        original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(publication.os, "rename", swap_before_quarantine)

    published = SegmentFileWriter().write(
        directory=tmp_path,
        stem="Radio",
        codec="mp3",
        data=b"OUR_COMPLETE_RECORDING",
    )

    private = tmp_path / ".brynse-recovery"
    assert swapped
    assert published.read_bytes() == b"OUR_COMPLETE_RECORDING"
    assert (private / ".original-staging").read_bytes() == b"OUR_COMPLETE_RECORDING"

    retained = list(private.glob("cleanup-*"))
    assert len(retained) == 1
    assert retained[0].read_bytes() == foreign
    assert not list(private.glob("recovery-*"))


def test_private_cleanup_normal_success_leaves_no_quarantine(
    tmp_path: Path,
) -> None:
    from brynse.output import SegmentFileWriter

    for index in range(3):
        SegmentFileWriter().write(
            directory=tmp_path,
            stem="Radio",
            codec="mp3",
            data=f"AUDIO_{index}".encode(),
        )

    assert list((tmp_path / ".brynse-recovery").iterdir()) == []
