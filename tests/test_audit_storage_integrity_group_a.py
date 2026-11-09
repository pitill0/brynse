"""Group A: output preflight, directory identity and spool recoverability."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from brynse import _recoverable_publication as publication
from brynse.streaming_spool import StreamingSpool


def test_preflight_links_into_final_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = publication._link_fd
    seen_final: list[str] = []

    def observe(fd: int, directory_fd: int, name: str) -> None:
        if name.startswith(".brynse-probe-out-"):
            seen_final.append(name)
            assert os.fstat(directory_fd).st_ino == tmp_path.stat().st_ino
        real(fd, directory_fd, name)

    monkeypatch.setattr(publication, "_link_fd", observe)
    assert publication.preflight_publication_directory(tmp_path) == tmp_path
    assert len(seen_final) == 1
    assert not list(tmp_path.glob(".brynse-probe-out-*"))


def test_preflight_rejects_output_link_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = publication._link_fd

    def deny_final(fd: int, directory_fd: int, name: str) -> None:
        if name.startswith(".brynse-probe-out-"):
            raise PermissionError(errno.EACCES, "final output readonly")
        real(fd, directory_fd, name)

    monkeypatch.setattr(publication, "_link_fd", deny_final)
    with pytest.raises(PermissionError, match="final output readonly"):
        publication.preflight_publication_directory(tmp_path)
    assert not list(tmp_path.glob(".brynse-probe-out-*"))


def test_directory_identity_detects_symlink_switch(tmp_path: Path) -> None:
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    target = tmp_path / "output"
    target.symlink_to(original, target_is_directory=True)
    identity = publication.OutputDirectoryIdentity.capture(target)
    identity.verify()
    target.unlink()
    target.symlink_to(replacement, target_is_directory=True)
    with pytest.raises(OSError, match="output directory changed"):
        identity.verify()


def test_directory_identity_detects_replaced_inode(tmp_path: Path) -> None:
    target = tmp_path / "output"
    target.mkdir()
    identity = publication.OutputDirectoryIdentity.capture(target)
    target.rename(tmp_path / "previous")
    target.mkdir()
    with pytest.raises(OSError, match="output directory changed"):
        identity.verify()


def test_durable_directory_syncs_each_created_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []
    original = publication.os.fsync

    def observe(fd: int) -> None:
        assert os.fstat(fd).st_mode
        seen.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        original(fd)

    monkeypatch.setattr(publication.os, "fsync", observe)
    target = tmp_path / "nested" / "output"
    publication.mkdir_durably(target)
    assert target.is_dir()
    assert seen == [tmp_path, tmp_path / "nested"]


def test_spool_retained_on_failed_publication(tmp_path: Path) -> None:
    spool = StreamingSpool(directory=tmp_path / "spool", min_free_bytes=None)
    spool.append(b"audio bytes")
    retained = spool.recovery_paths
    assert retained
    spool.close(preserve=True)
    assert all(p.exists() for p in retained)
    assert b"audio bytes" in b"".join(p.read_bytes() for p in retained)


def test_spool_normal_cleanup_unchanged(tmp_path: Path) -> None:
    spool = StreamingSpool(directory=tmp_path / "spool", min_free_bytes=None)
    spool.append(b"audio bytes")
    paths = spool.recovery_paths
    spool.close()
    assert all(not p.exists() for p in paths)


def test_active_session_publication_fails_closed_on_swapped_destination(tmp_path: Path) -> None:
    """The final link must never be created in a replacement output directory."""
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    destination = tmp_path / "chosen"
    destination.symlink_to(original, target_is_directory=True)
    source = original / "audio.mp3"
    source.write_bytes(b"complete audio")
    identity = publication.OutputDirectoryIdentity.capture(destination)
    token = publication._ACTIVE_OUTPUT_IDENTITY.set(identity)
    try:
        destination.unlink()
        destination.symlink_to(replacement, target_is_directory=True)
        with (
            source.open("rb") as handle,
            pytest.raises(OSError, match="changed since preflight"),
        ):
            publication.publish_fd_recoverably(
                source_fd=handle.fileno(),
                directory=destination,
                stem="recorded",
                extension=".mp3",
            )
    finally:
        publication._ACTIVE_OUTPUT_IDENTITY.reset(token)
    assert not list(replacement.iterdir())
    assert source.read_bytes() == b"complete audio"


def test_active_session_checks_opened_inode_not_just_prior_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink swap at os.open cannot race a previous pathname check."""
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    destination = tmp_path / "chosen"
    destination.symlink_to(original, target_is_directory=True)
    identity = publication.OutputDirectoryIdentity.capture(destination)
    real_open = publication.os.open

    def swapped_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if path == original:
            # Simulate a race after resolution but before descriptor opening.
            original.rename(tmp_path / "moved")
            replacement.rename(original)
        return real_open(path, flags, *args, **kwargs)

    token = publication._ACTIVE_OUTPUT_IDENTITY.set(identity)
    try:
        monkeypatch.setattr(publication.os, "open", swapped_open)
        with pytest.raises(OSError, match="changed since preflight"):
            publication._open_checked_output_directory(destination)
    finally:
        publication._ACTIVE_OUTPUT_IDENTITY.reset(token)


def test_preserved_spool_syncs_chunk_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spool = StreamingSpool(directory=tmp_path / "spool", storage_chunk_size=4)
    spool.append(b"abcdefgh")
    paths = spool.recovery_paths
    sync_modes: list[str] = []
    real_fsync = os.fsync

    def observe(fd: int) -> None:
        import stat

        mode = os.fstat(fd).st_mode
        sync_modes.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", observe)
    spool.sync_for_recovery()
    spool.close(preserve=True)
    assert sync_modes.count("file") == len(paths)
    assert sync_modes[-1] == "directory"
    assert b"".join(p.read_bytes() for p in paths) == b"abcdefgh"


def test_failed_spool_sync_does_not_delete_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spool = StreamingSpool(directory=tmp_path / "spool")
    spool.append(b"encoded")
    paths = spool.recovery_paths

    def fail_sync(_fd: int) -> None:
        raise OSError(errno.EIO, "simulated fsync failure")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="simulated fsync failure"):
        spool.sync_for_recovery()
    spool.close(preserve=True)
    assert [p.read_bytes() for p in paths] == [b"encoded"]
