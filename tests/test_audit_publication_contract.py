"""Adversarial contracts for publishing completed audio files."""

from __future__ import annotations

import contextlib
import errno
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from brynse._recoverable_publication import ensure_private_recovery_directory, publish_private_file
from brynse.output import SegmentFileWriteError, SegmentFileWriter


def test_hardlink_failure_keeps_complete_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed final publication retains completed audio and backup."""
    import brynse._recoverable_publication as publication

    payload = b"COMPLETE_RECORDING"
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
            raise OSError(errno.EPERM, "final hardlink denied")
        original_link(source_fd, directory_fd, name)

    monkeypatch.setattr(publication, "_link_fd", deny_final_link)

    with pytest.raises(SegmentFileWriteError):
        SegmentFileWriter().write(
            directory=tmp_path,
            stem="Radio",
            codec="mp3",
            data=payload,
        )

    private = tmp_path / ".brynse-recovery"
    staging = list(private.glob(".brynse-file-*"))
    backups = list(private.glob("recovery-*.mp3"))

    assert final_attempts == 1
    assert len(staging) == 1
    assert len(backups) == 1
    assert staging[0].read_bytes() == payload
    assert backups[0].read_bytes() == payload
    assert not list(tmp_path.glob("*.mp3"))


def test_unlink_failure_after_publication_is_not_retryable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once published, a cleanup failure must not invite duplication."""
    original_unlink = os.unlink
    injected = False

    def fail_once(path: object, *args: object, **kwargs: object) -> None:
        nonlocal injected
        if str(path).startswith("cleanup-") and not injected:
            injected = True
            raise OSError(errno.EIO, "injected cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr("brynse._recoverable_publication.os.unlink", fail_once)

    published = SegmentFileWriter().write(
        directory=tmp_path,
        stem="Radio",
        codec="mp3",
        data=b"COMPLETE_RECORDING",
    )

    assert injected
    assert published == tmp_path / "Radio.mp3"
    assert published.read_bytes() == b"COMPLETE_RECORDING"
    assert len(list(tmp_path.glob("*.mp3"))) == 1


def test_completed_recording_respects_standard_output_mode(
    tmp_path: Path,
) -> None:
    """An ordinary completed recording honors umask 022, not 0600."""
    script = """
import os
import stat
import sys
from pathlib import Path
from brynse.output import SegmentFileWriter

os.umask(0o022)
target = SegmentFileWriter().write(
    directory=Path(sys.argv[1]),
    stem="Radio",
    codec="mp3",
    data=b"COMPLETE_RECORDING",
)
print(oct(stat.S_IMODE(target.stat().st_mode)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.stdout.strip() == "0o644"


def test_temporary_symlink_swap_cannot_publish_foreign_content(
    tmp_path: Path,
) -> None:
    """Changing the source pathname must not publish a foreign inode."""
    directory = tmp_path / "output"
    directory.mkdir()

    external = tmp_path / "foreign.txt"
    external.write_bytes(b"FOREIGN_SECRET")

    temporary = ensure_private_recovery_directory(directory) / ".incoming.tmp"
    temporary.write_bytes(b"OUR_RECORDING")

    from brynse import _recoverable_publication as publication

    original_link = publication._link_fd
    swapped = False

    def swap_then_link(source: int, target: int, name: str) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            temporary.unlink()
            temporary.symlink_to(external)
        original_link(source, target, name)

    published = None
    with (
        patch("brynse._recoverable_publication._link_fd", side_effect=swap_then_link),
        contextlib.suppress(OSError),
    ):
        published = publish_private_file(
            temp_path=temporary,
            directory=directory,
            stem="Radio",
            extension=".mp3",
        )

    assert swapped
    assert external.read_bytes() == b"FOREIGN_SECRET"

    output = directory / "Radio.mp3"
    assert not output.is_symlink()

    if published is not None:
        assert published == output
        assert published.read_bytes() == b"OUR_RECORDING"
    elif output.exists():
        assert output.read_bytes() == b"OUR_RECORDING"


def test_existing_destination_symlink_is_not_overwritten(
    tmp_path: Path,
) -> None:
    """An existing destination symlink must not redirect publication."""
    directory = tmp_path / "output"
    directory.mkdir()

    external = tmp_path / "foreign.txt"
    external.write_bytes(b"FOREIGN_SECRET")

    destination = directory / "Radio.mp3"
    destination.symlink_to(external)

    temporary = ensure_private_recovery_directory(directory) / ".incoming.tmp"
    temporary.write_bytes(b"OUR_RECORDING")

    published = publish_private_file(
        temp_path=temporary,
        directory=directory,
        stem="Radio",
        extension=".mp3",
    )

    assert destination.is_symlink()
    assert external.read_bytes() == b"FOREIGN_SECRET"
    assert published == directory / "Radio-2.mp3"
    assert published.read_bytes() == b"OUR_RECORDING"
    assert not published.is_symlink()


def test_cleanup_cannot_unlink_replaced_foreign_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign replacement must survive descriptor-based publication."""
    temporary = ensure_private_recovery_directory(tmp_path) / ".incoming.tmp"
    temporary.write_bytes(b"OUR_RECORDING")
    retained = tmp_path / ".retained-original"
    foreign_payload = b"FOREIGN_REPLACEMENT"

    from brynse import _recoverable_publication as publication

    original_link = publication._link_fd
    swapped = False

    def substitute_before_link(src: int, dst: int, name: str) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            temporary.rename(retained)
            temporary.write_bytes(foreign_payload)
        original_link(src, dst, name)

    monkeypatch.setattr(
        "brynse._recoverable_publication._link_fd",
        substitute_before_link,
    )

    published = publish_private_file(
        temp_path=temporary,
        directory=tmp_path,
        stem="Radio",
        extension=".mp3",
    )

    assert swapped
    assert published.read_bytes() == b"OUR_RECORDING"
    assert retained.read_bytes() == b"OUR_RECORDING"
    assert any(
        entry.read_bytes() == foreign_payload for entry in temporary.parent.glob("cleanup-*")
    )
    assert not temporary.exists()
