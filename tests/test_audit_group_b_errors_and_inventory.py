"""First consolidated Group B regression contracts."""

from __future__ import annotations

import errno
import logging
from pathlib import Path

import pytest

from brynse import _recoverable_publication as publication
from brynse import generic_cli
from brynse.output import SegmentFileWriter


def test_generic_preflight_rejects_bad_recovery_before_reading_input(tmp_path: Path) -> None:
    private = tmp_path / ".brynse-recovery"
    private.mkdir(mode=0o755)
    private.chmod(0o755)
    touched = []

    def chunks():
        touched.append(True)
        yield b"stream bytes"

    with pytest.raises(generic_cli.GenericCliError, match="output storage incompatible"):
        generic_cli._run_generic_pipeline(
            chunks=chunks(),
            codec="mp3",
            output_directory=tmp_path,
        )
    assert not touched


def test_recovery_inventory_reports_artifacts_without_deleting(tmp_path: Path) -> None:
    private = publication.ensure_private_recovery_directory(tmp_path)
    (private / "recovery-example.mp3").write_bytes(b"recoverable audio")
    (private / ".brynse-file-old.tmp").write_bytes(b"partial audio")
    assert publication.inspect_recovery_directory(tmp_path) == (
        ".brynse-file-old.tmp",
        "recovery-example.mp3",
    )
    assert (private / "recovery-example.mp3").read_bytes() == b"recoverable audio"


def test_recovery_inventory_rejects_bad_mode(tmp_path: Path) -> None:
    private = publication.ensure_private_recovery_directory(tmp_path)
    private.chmod(0o755)
    with pytest.raises(PermissionError, match="ownership/mode"):
        publication.inspect_recovery_directory(tmp_path)


def test_published_but_unsynced_error_has_distinct_message(tmp_path: Path) -> None:
    exc = publication.PublicationDurabilityError(
        tmp_path / ".brynse-recovery" / "recovery-a.mp3",
        tmp_path / "Radio.mp3",
        OSError(errno.EIO, "injected directory fsync failure"),
    )
    assert "published at" in str(exc)
    assert "durability unconfirmed" in str(exc)
    assert str(exc.published_path) in str(exc)
    assert str(exc.recovery_path) in str(exc)
    assert exc.errno == errno.EIO


def test_cleanup_failure_is_logged_without_retry_or_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original = publication._cleanup_verified_link
    did_fail = False

    def fail_first(*args, **kwargs):
        nonlocal did_fail
        if not did_fail:
            did_fail = True
            raise OSError(errno.EIO, "injected cleanup error")
        return original(*args, **kwargs)

    monkeypatch.setattr(publication, "_cleanup_verified_link", fail_first)
    with caplog.at_level(logging.WARNING, logger=publication.__name__):
        result = SegmentFileWriter().write(
            directory=tmp_path, stem="Radio", codec="mp3", data=b"recording"
        )
    assert did_fail
    assert result.read_bytes() == b"recording"
    assert "retained recovery artifact" in caplog.text
