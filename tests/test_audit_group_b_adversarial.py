"""Adversarial Group B contracts: concurrent reconciliation and error semantics."""

from __future__ import annotations

import errno
import logging
import os
import threading
from pathlib import Path

import pytest

from brynse import _recoverable_publication as publication
from brynse.output import SegmentFileWriter


def _published_with_recovery(tmp_path: Path) -> tuple[Path, Path]:
    private = publication.ensure_private_recovery_directory(tmp_path)
    published = tmp_path / "Radio.mp3"
    published.write_bytes(b"DO_NOT_LOSE_AUDIO")
    recovery = private / ("recovery-" + "a" * 32 + ".mp3")
    os.link(published, recovery)
    return published, recovery


def test_two_concurrent_reconcilers_preserve_published_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlapping cleanup cannot remove a published recording or another inode."""
    published, recovery = _published_with_recovery(tmp_path)
    start = threading.Event()
    proceed = threading.Event()
    original = publication._cleanup_verified_link
    calls = 0
    lock = threading.Lock()

    def pause_first(fd: int, name: str, identity: tuple[int, int]) -> bool:
        nonlocal calls
        with lock:
            calls += 1
            first = calls == 1
        if first:
            start.set()
            assert proceed.wait(5), "concurrent reconciler did not run"
        return original(fd, name, identity)

    monkeypatch.setattr(publication, "_cleanup_verified_link", pause_first)
    reports: list[publication.ReconciliationReport] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            reports.append(publication.reconcile_recovery_directory(tmp_path))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=run)
    first.start()
    try:
        assert start.wait(5), "first reconciler did not reach cleanup"
        second = threading.Thread(target=run)
        second.start()
        second.join(5)
        assert not second.is_alive(), "second reconciler hung"
    finally:
        proceed.set()
        first.join(5)
    assert not first.is_alive()
    assert not errors, errors
    assert published.read_bytes() == b"DO_NOT_LOSE_AUDIO"
    assert not recovery.exists()
    assert sum(len(r.removed) for r in reports) == 1


def test_recovery_replacement_during_cleanup_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced private name must never be unlinked as the old inode."""
    published, recovery = _published_with_recovery(tmp_path)
    original = publication._cleanup_verified_link

    def replace_then_cleanup(fd: int, name: str, identity: tuple[int, int]) -> bool:
        recovery.unlink()
        recovery.write_bytes(b"FOREIGN_RECORDING")
        return original(fd, name, identity)

    monkeypatch.setattr(publication, "_cleanup_verified_link", replace_then_cleanup)
    report = publication.reconcile_recovery_directory(tmp_path)
    assert report.removed == ()
    assert published.read_bytes() == b"DO_NOT_LOSE_AUDIO"
    assert any(p.read_bytes() == b"FOREIGN_RECORDING" for p in recovery.parent.glob("cleanup-*"))


def test_post_publication_fsync_failure_reports_both_durable_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncertain directory fsync cannot trigger a misleading retry."""
    real_fsync = publication.os.fsync
    output_inode = tmp_path.stat().st_ino

    def fail_output(fd: int) -> None:
        # Only fail after final file is linked, not during initialization.
        if os.fstat(fd).st_ino == output_inode and (tmp_path / "Radio.mp3").exists():
            raise OSError(errno.EIO, "injected final directory sync error")
        real_fsync(fd)

    monkeypatch.setattr(publication.os, "fsync", fail_output)
    with pytest.raises(publication.PublicationDurabilityError) as caught:
        SegmentFileWriter().write(
            directory=tmp_path, stem="Radio", codec="mp3", data=b"DO_NOT_LOSE_AUDIO"
        )
    exc = caught.value
    assert exc.published_path.read_bytes() == b"DO_NOT_LOSE_AUDIO"
    assert exc.recovery_path.read_bytes() == b"DO_NOT_LOSE_AUDIO"
    assert "durability unconfirmed" in str(exc)
    assert str(exc.published_path) in str(exc)
    assert str(exc.recovery_path) in str(exc)


def test_reconciliation_fsync_failure_preserves_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    published, recovery = _published_with_recovery(tmp_path)
    real_fsync = publication.os.fsync
    output_inode = tmp_path.stat().st_ino

    def fail_output(fd: int) -> None:
        if os.fstat(fd).st_ino == output_inode:
            raise OSError(errno.EIO, "injected reconciliation sync error")
        real_fsync(fd)

    monkeypatch.setattr(publication.os, "fsync", fail_output)
    with caplog.at_level(logging.WARNING, logger=publication.__name__):
        report = publication.reconcile_recovery_directory(tmp_path)
    assert report.removed == ()
    assert recovery.name in report.retained
    assert published.read_bytes() == b"DO_NOT_LOSE_AUDIO"
    assert recovery.read_bytes() == b"DO_NOT_LOSE_AUDIO"
    assert "durability unavailable" in caplog.text


@pytest.mark.parametrize("error_kind", ["segment", "durability"])
def test_radio_runner_translates_publication_errors_without_traceback_contract(
    tmp_path: Path, error_kind: str
) -> None:
    """The radio CLI consumes RippingRunError, not raw output exceptions."""
    from brynse.integrations.radio.runner import RippingRunConfig, RippingRunError, RippingRunner
    from brynse.output import SegmentFileWriteError
    from brynse.source import BinaryIOStreamSource

    if error_kind == "segment":
        expected_error: Exception = SegmentFileWriteError(
            f"output storage unavailable; completed staging: {tmp_path}/recovery.mp3"
        )
    else:
        expected_error = publication.PublicationDurabilityError(
            tmp_path / ".brynse-recovery" / "recovery.mp3",
            tmp_path / "Radio.mp3",
            OSError(errno.EIO, "directory fsync error"),
        )

    class BrokenStream:
        def read(self, size: int = -1) -> bytes:
            raise expected_error

        def close(self) -> None:
            pass

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            codec="mp3",
        ),
        stream_opener=lambda url: BinaryIOStreamSource(
            BrokenStream(),
            metadata={"Content-Type": "audio/mpeg", "icy-metaint": "417"},
        ),
    )
    with pytest.raises(RippingRunError) as caught:
        runner.run()
    if error_kind == "durability":
        assert "publication durability unconfirmed" in str(caught.value)
        assert "published at" in str(caught.value)
        assert "durability unconfirmed" in str(caught.value)
    else:
        assert "recording publication failed" in str(caught.value)
        assert "completed staging" in str(caught.value)


def test_independent_process_reconcilers_never_remove_published_audio(tmp_path: Path) -> None:
    """Two non-root-compatible CLI-style processes may reconcile one directory."""
    import subprocess
    import sys

    published, recovery = _published_with_recovery(tmp_path)
    script = """
import json
import sys
from pathlib import Path
from brynse._recoverable_publication import reconcile_recovery_directory
report = reconcile_recovery_directory(Path(sys.argv[1]))
print(json.dumps({"removed": list(report.removed), "retained": list(report.retained)}))
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    for proc in processes:
        out, err = proc.communicate(timeout=15)
        assert proc.returncode == 0, (out, err)
    assert published.read_bytes() == b"DO_NOT_LOSE_AUDIO"
    assert not recovery.exists()
    assert not list((tmp_path / ".brynse-recovery").glob("cleanup-*"))
