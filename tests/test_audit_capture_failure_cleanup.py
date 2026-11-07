"""Exclusive capture creation and partial-failure cleanup."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

import brynse.integrations.radio.capture as capture_module
from brynse.integrations.radio.capture import RadioSessionCapture


def _create_capture(directory: Path) -> RadioSessionCapture:
    return RadioSessionCapture(
        directory=directory,
        codec="mp3",
        metaint=16000,
        metadata_threshold_seconds=8.0,
        transient_exclusion=True,
        autonomous_boundaries=False,
        source_url="https://example.test/stream",
        source_metadata={"Content-Type": "audio/mpeg"},
    )


@pytest.mark.parametrize("fail_at", [2, 3])
def test_failed_open_removes_only_owned_capture_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_at: int,
) -> None:
    directory = tmp_path / "shared"
    directory.mkdir()

    unrelated = directory / "unrelated.txt"
    unrelated.write_text("KEEP", encoding="utf-8")

    opener = capture_module._private_opener
    calls = 0

    def fail_open(path: str, flags: int) -> int:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError(errno.EIO, "injected open failure")
        return opener(path, flags)

    monkeypatch.setattr(capture_module, "_private_opener", fail_open)

    with pytest.raises(OSError, match="injected open failure"):
        _create_capture(directory)

    assert calls == fail_at
    assert unrelated.read_text(encoding="utf-8") == "KEEP"
    assert sorted(path.name for path in directory.iterdir()) == ["unrelated.txt"]


def test_existing_manifest_is_not_overwritten_or_removed(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "session"
    directory.mkdir()
    manifest = directory / "manifest.json"
    manifest.write_bytes(b"ORIGINAL")

    with pytest.raises(FileExistsError):
        _create_capture(directory)

    assert manifest.read_bytes() == b"ORIGINAL"
    assert sorted(path.name for path in directory.iterdir()) == ["manifest.json"]


def test_manifest_collision_after_precheck_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "session"
    directory.mkdir()

    original_opener = capture_module._private_opener
    injected = False

    def concurrent_manifest(path: str, flags: int) -> int:
        nonlocal injected
        if not injected:
            injected = True
            (directory / "manifest.json").write_bytes(b"COMPETING SESSION")
        return original_opener(path, flags)

    monkeypatch.setattr(
        capture_module,
        "_private_opener",
        concurrent_manifest,
    )

    with pytest.raises(FileExistsError):
        _create_capture(directory)

    assert injected
    assert (directory / "manifest.json").read_bytes() == (b"COMPETING SESSION")
    assert sorted(path.name for path in directory.iterdir()) == ["manifest.json"]


def test_cleanup_preserves_replaced_manifest_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "shared"
    directory.mkdir()

    manifest = directory / "manifest.json"
    original_opener = capture_module._private_opener
    calls = 0

    def replaced_during_creation(path: str, flags: int) -> int:
        nonlocal calls
        calls += 1

        if calls == 3:
            assert manifest.exists()
            manifest.unlink()
            manifest.write_bytes(b"FOREIGN MANIFEST")
            raise OSError(errno.EIO, "injected failure after replacement")

        return original_opener(path, flags)

    monkeypatch.setattr(capture_module, "_private_opener", replaced_during_creation)

    with pytest.raises(OSError, match="injected failure after replacement"):
        _create_capture(directory)

    assert calls == 3
    assert manifest.read_bytes() == b"FOREIGN MANIFEST"
    assert sorted(path.name for path in directory.iterdir()) == ["manifest.json"]
