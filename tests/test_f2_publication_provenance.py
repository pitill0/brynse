"""F-2 provenance must be durable and failure-visible."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import brynse.publication_provenance as provenance
from brynse.integrations.radio.session_output import PublicationReceipt
from brynse.models import SegmentByteRange

SESSION_ID = "0123456789abcdef0123456789abcdef"


def _receipt(tmp_path: Path) -> PublicationReceipt:
    outgoing = tmp_path / "song.mp3"
    unowned = tmp_path / "unowned.mp3"
    outgoing.write_bytes(b"song")
    unowned.write_bytes(b"excluded")

    return PublicationReceipt(
        transition=SimpleNamespace(
            outgoing=SimpleNamespace(label="Song A"),
            incoming=SimpleNamespace(label="Song B"),
            boundary=SimpleNamespace(
                split=SimpleNamespace(kind="exclusion"),
                temporal=SimpleNamespace(
                    kind="exclusion",
                    outgoing_end_seconds=10.0,
                    incoming_start_seconds=12.0,
                ),
            ),
        ),
        outgoing_range=SegmentByteRange(100, 400),
        outgoing_path=outgoing,
        incoming_start=500,
        unowned_range=SegmentByteRange(400, 500),
        unowned_path=unowned,
    )


def test_persistent_receipt_contains_exact_source_ranges(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)

    path = provenance.publish_provenance(
        receipt=receipt,
        directory=tmp_path,
        session_id=SESSION_ID,
        source_id="test-station",
    )

    assert path.is_file()
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["session_id"] == SESSION_ID
    assert record["source_id"] == "test-station"
    assert record["outgoing"] == {
        "start_offset": 100,
        "end_offset": 400,
        "file": "song.mp3",
    }
    assert record["unowned"] == {
        "start_offset": 400,
        "end_offset": 500,
        "file": "unowned.mp3",
    }
    assert record["incoming_start_offset"] == 500
    assert record["decision_kind"] == "exclusion"
    assert record["outgoing_end_seconds"] == 10.0
    assert record["incoming_start_seconds"] == 12.0


def test_provenance_failure_is_not_silenced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(tmp_path)

    def fail_publish(**_kwargs: object) -> Path:
        raise OSError("injected provenance failure")

    monkeypatch.setattr(provenance, "publish_private_file", fail_publish)

    with pytest.raises(OSError, match="injected provenance"):
        provenance.publish_provenance(
            receipt=receipt,
            directory=tmp_path,
            session_id=SESSION_ID,
            source_id="test-station",
        )

    private_dir = tmp_path / ".brynse-recovery"
    staged = list(private_dir.glob(".brynse-provenance-*.json"))
    assert len(staged) == 1
    assert json.loads(staged[0].read_text())["session_id"] == SESSION_ID
    assert receipt.outgoing_path.read_bytes() == b"song"
    assert receipt.unowned_path.read_bytes() == b"excluded"


def test_invalid_session_identity_rejected(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)

    with pytest.raises(ValueError, match="session_id"):
        provenance.publish_provenance(
            receipt=receipt,
            directory=tmp_path,
            session_id="../other-session",
            source_id="test-station",
        )

    assert not list(tmp_path.glob("brynse-provenance-*.json"))


def test_json_publication_collision_preserves_both_receipts(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    args = {
        "receipt": receipt,
        "directory": tmp_path,
        "session_id": SESSION_ID,
        "source_id": "test-station",
    }

    first = provenance.publish_provenance(**args)
    second = provenance.publish_provenance(**args)

    assert first != second
    assert first.suffix == second.suffix == ".json"
    assert first.read_bytes() == second.read_bytes()
    assert first.is_file() and second.is_file()


def test_arbitrary_extension_remains_rejected(tmp_path: Path) -> None:
    from brynse._recoverable_publication import (
        ensure_private_recovery_directory,
        publish_private_file,
    )

    private = ensure_private_recovery_directory(tmp_path)
    staged = private / ".test-unsupported"
    staged.write_bytes(b"unchanged")

    with pytest.raises(ValueError, match="unsupported output extension"):
        publish_private_file(
            temp_path=staged,
            directory=tmp_path,
            stem="unsupported",
            extension=".exe",
        )

    assert staged.read_bytes() == b"unchanged"
    assert not (tmp_path / "unsupported.exe").exists()


def test_terminal_receipt_accounts_published_and_retained(tmp_path: Path) -> None:
    from brynse.publication_coverage import PublicationCoverageLedger

    output = tmp_path / "first.mp3"
    output.write_bytes(b"song")
    ledger = PublicationCoverageLedger(start_offset=0)
    ledger.record_published(kind="track", start=0, end=400, path=output)
    ledger.declare_retained_tail(start=400, end=900)

    path = provenance.publish_terminal_provenance(
        ledger=ledger,
        directory=tmp_path,
        session_id=SESSION_ID,
        source_id="sha256:" + "f" * 64,
        spool_start=400,
        spool_end=900,
        spool_paths=(tmp_path / "retained.chunk",),
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["source_start_offset"] == 0
    assert record["source_end_offset"] == 900
    assert record["retained_tail"] == {"start_offset": 400, "end_offset": 900}
    assert record["published"] == [
        {
            "kind": "track",
            "start_offset": 0,
            "end_offset": 400,
            "file": "first.mp3",
            "declared_overlap": False,
        }
    ]
    assert record["spool_chunks"] == ["retained.chunk"]


def test_terminal_provenance_rejects_uncovered_spool_end(tmp_path: Path) -> None:
    from brynse.publication_coverage import CoverageError, PublicationCoverageLedger

    ledger = PublicationCoverageLedger(start_offset=0)
    with pytest.raises(CoverageError, match="undeclared tail"):
        provenance.publish_terminal_provenance(
            ledger=ledger,
            directory=tmp_path,
            session_id=SESSION_ID,
            source_id="source",
            spool_start=0,
            spool_end=100,
            spool_paths=(),
        )
    assert not list(tmp_path.glob("brynse-terminal-*.json"))


def test_json_recovery_link_reconciles_safely(tmp_path: Path) -> None:
    from brynse._recoverable_publication import (
        ensure_private_recovery_directory,
        reconcile_recovery_directory,
    )

    private = ensure_private_recovery_directory(tmp_path)
    published = tmp_path / "brynse-provenance.json"
    published.write_text('{"ok":true}')
    recovery = private / ("recovery-" + "a" * 32 + ".json")
    recovery.hardlink_to(published)
    report = reconcile_recovery_directory(tmp_path)
    assert report.removed == (recovery.name,)
    assert report.retained == ()
    assert published.read_text() == '{"ok":true}'
    assert not recovery.exists()
