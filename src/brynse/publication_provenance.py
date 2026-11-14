"""Durable source-byte provenance for confirmed F-2 publications."""

import json
import os
import re
import tempfile
from pathlib import Path

from brynse._recoverable_publication import (
    ensure_private_recovery_directory,
    publish_private_file,
)
from brynse.integrations.radio.session_output import PublicationReceipt
from brynse.publication_coverage import PublicationCoverageLedger


def _range(start: int, end: int, path: Path) -> dict[str, object]:
    return {
        "start_offset": start,
        "end_offset": end,
        "file": path.name,
    }


def _value(obj: object, attribute: str) -> object:
    value = getattr(obj, attribute, None)
    return getattr(value, "value", value)


def publish_provenance(
    *,
    receipt: PublicationReceipt,
    directory: Path,
    session_id: str,
    source_id: str,
) -> Path:
    """Persist a transition receipt before the caller discards source bytes.

    A failed publication is propagated. The private staging artifact is
    retained for recovery, rather than being silently deleted.
    """
    if re.fullmatch(r"[0-9a-f]{32}", session_id) is None:
        raise ValueError("session_id must be a lowercase UUID hex string")
    if not source_id.strip():
        raise ValueError("source_id must not be empty")

    boundary = receipt.transition.boundary
    split = boundary.split
    temporal = getattr(boundary, "temporal", None)

    outgoing = receipt.outgoing_range
    unowned = receipt.unowned_range

    if (unowned is None) != (receipt.unowned_path is None):
        raise ValueError("unowned range and path must both be present or absent")

    record: dict[str, object] = {
        "schema_version": 1,
        "session_id": session_id,
        "source_id": source_id,
        "decision_kind": _value(split, "kind"),
        "temporal_kind": (_value(temporal, "kind") if temporal is not None else None),
        "outgoing_end_seconds": (
            getattr(temporal, "outgoing_end_seconds", None) if temporal is not None else None
        ),
        "incoming_start_seconds": (
            getattr(temporal, "incoming_start_seconds", None) if temporal is not None else None
        ),
        "outgoing_label": receipt.transition.outgoing.label,
        "incoming_label": getattr(receipt.transition.incoming, "label", None),
        "outgoing": _range(
            outgoing.start_offset,
            outgoing.end_offset,
            receipt.outgoing_path,
        ),
        "incoming_start_offset": receipt.incoming_start,
        "unowned": (
            _range(
                unowned.start_offset,
                unowned.end_offset,
                receipt.unowned_path,
            )
            if unowned is not None and receipt.unowned_path is not None
            else None
        ),
    }

    end = unowned.end_offset if unowned is not None else outgoing.end_offset
    stem = f"brynse-provenance-{session_id}-{outgoing.start_offset:012d}-{end:012d}"
    return _publish_record(directory=directory, stem=stem, record=record)


def _publish_record(*, directory: Path, stem: str, record: dict[str, object]) -> Path:
    """Publish one JSON inode using the same fail-safe publication as audio."""
    data = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")

    private_dir = ensure_private_recovery_directory(directory)
    fd, temporary = tempfile.mkstemp(
        prefix=".brynse-provenance-",
        suffix=".json",
        dir=private_dir,
    )
    temporary_path = Path(temporary)

    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())

    return publish_private_file(
        temp_path=temporary_path,
        directory=directory,
        stem=stem,
        extension=".json",
    )


def publish_terminal_provenance(
    *,
    ledger: PublicationCoverageLedger,
    directory: Path,
    session_id: str,
    source_id: str,
    spool_start: int,
    spool_end: int,
    spool_paths: tuple[Path, ...],
) -> Path:
    """Durably declare exact retained vs published bytes at normal EOF/stop.

    Spool paths are informational, never proof of publication or safe cleanup.
    """
    if re.fullmatch(r"[0-9a-f]{32}", session_id) is None:
        raise ValueError("session_id must be a lowercase UUID hex string")
    if not source_id.strip():
        raise ValueError("source_id must not be empty")
    if ledger.start_offset != 0:
        raise ValueError("terminal provenance must account from source offset zero")
    ledger.verify_complete(end_offset=spool_end)
    tail = ledger.retained_tail
    if tail is not None and not (spool_start <= tail.start < tail.end == spool_end):
        raise ValueError("retained tail not contained in spool offsets")
    if tail is None and spool_end != ledger.covered_end:
        raise ValueError("terminal coverage incomplete")
    record: dict[str, object] = {
        "schema_version": 1,
        "record_type": "terminal_coverage",
        "session_id": session_id,
        "source_id": source_id,
        "source_start_offset": 0,
        "source_end_offset": spool_end,
        "spool_start_offset": spool_start,
        "spool_end_offset": spool_end,
        "retained_tail": (
            {"start_offset": tail.start, "end_offset": tail.end} if tail is not None else None
        ),
        "spool_chunks": [p.name for p in spool_paths],
        "published": [
            {
                "kind": item.kind,
                "start_offset": item.start,
                "end_offset": item.end,
                "file": item.path.name,
                "declared_overlap": item.declared_overlap,
            }
            for item in ledger.published
        ],
    }
    return _publish_record(
        directory=directory,
        stem=f"brynse-terminal-{session_id}",
        record=record,
    )
