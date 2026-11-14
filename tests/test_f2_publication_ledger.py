"""F-2: exact publication coverage, without tolerance or silent loss."""

from pathlib import Path

import pytest

from brynse.publication_coverage import CoverageError, PublicationCoverageLedger


def _artifact(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"test-published-audio")
    return path


def test_track_exclusion_track_forms_exact_partition(tmp_path: Path) -> None:
    ledger = PublicationCoverageLedger(start_offset=100)

    ledger.record_published(
        kind="track",
        start=100,
        end=400,
        path=_artifact(tmp_path, "track1.mp3"),
    )
    ledger.record_published(
        kind="unowned",
        start=400,
        end=550,
        path=_artifact(tmp_path, "unowned.mp3"),
    )
    ledger.record_published(
        kind="track",
        start=550,
        end=900,
        path=_artifact(tmp_path, "track2.mp3"),
    )

    ledger.verify_complete(end_offset=900)


def test_omitted_exclusion_is_rejected(tmp_path: Path) -> None:
    ledger = PublicationCoverageLedger(start_offset=100)

    ledger.record_published(
        kind="track",
        start=100,
        end=400,
        path=_artifact(tmp_path, "track1.mp3"),
    )

    with pytest.raises(CoverageError, match="gap"):
        ledger.record_published(
            kind="track",
            start=550,
            end=900,
            path=_artifact(tmp_path, "track2.mp3"),
        )


def test_undeclared_overlap_is_rejected(tmp_path: Path) -> None:
    ledger = PublicationCoverageLedger(start_offset=100)

    ledger.record_published(
        kind="track",
        start=100,
        end=500,
        path=_artifact(tmp_path, "track1.mp3"),
    )

    with pytest.raises(CoverageError, match="overlap"):
        ledger.record_published(
            kind="track",
            start=490,
            end=900,
            path=_artifact(tmp_path, "track2.mp3"),
        )


def test_declared_crossfade_can_overlap(tmp_path: Path) -> None:
    ledger = PublicationCoverageLedger(start_offset=100)

    ledger.record_published(
        kind="track",
        start=100,
        end=500,
        path=_artifact(tmp_path, "track1.mp3"),
    )
    ledger.record_published(
        kind="track",
        start=490,
        end=900,
        path=_artifact(tmp_path, "track2.mp3"),
        declared_overlap=True,
    )

    ledger.verify_complete(end_offset=900)


def test_terminal_tail_must_be_declared(tmp_path: Path) -> None:
    ledger = PublicationCoverageLedger(start_offset=100)

    ledger.record_published(
        kind="track",
        start=100,
        end=500,
        path=_artifact(tmp_path, "track1.mp3"),
    )

    with pytest.raises(CoverageError, match="tail"):
        ledger.verify_complete(end_offset=900)

    ledger.declare_retained_tail(start=500, end=900)
    ledger.verify_complete(end_offset=900)


def test_terminal_declaration_cannot_hide_a_gap(tmp_path: Path) -> None:
    ledger = PublicationCoverageLedger(start_offset=100)

    ledger.record_published(
        kind="track",
        start=100,
        end=400,
        path=_artifact(tmp_path, "track1.mp3"),
    )

    with pytest.raises(CoverageError, match="gap"):
        ledger.declare_retained_tail(start=500, end=900)


@pytest.mark.parametrize(
    "invalid_kind",
    ["missing", "empty", "symlink", "directory"],
)
def test_rejected_artifact_preserves_ledger_state(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    """An invalid artifact cannot advance the published source-byte cursor."""
    ledger = PublicationCoverageLedger(start_offset=100)
    valid = _artifact(tmp_path, "first.mp3")

    ledger.record_published(
        kind="track",
        start=100,
        end=400,
        path=valid,
    )

    invalid = tmp_path / "invalid.mp3"

    if invalid_kind == "empty":
        invalid.touch()
    elif invalid_kind == "symlink":
        invalid.symlink_to(valid)
    elif invalid_kind == "directory":
        invalid.mkdir()

    published_before = ledger.published

    with pytest.raises(CoverageError, match="missing|unsafe|regular"):
        ledger.record_published(
            kind="unowned",
            start=400,
            end=500,
            path=invalid,
        )

    assert ledger.covered_end == 400
    assert ledger.published == published_before
    assert ledger.retained_tail is None

    # The rejected interval remains an actual gap.
    with pytest.raises(CoverageError, match="gap"):
        ledger.record_published(
            kind="track",
            start=500,
            end=700,
            path=_artifact(tmp_path, "second.mp3"),
        )


def test_rejected_partial_publication_does_not_advance_coverage(
    tmp_path: Path,
) -> None:
    """A failed second artifact preserves the first confirmed publication."""
    ledger = PublicationCoverageLedger(start_offset=100)

    ledger.record_published(
        kind="track",
        start=100,
        end=400,
        path=_artifact(tmp_path, "outgoing.mp3"),
    )

    with pytest.raises(CoverageError, match="missing|unsafe"):
        ledger.record_published(
            kind="unowned",
            start=400,
            end=500,
            path=tmp_path / "not-published.mp3",
        )

    assert ledger.covered_end == 400
    assert len(ledger.published) == 1

    with pytest.raises(CoverageError, match="tail"):
        ledger.verify_complete(end_offset=500)
