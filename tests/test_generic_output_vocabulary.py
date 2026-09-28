from pathlib import Path


def test_obsolete_track_output_vocabulary_is_removed() -> None:
    root = Path(__file__).resolve().parents[1]

    paths = (
        root / "src/brynse/output.py",
        root / "src/brynse/integrations/radio/session_output.py",
        root / "src/brynse/__init__.py",
        root / "src/brynse/integrations/radio/ripping.py",
    )

    forbidden = (
        "TrackRangePlanner",
        "EncodedTrackWriter",
        "TrackFileWriter",
        "Mp3TrackFinalizer",
        "AacTrackFinalizer",
        "TrackFinalizeError",
        "TrackFileWriteError",
        "TrackOutputService",
        "create_safe_track_output_service",
        "write_track",
    )

    violations: list[str] = []

    for path in paths:
        source = path.read_text(encoding="utf-8")

        for name in forbidden:
            if name in source:
                violations.append(f"{path.relative_to(root)}: {name}")

    assert not violations, "obsolete track-oriented output vocabulary remains:\n" + "\n".join(
        violations
    )
