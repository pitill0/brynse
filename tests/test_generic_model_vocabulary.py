from pathlib import Path


def test_track_range_compatibility_vocabulary_is_removed() -> None:
    root = Path(__file__).resolve().parents[1]

    paths = (
        root / "src/brynse/models.py",
        root / "src/brynse/output.py",
        root / "src/brynse/__init__.py",
        root / "src/brynse/integrations/radio/ripping.py",
    )

    forbidden = (
        "TrackByteRange",
        "TrackWritePlan",
    )

    violations: list[str] = []

    for path in paths:
        source = path.read_text(encoding="utf-8")

        for name in forbidden:
            if name in source:
                violations.append(f"{path.relative_to(root)}: {name}")

    assert not violations, "obsolete radio/output compatibility vocabulary remains:\n" + "\n".join(
        violations
    )
