from pathlib import Path

RADIO_MODEL_NAMES = {
    "BoundaryMatch",
    "ContentKind",
    "IcyParseResult",
    "MetadataEvent",
    "MetadataSemanticDecision",
    "RippingIngestResult",
    "TimedMetadataEvent",
    "TrackCandidate",
}


def test_package_root_does_not_export_radio_models() -> None:
    root = Path(__file__).resolve().parents[1]

    package_init = (root / "src/fluxtuner_ripper/__init__.py").read_text(encoding="utf-8")

    assert "from fluxtuner_ripper.radio_models import" not in package_init

    violations = [name for name in RADIO_MODEL_NAMES if f'"{name}"' in package_init]

    assert not violations, "radio-domain models exported from package root:\n" + "\n".join(
        sorted(violations)
    )
