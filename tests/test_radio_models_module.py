from pathlib import Path


def test_radio_models_are_separated_from_core_models() -> None:
    root = Path(__file__).resolve().parents[1]

    core_models = (root / "src/fluxtuner_ripper/models.py").read_text(encoding="utf-8")

    radio_models = root / "src/fluxtuner_ripper/integrations/radio/models.py"

    assert radio_models.exists()

    assert not (root / "src/fluxtuner_ripper/radio_models.py").exists()

    forbidden = (
        "class ContentKind",
        "class MetadataEvent",
        "class TimedMetadataEvent",
        "class TrackCandidate",
        "class MetadataSemanticDecision",
        "class BoundaryMatch",
        "class IcyParseResult",
        "class RippingIngestResult",
    )

    violations = [name for name in forbidden if name in core_models]

    assert not violations, "radio-specific models remain in core models.py:\n" + "\n".join(
        violations
    )
