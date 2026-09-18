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

CORE_MODULES = {
    "acoustic.py",
    "autonomous.py",
    "basin.py",
    "boundaries.py",
    "buffer.py",
    "confidence.py",
    "external_boundaries.py",
    "frames.py",
    "generic_cli.py",
    "generic_output.py",
    "generic_runner.py",
    "hybrid.py",
    "ingest.py",
    "local_discontinuity.py",
    "machine.py",
    "matching.py",
    "models.py",
    "mp3_refinement.py",
    "orchestrator.py",
    "output.py",
    "providers.py",
    "shadow.py",
    "source.py",
    "spectral.py",
    "spooling_ingest.py",
    "streaming_fifo.py",
    "streaming_runner.py",
    "streaming_runtime.py",
    "streaming_sink.py",
    "streaming_sources.py",
    "streaming_spool.py",
    "structural.py",
    "temporal_variability.py",
    "transition_assessment.py",
    "transition_evidence.py",
}


def _src_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src/fluxtuner_ripper"


def test_core_models_do_not_define_radio_models() -> None:
    source = (_src_root() / "models.py").read_text(encoding="utf-8")

    violations = [name for name in RADIO_MODEL_NAMES if f"class {name}" in source]

    assert not violations, "radio-domain models found in core models.py:\n" + "\n".join(
        sorted(violations)
    )


def test_core_models_do_not_depend_on_radio_models() -> None:
    source = (_src_root() / "models.py").read_text(encoding="utf-8")

    assert "fluxtuner_ripper.integrations.radio.models" not in source


def test_core_modules_do_not_import_radio_models() -> None:
    src = _src_root()
    violations: list[str] = []

    for name in sorted(CORE_MODULES):
        path = src / name

        if not path.exists():
            continue

        source = path.read_text(encoding="utf-8")

        if "from fluxtuner_ripper.integrations.radio.models import" in source:
            violations.append(name)

    assert not violations, "core modules depend on radio_models:\n" + "\n".join(violations)


def test_ripping_module_is_not_a_generic_facade() -> None:
    source = (_src_root() / "integrations/radio/ripping.py").read_text(encoding="utf-8")

    forbidden = {
        "fluxtuner_ripper.acoustic",
        "fluxtuner_ripper.matching",
        "fluxtuner_ripper.integrations.radio.metadata",
        "fluxtuner_ripper.output",
    }

    violations = [module for module in sorted(forbidden) if module in source]

    assert not violations, "ripping.py regained facade dependencies:\n" + "\n".join(violations)


def test_package_root_does_not_export_radio_models() -> None:
    source = (_src_root() / "__init__.py").read_text(encoding="utf-8")

    assert "from fluxtuner_ripper.integrations.radio.models import" not in source

    violations = [name for name in RADIO_MODEL_NAMES if f'"{name}"' in source]

    assert not violations, "package root exports radio-domain models:\n" + "\n".join(
        sorted(violations)
    )
