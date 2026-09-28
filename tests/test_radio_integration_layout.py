from pathlib import Path

RADIO_MODULES = {
    "models.py",
    "icy.py",
    "metadata.py",
    "orchestrator.py",
    "ripping.py",
    "session.py",
    "session_output.py",
    "runner.py",
    "transient.py",
    "cli.py",
}

LEGACY_MODULES = {
    "radio_models.py",
    "icy.py",
    "metadata.py",
    "radio_orchestrator.py",
    "ripping.py",
    "session.py",
    "session_output.py",
    "runner.py",
    "transient.py",
    "cli.py",
}


def test_complete_radio_layer_lives_under_radio_integration() -> None:
    root = Path(__file__).resolve().parents[1]
    package = root / "src/brynse"
    radio = package / "integrations/radio"

    assert radio.is_dir()

    missing = [name for name in sorted(RADIO_MODULES) if not (radio / name).exists()]

    assert not missing, "radio integration modules missing:\n" + "\n".join(missing)

    remaining = [name for name in sorted(LEGACY_MODULES) if (package / name).exists()]

    assert not remaining, "legacy top-level radio modules remain:\n" + "\n".join(remaining)
