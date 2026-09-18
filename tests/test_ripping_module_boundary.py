from pathlib import Path


def test_production_modules_do_not_use_ripping_as_generic_facade() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src/fluxtuner_ripper"

    allowed = {
        "__init__.py",
        "runner.py",
        "session.py",
        "session_output.py",
    }

    violations: list[str] = []

    for path in sorted(src.glob("*.py")):
        if path.name in allowed or path.name == "ripping.py":
            continue

        source = path.read_text(encoding="utf-8")

        if "from fluxtuner_ripper.integrations.radio.ripping import" in source:
            violations.append(path.name)

    assert not violations, "production modules still use ripping.py as a facade:\n" + "\n".join(
        violations
    )
