import ast
from pathlib import Path


def test_ripping_module_only_imports_runtime_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "src/fluxtuner_ripper/integrations/radio/ripping.py"

    tree = ast.parse(path.read_text(encoding="utf-8"))

    forbidden_modules = {
        "fluxtuner_ripper.acoustic",
        "fluxtuner_ripper.matching",
        "fluxtuner_ripper.integrations.radio.metadata",
        "fluxtuner_ripper.output",
    }

    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
            violations.append(f"line {node.lineno}: {node.module}")

    assert not violations, "ripping.py still contains historical facade imports:\n" + "\n".join(
        violations
    )
