import ast
from pathlib import Path


def test_package_root_does_not_export_radio_integration() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "src/fluxtuner_ripper/__init__.py"

    tree = ast.parse(path.read_text(encoding="utf-8"))

    violations: list[str] = []

    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("fluxtuner_ripper.integrations.radio")
        ):
            violations.append(f"line {node.lineno}: {node.module}")

    assert not violations, "package root exports radio integration symbols:\n" + "\n".join(
        violations
    )
