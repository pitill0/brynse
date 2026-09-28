import re
from pathlib import Path


def test_orchestrator_core_has_no_radio_track_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "src/brynse/orchestrator.py"

    source = path.read_text(encoding="utf-8")

    forbidden_patterns = (
        r"\bTrackCandidate\b",
        r"\bBoundaryMatch\b",
        r"\bBoundaryResolution\b",
        r"\bBoundaryResolver\b",
        r"\bRippingOrchestrator\b",
        r"\bresolve_boundary\b",
    )

    violations = [pattern for pattern in forbidden_patterns if re.search(pattern, source)]

    assert not violations, "radio-specific orchestration vocabulary remains:\n" + "\n".join(
        violations
    )
