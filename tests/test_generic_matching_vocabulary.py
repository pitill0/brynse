import re
from pathlib import Path


def test_matching_module_has_no_radio_specific_track_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "src/fluxtuner_ripper/matching.py"

    source = path.read_text(encoding="utf-8")

    forbidden_patterns = (
        r"\bTrackCandidate\b",
        r"\bBoundaryMatch\b",
        r"\bas_boundary_candidate\b",
        r"\bdef match\(",
        r"\btrack\s*=",
    )

    violations = [pattern for pattern in forbidden_patterns if re.search(pattern, source)]

    assert not violations, "radio-specific matching vocabulary remains:\n" + "\n".join(violations)
