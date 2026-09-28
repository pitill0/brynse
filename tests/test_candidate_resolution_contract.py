from dataclasses import fields

from brynse.orchestrator import CandidateResolution


def test_candidate_resolution_contract_is_algorithm_agnostic() -> None:
    field_names = {field.name for field in fields(CandidateResolution)}

    assert field_names == {
        "candidate",
        "temporal",
        "split",
    }
