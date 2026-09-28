from __future__ import annotations

import pytest

from brynse.local_discontinuity import LocalDiscontinuityEvidence
from brynse.spectral import SpectralDistributionEvidence
from brynse.temporal_variability import TemporalVariabilityEvidence
from brynse.transition_evidence import (
    TransitionEvidenceExtractor,
    TransitionEvidenceState,
)


def _spectral(*, between: float = 0.42) -> SpectralDistributionEvidence:
    return SpectralDistributionEvidence(
        between=between,
        before_sequential_variability=0.1,
        after_sequential_variability=0.2,
        before_distribution_spread=0.05,
        after_distribution_spread=0.08,
    )


@pytest.mark.parametrize(
    "field",
    [
        "spectral_change",
        "variability_ratio_8s",
        "variability_ratio_16s",
        "variability_ratio_24s",
        "variability_ratio_40s",
        "local_discontinuity_250ms",
        "local_discontinuity_500ms",
        "local_discontinuity_1s",
        "local_discontinuity_2s",
    ],
)
def test_transition_evidence_state_rejects_negative_values(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        TransitionEvidenceState(**{field: -0.01})


def test_transition_evidence_extractor_maps_spectral_change_only() -> None:
    state = TransitionEvidenceExtractor().extract(spectral=_spectral(between=0.42))

    assert state == TransitionEvidenceState(spectral_change=0.42)
    assert state.has_temporal_variability is False


def test_transition_evidence_extractor_preserves_multihorizon_variability() -> None:
    state = TransitionEvidenceExtractor().extract(
        spectral=_spectral(between=0.33),
        variability_ratio_8s=0.8,
        variability_ratio_16s=1.1,
        variability_ratio_24s=1.4,
        variability_ratio_40s=1.7,
    )

    assert state.spectral_change == 0.33
    assert state.variability_ratio_8s == 0.8
    assert state.variability_ratio_16s == 1.1
    assert state.variability_ratio_24s == 1.4
    assert state.variability_ratio_40s == 1.7
    assert state.has_temporal_variability is True


def test_transition_evidence_extractor_handles_missing_spectral_context() -> None:
    state = TransitionEvidenceExtractor().extract(spectral=None)

    assert state == TransitionEvidenceState()


def test_transition_evidence_extractor_maps_local_discontinuity() -> None:
    local = LocalDiscontinuityEvidence(
        change_250ms=0.21,
        change_500ms=0.43,
        change_1s=0.65,
        change_2s=0.87,
    )

    state = TransitionEvidenceExtractor().extract(
        spectral=_spectral(between=0.55),
        local_discontinuity=local,
    )

    assert state == TransitionEvidenceState(
        spectral_change=0.55,
        local_discontinuity_250ms=0.21,
        local_discontinuity_500ms=0.43,
        local_discontinuity_1s=0.65,
        local_discontinuity_2s=0.87,
    )


def test_transition_evidence_extractor_maps_temporal_variability() -> None:
    temporal = TemporalVariabilityEvidence(
        pre_profile_pairwise=(1.0, 1.1, 1.2, 1.3, 1.4),
        post_profile_pairwise=(0.8, 0.9, 1.0, 1.1, 1.2),
        variability_ratio_8s=0.8,
        variability_ratio_16s=0.9,
        variability_ratio_24s=1.0,
        variability_ratio_40s=1.1,
    )

    state = TransitionEvidenceExtractor().extract(
        spectral=_spectral(between=0.55),
        temporal=temporal,
    )

    assert state == TransitionEvidenceState(
        spectral_change=0.55,
        variability_ratio_8s=0.8,
        variability_ratio_16s=0.9,
        variability_ratio_24s=1.0,
        variability_ratio_40s=1.1,
    )


def test_transition_evidence_state_exposes_flat_serializable_mapping() -> None:
    state = TransitionEvidenceState(
        spectral_change=0.55,
        variability_ratio_8s=0.8,
        variability_ratio_16s=0.9,
        variability_ratio_24s=1.0,
        variability_ratio_40s=1.1,
    )

    assert state.as_dict() == {
        "spectral_change": 0.55,
        "variability_ratio_8s": 0.8,
        "variability_ratio_16s": 0.9,
        "variability_ratio_24s": 1.0,
        "variability_ratio_40s": 1.1,
        "local_discontinuity_250ms": None,
        "local_discontinuity_500ms": None,
        "local_discontinuity_1s": None,
        "local_discontinuity_2s": None,
    }


def test_transition_evidence_state_exposes_missing_values_as_none() -> None:
    assert TransitionEvidenceState().as_dict() == {
        "spectral_change": None,
        "variability_ratio_8s": None,
        "variability_ratio_16s": None,
        "variability_ratio_24s": None,
        "variability_ratio_40s": None,
        "local_discontinuity_250ms": None,
        "local_discontinuity_500ms": None,
        "local_discontinuity_1s": None,
        "local_discontinuity_2s": None,
    }
