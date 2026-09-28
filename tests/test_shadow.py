from __future__ import annotations

from array import array

import pytest

from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
)
from brynse.confidence import (
    BoundaryConfidenceRoute,
    ShadowBoundaryAssessment,
)
from brynse.local_discontinuity import LocalDiscontinuityEvidence
from brynse.models import DecodedPcm
from brynse.shadow import (
    ShadowBoundaryAnalysis,
    ShadowBoundaryAnalyzer,
)
from brynse.spectral import SpectralDistributionEvidence
from brynse.temporal_variability import TemporalVariabilityEvidence
from brynse.transition_assessment import (
    ShadowTransitionAssessment,
    TransitionAssessmentLabel,
)
from brynse.transition_evidence import TransitionEvidenceState


def _pcm() -> DecodedPcm:
    return DecodedPcm(
        sample_rate=8000,
        channels=1,
        sample_width_bytes=2,
        data=array("h", [0] * 160).tobytes(),
    )


def _hypothesis() -> BoundaryHypothesis:
    basin = BoundaryProposal(
        time_seconds=100.0,
        source=BoundaryProposalSource.BASIN,
    )
    structural = BoundaryProposal(
        time_seconds=100.5,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.5,
    )
    return BoundaryHypothesis(
        time_seconds=100.25,
        proposals=(basin, structural),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_depth=0.04,
            ),
            BoundaryEvidence(
                proposal=structural,
                structural_novelty=0.5,
            ),
        ),
    )


def _spectral(*, between: float = 0.3) -> SpectralDistributionEvidence:
    return SpectralDistributionEvidence(
        between=between,
        before_sequential_variability=0.1,
        after_sequential_variability=0.2,
        before_distribution_spread=0.05,
        after_distribution_spread=0.08,
    )


class _SpectralAnalyzer:
    required_preroll_seconds = 14.0
    required_postroll_seconds = 14.0

    def __init__(
        self,
        result: SpectralDistributionEvidence | None,
    ) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def analyze(self, **kwargs: object) -> SpectralDistributionEvidence | None:
        self.calls.append(kwargs)
        return self.result


class _TemporalAnalyzer:
    available = True
    required_preroll_seconds = 42.0
    required_postroll_seconds = 42.0

    def __init__(self, result: TemporalVariabilityEvidence | None) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def analyze(self, **kwargs: object) -> TemporalVariabilityEvidence | None:
        self.calls.append(kwargs)
        return self.result


class _LocalDiscontinuityAnalyzer:
    required_preroll_seconds = 2.0
    required_postroll_seconds = 2.0

    def __init__(self, result: LocalDiscontinuityEvidence | None) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def analyze(self, **kwargs: object) -> LocalDiscontinuityEvidence | None:
        self.calls.append(kwargs)
        return self.result


class _ConfidenceEvaluator:
    def __init__(
        self,
        result: ShadowBoundaryAssessment,
    ) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def assess(self, **kwargs: object) -> ShadowBoundaryAssessment:
        self.calls.append(kwargs)
        return self.result


class _TransitionEvaluator:
    def __init__(self, result: ShadowTransitionAssessment) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def assess(self, **kwargs: object) -> ShadowTransitionAssessment:
        self.calls.append(kwargs)
        return self.result


def test_shadow_boundary_analyzer_composes_spectral_and_confidence() -> None:
    hypothesis = _hypothesis()
    spectral = _spectral(between=0.42)
    spectral_analyzer = _SpectralAnalyzer(spectral)
    assessment = ShadowBoundaryAssessment(
        hypothesis=hypothesis,
        spectral_between=0.42,
        candidate_confidence=BoundaryConfidence.HIGH,
        route=BoundaryConfidenceRoute.BOTH_DISTRIBUTION,
    )
    confidence_evaluator = _ConfidenceEvaluator(assessment)

    result = ShadowBoundaryAnalyzer(
        spectral_analyzer=spectral_analyzer,
        confidence_evaluator=confidence_evaluator,
    ).analyze(
        hypothesis=hypothesis,
        pcm=_pcm(),
        absolute_start_time_seconds=90.0,
    )

    assert result.hypothesis is hypothesis
    assert result.spectral is spectral
    assert result.assessment is assessment
    assert result.transition_evidence == TransitionEvidenceState(
        spectral_change=0.42,
    )
    assert spectral_analyzer.calls == [
        {
            "pcm": _pcm(),
            "boundary_time_seconds": hypothesis.time_seconds,
            "absolute_start_time_seconds": 90.0,
        }
    ]
    assert confidence_evaluator.calls == [
        {
            "hypothesis": hypothesis,
            "spectral_between": 0.42,
        }
    ]
    assert hypothesis.confidence is BoundaryConfidence.UNRESOLVED


def test_shadow_boundary_analyzer_passes_missing_spectral_context_to_evaluator() -> None:
    hypothesis = _hypothesis()
    spectral_analyzer = _SpectralAnalyzer(None)
    assessment = ShadowBoundaryAssessment(
        hypothesis=hypothesis,
        spectral_between=None,
        candidate_confidence=BoundaryConfidence.UNRESOLVED,
    )
    confidence_evaluator = _ConfidenceEvaluator(assessment)

    result = ShadowBoundaryAnalyzer(
        spectral_analyzer=spectral_analyzer,
        confidence_evaluator=confidence_evaluator,
    ).analyze(
        hypothesis=hypothesis,
        pcm=_pcm(),
    )

    assert result.spectral is None
    assert result.transition_evidence == TransitionEvidenceState()
    assert confidence_evaluator.calls == [
        {
            "hypothesis": hypothesis,
            "spectral_between": None,
        }
    ]


def test_shadow_boundary_analysis_rejects_mismatched_assessment_hypothesis() -> None:
    first = _hypothesis()
    second = _hypothesis()
    assessment = ShadowBoundaryAssessment(
        hypothesis=second,
        spectral_between=None,
        candidate_confidence=BoundaryConfidence.UNRESOLVED,
    )

    with pytest.raises(ValueError, match="assessment hypothesis"):
        ShadowBoundaryAnalysis(
            hypothesis=first,
            spectral=None,
            assessment=assessment,
        )


def test_shadow_boundary_analyzer_composes_multihorizon_temporal_evidence() -> None:
    hypothesis = _hypothesis()
    temporal = TemporalVariabilityEvidence(
        pre_profile_pairwise=(1.0, 1.1, 1.2, 1.3, 1.4),
        post_profile_pairwise=(0.8, 0.9, 1.0, 1.1, 1.2),
        variability_ratio_8s=0.8,
        variability_ratio_16s=0.9,
        variability_ratio_24s=1.0,
        variability_ratio_40s=1.1,
    )
    temporal_analyzer = _TemporalAnalyzer(temporal)
    temporal_pcm = _pcm()

    result = ShadowBoundaryAnalyzer(
        spectral_analyzer=_SpectralAnalyzer(None),
        temporal_variability_analyzer=temporal_analyzer,
        confidence_evaluator=_ConfidenceEvaluator(
            ShadowBoundaryAssessment(
                hypothesis=hypothesis,
                spectral_between=None,
                candidate_confidence=BoundaryConfidence.UNRESOLVED,
            )
        ),
    ).analyze(
        hypothesis=hypothesis,
        pcm=_pcm(),
        temporal_pcm=temporal_pcm,
        absolute_start_time_seconds=90.0,
        measure_spectral=False,
        measure_temporal=True,
    )

    assert result.temporal_variability is temporal
    assert result.transition_evidence == TransitionEvidenceState(
        variability_ratio_8s=0.8,
        variability_ratio_16s=0.9,
        variability_ratio_24s=1.0,
        variability_ratio_40s=1.1,
    )
    assert temporal_analyzer.calls == [
        {
            "pcm": temporal_pcm,
            "boundary_time_seconds": hypothesis.time_seconds,
            "absolute_start_time_seconds": 90.0,
        }
    ]


def test_shadow_boundary_analyzer_composes_local_discontinuity_evidence() -> None:
    hypothesis = _hypothesis()
    local = LocalDiscontinuityEvidence(
        change_250ms=0.21,
        change_500ms=0.43,
        change_1s=0.65,
        change_2s=0.87,
    )
    local_analyzer = _LocalDiscontinuityAnalyzer(local)
    local_pcm = _pcm()

    result = ShadowBoundaryAnalyzer(
        spectral_analyzer=_SpectralAnalyzer(None),
        local_discontinuity_analyzer=local_analyzer,
        confidence_evaluator=_ConfidenceEvaluator(
            ShadowBoundaryAssessment(
                hypothesis=hypothesis,
                spectral_between=None,
                candidate_confidence=BoundaryConfidence.UNRESOLVED,
            )
        ),
    ).analyze(
        hypothesis=hypothesis,
        pcm=_pcm(),
        local_discontinuity_pcm=local_pcm,
        absolute_start_time_seconds=90.0,
        measure_spectral=False,
        measure_local_discontinuity=True,
    )

    assert result.local_discontinuity is local
    assert result.transition_evidence == TransitionEvidenceState(
        local_discontinuity_250ms=0.21,
        local_discontinuity_500ms=0.43,
        local_discontinuity_1s=0.65,
        local_discontinuity_2s=0.87,
    )
    assert local_analyzer.calls == [
        {
            "pcm": local_pcm,
            "boundary_time_seconds": hypothesis.time_seconds,
            "absolute_start_time_seconds": 90.0,
        }
    ]


def test_shadow_boundary_analyzer_requires_local_pcm_when_enabled() -> None:
    hypothesis = _hypothesis()

    with pytest.raises(ValueError, match="local_discontinuity_pcm"):
        ShadowBoundaryAnalyzer(
            spectral_analyzer=_SpectralAnalyzer(None),
            local_discontinuity_analyzer=_LocalDiscontinuityAnalyzer(None),
            confidence_evaluator=_ConfidenceEvaluator(
                ShadowBoundaryAssessment(
                    hypothesis=hypothesis,
                    spectral_between=None,
                    candidate_confidence=BoundaryConfidence.UNRESOLVED,
                )
            ),
        ).analyze(
            hypothesis=hypothesis,
            pcm=_pcm(),
            measure_spectral=False,
            measure_local_discontinuity=True,
        )


def test_shadow_boundary_analyzer_requires_temporal_pcm_when_enabled() -> None:
    hypothesis = _hypothesis()

    with pytest.raises(ValueError, match="temporal_pcm"):
        ShadowBoundaryAnalyzer(
            spectral_analyzer=_SpectralAnalyzer(None),
            temporal_variability_analyzer=_TemporalAnalyzer(None),
            confidence_evaluator=_ConfidenceEvaluator(
                ShadowBoundaryAssessment(
                    hypothesis=hypothesis,
                    spectral_between=None,
                    candidate_confidence=BoundaryConfidence.UNRESOLVED,
                )
            ),
        ).analyze(
            hypothesis=hypothesis,
            pcm=_pcm(),
            measure_spectral=False,
            measure_temporal=True,
        )


def test_shadow_boundary_analyzer_reports_temporal_required_context() -> None:
    analyzer = ShadowBoundaryAnalyzer(
        spectral_analyzer=_SpectralAnalyzer(None),
        temporal_variability_analyzer=_TemporalAnalyzer(None),
    )

    assert analyzer.required_preroll_seconds == 42.0
    assert analyzer.required_postroll_seconds == 42.0


def test_shadow_boundary_analysis_exposes_flat_evidence_record() -> None:
    hypothesis = _hypothesis()
    analysis = ShadowBoundaryAnalysis(
        hypothesis=hypothesis,
        spectral=_spectral(between=0.55),
        assessment=ShadowBoundaryAssessment(
            hypothesis=hypothesis,
            spectral_between=0.55,
            candidate_confidence=BoundaryConfidence.UNRESOLVED,
        ),
        transition_evidence=TransitionEvidenceState(
            spectral_change=0.55,
            variability_ratio_8s=0.8,
            variability_ratio_16s=0.9,
            variability_ratio_24s=1.0,
            variability_ratio_40s=1.1,
        ),
        transition_assessment=ShadowTransitionAssessment(
            evidence=TransitionEvidenceState(
                spectral_change=0.55,
                variability_ratio_8s=0.8,
                variability_ratio_16s=0.9,
                variability_ratio_24s=1.0,
                variability_ratio_40s=1.1,
            ),
        ),
    )

    assert analysis.evidence_record() == {
        "time_seconds": hypothesis.time_seconds,
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


def test_shadow_boundary_analyzer_composes_transition_assessment() -> None:
    hypothesis = _hypothesis()
    transition_evidence = TransitionEvidenceState(spectral_change=0.42)
    transition_assessment = ShadowTransitionAssessment(
        evidence=transition_evidence,
        label=TransitionAssessmentLabel.REAL_LIKE,
    )
    transition_evaluator = _TransitionEvaluator(transition_assessment)

    result = ShadowBoundaryAnalyzer(
        spectral_analyzer=_SpectralAnalyzer(_spectral(between=0.42)),
        confidence_evaluator=_ConfidenceEvaluator(
            ShadowBoundaryAssessment(
                hypothesis=hypothesis,
                spectral_between=0.42,
                candidate_confidence=BoundaryConfidence.UNRESOLVED,
            )
        ),
        transition_evaluator=transition_evaluator,
    ).analyze(
        hypothesis=hypothesis,
        pcm=_pcm(),
    )

    assert result.transition_evidence == TransitionEvidenceState(spectral_change=0.42)
    assert result.transition_assessment is transition_assessment
    assert transition_evaluator.calls == [
        {
            "evidence": TransitionEvidenceState(spectral_change=0.42),
        }
    ]


def test_shadow_boundary_analysis_rejects_mismatched_transition_assessment_evidence() -> None:
    hypothesis = _hypothesis()

    with pytest.raises(ValueError, match="transition assessment evidence"):
        ShadowBoundaryAnalysis(
            hypothesis=hypothesis,
            spectral=None,
            assessment=ShadowBoundaryAssessment(
                hypothesis=hypothesis,
                spectral_between=None,
                candidate_confidence=BoundaryConfidence.UNRESOLVED,
            ),
            transition_evidence=TransitionEvidenceState(spectral_change=0.2),
            transition_assessment=ShadowTransitionAssessment(
                evidence=TransitionEvidenceState(spectral_change=0.3),
            ),
        )
