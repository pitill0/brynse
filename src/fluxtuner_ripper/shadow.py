"""Composition layer for non-operative autonomous boundary shadow analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from fluxtuner_ripper.boundaries import BoundaryHypothesis
from fluxtuner_ripper.confidence import (
    ShadowBoundaryAssessment,
    ShadowBoundaryConfidenceEvaluator,
)
from fluxtuner_ripper.local_discontinuity import (
    LocalDiscontinuityAnalyzer,
    LocalDiscontinuityEvidence,
)
from fluxtuner_ripper.models import DecodedPcm
from fluxtuner_ripper.spectral import (
    FfmpegSpectralDistributionAnalyzer,
    SpectralDistributionEvidence,
)
from fluxtuner_ripper.temporal_variability import (
    TemporalVariabilityAnalyzer,
    TemporalVariabilityEvidence,
)
from fluxtuner_ripper.transition_assessment import (
    ShadowTransitionAssessment,
    ShadowTransitionEvaluator,
)
from fluxtuner_ripper.transition_evidence import (
    TransitionEvidenceExtractor,
    TransitionEvidenceState,
)


class SpectralDistributionAnalyzer(Protocol):
    """Protocol for extracting spectral evidence around one hypothesis."""

    @property
    def required_preroll_seconds(self) -> float: ...

    @property
    def required_postroll_seconds(self) -> float: ...

    def analyze(
        self,
        *,
        pcm: DecodedPcm,
        boundary_time_seconds: float,
        absolute_start_time_seconds: float = 0.0,
    ) -> SpectralDistributionEvidence | None: ...


class MultiHorizonTemporalVariabilityAnalyzer(Protocol):
    """Protocol for exact B7.3e temporal-variability evidence extraction."""

    @property
    def available(self) -> bool: ...

    @property
    def required_preroll_seconds(self) -> float: ...

    @property
    def required_postroll_seconds(self) -> float: ...

    def analyze(
        self,
        *,
        pcm: DecodedPcm,
        boundary_time_seconds: float,
        absolute_start_time_seconds: float = 0.0,
    ) -> TemporalVariabilityEvidence | None: ...


class ShortHorizonLocalDiscontinuityAnalyzer(Protocol):
    """Protocol for validated local PRE/POST discontinuity evidence."""

    @property
    def required_preroll_seconds(self) -> float: ...

    @property
    def required_postroll_seconds(self) -> float: ...

    def analyze(
        self,
        *,
        pcm: DecodedPcm,
        boundary_time_seconds: float,
        absolute_start_time_seconds: float = 0.0,
    ) -> LocalDiscontinuityEvidence | None: ...


class ShadowConfidenceEvaluator(Protocol):
    """Protocol for turning descriptive evidence into a shadow assessment."""

    def assess(
        self,
        *,
        hypothesis: BoundaryHypothesis,
        spectral_between: float | None,
    ) -> ShadowBoundaryAssessment: ...


class TransitionEvaluator(Protocol):
    """Protocol for interpreting composed transition evidence in shadow mode."""

    def assess(
        self,
        *,
        evidence: TransitionEvidenceState,
    ) -> ShadowTransitionAssessment: ...


@dataclass(frozen=True)
class ShadowBoundaryAnalysis:
    """Complete non-operative analysis result for one boundary hypothesis."""

    hypothesis: BoundaryHypothesis
    spectral: SpectralDistributionEvidence | None
    assessment: ShadowBoundaryAssessment
    temporal_variability: TemporalVariabilityEvidence | None = None
    local_discontinuity: LocalDiscontinuityEvidence | None = None
    transition_evidence: TransitionEvidenceState = field(default_factory=TransitionEvidenceState)
    transition_assessment: ShadowTransitionAssessment = field(
        default_factory=lambda: ShadowTransitionAssessment(evidence=TransitionEvidenceState())
    )

    def __post_init__(self) -> None:
        if self.assessment.hypothesis is not self.hypothesis:
            raise ValueError("assessment hypothesis must match analysis hypothesis")
        if self.transition_assessment.evidence != self.transition_evidence:
            raise ValueError("transition assessment evidence must match transition evidence")

    def evidence_record(self) -> dict[str, float | None]:
        """Return flat shadow evidence suitable for JSONL/auditor output."""
        return {
            "time_seconds": self.hypothesis.time_seconds,
            **self.transition_evidence.as_dict(),
        }


class ShadowBoundaryAnalyzer:
    """Compose spectral evidence extraction and shadow confidence evaluation."""

    def __init__(
        self,
        *,
        spectral_analyzer: SpectralDistributionAnalyzer | None = None,
        confidence_evaluator: ShadowConfidenceEvaluator | None = None,
        temporal_variability_analyzer: MultiHorizonTemporalVariabilityAnalyzer | None = None,
        local_discontinuity_analyzer: ShortHorizonLocalDiscontinuityAnalyzer | None = None,
        transition_evidence_extractor: TransitionEvidenceExtractor | None = None,
        transition_evaluator: TransitionEvaluator | None = None,
    ) -> None:
        self._spectral_analyzer = (
            spectral_analyzer
            if spectral_analyzer is not None
            else FfmpegSpectralDistributionAnalyzer()
        )
        self._confidence_evaluator = (
            confidence_evaluator
            if confidence_evaluator is not None
            else ShadowBoundaryConfidenceEvaluator()
        )
        self._temporal_variability_analyzer = (
            temporal_variability_analyzer
            if temporal_variability_analyzer is not None
            else TemporalVariabilityAnalyzer()
        )
        self._local_discontinuity_analyzer = (
            local_discontinuity_analyzer
            if local_discontinuity_analyzer is not None
            else LocalDiscontinuityAnalyzer()
        )
        self._transition_evidence_extractor = (
            transition_evidence_extractor
            if transition_evidence_extractor is not None
            else TransitionEvidenceExtractor()
        )
        self._transition_evaluator = (
            transition_evaluator
            if transition_evaluator is not None
            else ShadowTransitionEvaluator()
        )

    @property
    def supports_temporal_variability(self) -> bool:
        return self._temporal_variability_analyzer.available

    @property
    def required_preroll_seconds(self) -> float:
        if not self.supports_temporal_variability:
            return max(
                self._spectral_analyzer.required_preroll_seconds,
                self._local_discontinuity_analyzer.required_preroll_seconds,
            )
        return max(
            self._spectral_analyzer.required_preroll_seconds,
            self._temporal_variability_analyzer.required_preroll_seconds,
            self._local_discontinuity_analyzer.required_preroll_seconds,
        )

    @property
    def required_postroll_seconds(self) -> float:
        if not self.supports_temporal_variability:
            return max(
                self._spectral_analyzer.required_postroll_seconds,
                self._local_discontinuity_analyzer.required_postroll_seconds,
            )
        return max(
            self._spectral_analyzer.required_postroll_seconds,
            self._temporal_variability_analyzer.required_postroll_seconds,
            self._local_discontinuity_analyzer.required_postroll_seconds,
        )

    def analyze(
        self,
        *,
        hypothesis: BoundaryHypothesis,
        pcm: DecodedPcm,
        temporal_pcm: DecodedPcm | None = None,
        local_discontinuity_pcm: DecodedPcm | None = None,
        absolute_start_time_seconds: float = 0.0,
        measure_spectral: bool = True,
        measure_temporal: bool = False,
        measure_local_discontinuity: bool = False,
    ) -> ShadowBoundaryAnalysis:
        """Return shadow evidence and confidence without mutating the hypothesis."""
        spectral = (
            self._spectral_analyzer.analyze(
                pcm=pcm,
                boundary_time_seconds=hypothesis.time_seconds,
                absolute_start_time_seconds=absolute_start_time_seconds,
            )
            if measure_spectral
            else None
        )
        if measure_temporal and temporal_pcm is None:
            raise ValueError("temporal_pcm is required when measure_temporal is enabled")

        temporal_variability = (
            self._temporal_variability_analyzer.analyze(
                pcm=temporal_pcm,
                boundary_time_seconds=hypothesis.time_seconds,
                absolute_start_time_seconds=absolute_start_time_seconds,
            )
            if measure_temporal and temporal_pcm is not None and self.supports_temporal_variability
            else None
        )
        if measure_local_discontinuity and local_discontinuity_pcm is None:
            raise ValueError(
                "local_discontinuity_pcm is required when measure_local_discontinuity is enabled"
            )

        local_discontinuity = (
            self._local_discontinuity_analyzer.analyze(
                pcm=local_discontinuity_pcm,
                boundary_time_seconds=hypothesis.time_seconds,
                absolute_start_time_seconds=absolute_start_time_seconds,
            )
            if measure_local_discontinuity and local_discontinuity_pcm is not None
            else None
        )

        transition_evidence = self._transition_evidence_extractor.extract(
            spectral=spectral,
            temporal=temporal_variability,
            local_discontinuity=local_discontinuity,
        )
        transition_assessment = self._transition_evaluator.assess(
            evidence=transition_evidence,
        )
        assessment = self._confidence_evaluator.assess(
            hypothesis=hypothesis,
            spectral_between=spectral.between if spectral is not None else None,
        )
        return ShadowBoundaryAnalysis(
            hypothesis=hypothesis,
            spectral=spectral,
            assessment=assessment,
            temporal_variability=temporal_variability,
            local_discontinuity=local_discontinuity,
            transition_evidence=transition_evidence,
            transition_assessment=transition_assessment,
        )
