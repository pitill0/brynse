from fluxtuner_ripper.transition_assessment import (
    ShadowTransitionAssessment,
    ShadowTransitionEvaluator,
    TransitionAssessmentLabel,
)
from fluxtuner_ripper.transition_evidence import TransitionEvidenceState


def test_shadow_transition_assessment_defaults_to_unresolved() -> None:
    evidence = TransitionEvidenceState(spectral_change=0.42)

    assessment = ShadowTransitionAssessment(evidence=evidence)

    assert assessment.evidence is evidence
    assert assessment.label is TransitionAssessmentLabel.UNRESOLVED


def test_shadow_transition_evaluator_does_not_classify_evidence_yet() -> None:
    evidence = TransitionEvidenceState(
        spectral_change=0.55,
        variability_ratio_8s=0.8,
        variability_ratio_16s=0.9,
        variability_ratio_24s=1.0,
        variability_ratio_40s=1.1,
    )

    assessment = ShadowTransitionEvaluator().assess(evidence=evidence)

    assert assessment.evidence is evidence
    assert assessment.label is TransitionAssessmentLabel.UNRESOLVED
