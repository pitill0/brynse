import pytest

from brynse.transition_evidence import (
    CrossfadeBasinEvidence,
    CrossfadeRiseEvidence,
    FadeRoutingEvidence,
    TransitionEvidenceState,
)
from brynse.transition_router import (
    TransitionKind,
    TransitionRouter,
    TransitionRoutingContext,
)


class StubTransitionRouter:
    def __init__(self, kind: TransitionKind) -> None:
        self._kind = kind

    def classify(
        self,
        *,
        context: TransitionRoutingContext,
    ) -> TransitionKind:
        _ = context
        return self._kind


def test_transition_kind_exposes_only_specialist_routing_classes() -> None:
    assert tuple(TransitionKind) == (
        TransitionKind.HARD_CUT,
        TransitionKind.FADE,
        TransitionKind.INTERSTITIAL,
        TransitionKind.AMBIGUOUS,
    )


def test_transition_routing_context_preserves_descriptive_inputs() -> None:
    evidence = TransitionEvidenceState(
        spectral_change=0.42,
    )

    context = TransitionRoutingContext(
        reference_time_seconds=120.5,
        evidence=evidence,
    )

    assert context.reference_time_seconds == 120.5
    assert context.evidence is evidence
    assert context.hypotheses == ()
    assert context.fade == FadeRoutingEvidence()


def test_transition_routing_context_rejects_negative_reference_time() -> None:
    with pytest.raises(
        ValueError,
        match="reference_time_seconds",
    ):
        TransitionRoutingContext(
            reference_time_seconds=-0.01,
            evidence=TransitionEvidenceState(),
        )


def test_transition_router_contract_classifies_context_without_boundaries() -> None:
    router: TransitionRouter = StubTransitionRouter(
        TransitionKind.AMBIGUOUS,
    )

    result = router.classify(
        context=TransitionRoutingContext(
            reference_time_seconds=42.0,
            evidence=TransitionEvidenceState(
                spectral_change=0.42,
            ),
        )
    )

    assert result is TransitionKind.AMBIGUOUS


def test_transition_kind_carries_no_temporal_boundary() -> None:
    result = TransitionKind.HARD_CUT

    assert not hasattr(result, "incoming_start_seconds")
    assert not hasattr(result, "outgoing_end_seconds")


def test_transition_routing_context_carries_no_resolved_boundary() -> None:
    context = TransitionRoutingContext(
        reference_time_seconds=42.0,
        evidence=TransitionEvidenceState(),
    )

    assert not hasattr(context, "incoming_start_seconds")
    assert not hasattr(context, "outgoing_end_seconds")



def test_transition_routing_context_preserves_fade_evidence() -> None:
    rise = CrossfadeRiseEvidence(
        time_seconds=431.15,
        jump_ratio=3.5,
        persistence_ratio=4.0,
        after_global_ratio=3.0,
    )
    basin = CrossfadeBasinEvidence(
        center_time_seconds=426.75,
        lead_seconds=0.85,
        width_seconds=0.7,
        prominence=1.5,
        quietness=1.8,
    )
    fade = FadeRoutingEvidence(
        rise=rise,
        basins=(basin,),
    )

    context = TransitionRoutingContext(
        reference_time_seconds=427.6,
        evidence=TransitionEvidenceState(),
        fade=fade,
    )

    assert context.fade is fade
    assert context.fade.rise is rise
    assert context.fade.basins == (basin,)
    assert not hasattr(context.fade, "incoming_start_seconds")
    assert not hasattr(context.fade, "outgoing_end_seconds")
    assert not hasattr(context.fade, "kind")
