from brynse.transition_evidence import TransitionEvidenceState
from brynse.transition_router import TransitionKind, TransitionRouter


class StubTransitionRouter:
    def __init__(self, kind: TransitionKind) -> None:
        self._kind = kind

    def classify(
        self,
        *,
        evidence: TransitionEvidenceState,
    ) -> TransitionKind:
        _ = evidence
        return self._kind


def test_transition_kind_exposes_only_specialist_routing_classes() -> None:
    assert tuple(TransitionKind) == (
        TransitionKind.HARD_CUT,
        TransitionKind.FADE,
        TransitionKind.INTERSTITIAL,
        TransitionKind.AMBIGUOUS,
    )


def test_transition_router_contract_classifies_without_boundaries() -> None:
    router: TransitionRouter = StubTransitionRouter(
        TransitionKind.AMBIGUOUS,
    )

    result = router.classify(
        evidence=TransitionEvidenceState(
            spectral_change=0.42,
        )
    )

    assert result is TransitionKind.AMBIGUOUS


def test_transition_kind_carries_no_temporal_boundary() -> None:
    result = TransitionKind.HARD_CUT

    assert not hasattr(result, "incoming_start_seconds")
    assert not hasattr(result, "outgoing_end_seconds")
