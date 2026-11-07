"""Safety contract for unsupported interstitial exclusions."""

from types import SimpleNamespace

from brynse.models import TemporalSplitDecision, TemporalSplitKind
from brynse.transition_specialists import (
    InterstitialSpecialist,
    TransitionSpecialistInput,
)


def test_interstitial_without_region_preserves_provisional_exclusion():
    """Without an observed region, preserve the provisional fallback.

    This is distinct from an unowned descriptive region, which
    must not authorize irreversible audio removal.
    """

    class IncomingPipeline:
        def refine(self, **kwargs):
            # Plausible provisional exclusion from a refiner,
            # but with no independent handoff evidence.
            return TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=100.0,
                incoming_start_seconds=112.0,
            )

        analyze_hypotheses = None

    observation = TransitionSpecialistInput(
        routing=SimpleNamespace(
            reference_time_seconds=102.0,
            hypotheses=(),
            acoustic_islands=(),
        ),
        selected=SimpleNamespace(time_seconds=102.0),
        profile=object(),
        window=object(),
        pcm8=object(),
        pcm16=object(),
    )

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(observation=observation)

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=112.0,
    )
