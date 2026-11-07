"""Regression for descriptive acoustic regions without exclusion ownership."""

from types import SimpleNamespace

from brynse.models import TemporalSplitDecision, TemporalSplitKind
from brynse.transition_router import (
    InterstitialRoutingAssessment,
    TransitionSemanticRoles,
)
from brynse.transition_specialists import (
    InterstitialSpecialist,
    TransitionSpecialistInput,
)


def test_region_only_island_does_not_discard_unowned_audio():
    """A bounded region is not proof its entire interior is disposable."""
    reference = 1316.1795918355076

    left = SimpleNamespace(
        time_seconds=1312.2230612233016,
        basin_geometry=SimpleNamespace(
            start_seconds=1311.6730612233016,
            minimum_seconds=1312.2230612233016,
            recovery_seconds=1312.6730612233016,
        ),
    )
    right = SimpleNamespace(
        time_seconds=1328.0980612233016,
        basin_geometry=SimpleNamespace(
            start_seconds=1327.3730612233016,
            minimum_seconds=1327.5230612233015,
            recovery_seconds=1327.6730612233016,
        ),
    )

    interior = SimpleNamespace(time_seconds=1315.1730612233016)
    region = SimpleNamespace(
        left=left,
        right=right,
        interior_hypotheses=(interior,),
    )
    island = SimpleNamespace(
        observed_region=region,
        incoming_handoff=right,
        terminal_decay=None,
        source_reset=None,
    )

    class IncomingPipeline:
        analyze_hypotheses = None

        def refine(self, **kwargs):
            return TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=1312.248,
                incoming_start_seconds=1327.6730612233016,
            )

    observation = TransitionSpecialistInput(
        routing=SimpleNamespace(
            reference_time_seconds=reference,
            hypotheses=(left, interior, right),
            acoustic_islands=(island,),
        ),
        selected=SimpleNamespace(time_seconds=1312.248),
        profile=object(),
        window=object(),
        pcm8=object(),
        pcm16=object(),
        interstitial_assessment=InterstitialRoutingAssessment(
            supported_handoffs=(left, right),
        ),
        semantic_roles=TransitionSemanticRoles(
            outgoing_handoff=left,
            incoming_handoff=right,
            acoustic_island=island,
            inserted_region=region,
        ),
        bounded_insert_ownership=None,
    )

    result = InterstitialSpecialist(
        incoming_pipeline=IncomingPipeline(),
    ).resolve(observation=observation)

    assert result.temporal == TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=reference,
        incoming_start_seconds=reference,
    ), (
        "UNOWNED_REGION_INTRO_LOSS: no independent evidence "
        "authorizes discarding audio after the metadata reference"
    )
