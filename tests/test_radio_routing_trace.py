import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
)
from brynse.fade_specialist import OutgoingFadeSelection
from brynse.integrations.radio import cli
from brynse.models import TemporalSplitDecision, TemporalSplitKind
from brynse.orchestrator import TransitionRoutingObservation
from brynse.transition_evidence import (
    CrossfadeRiseEvidence,
    FadeGeometryEvidence,
    FadeRoutingEvidence,
    IncomingFadeEvidence,
    OutgoingFadeEvidence,
    TransitionEvidenceState,
)
from brynse.transition_router import (
    TransitionKind,
    TransitionRoutingContext,
)


def _observation() -> TransitionRoutingObservation:
    return TransitionRoutingObservation(
        context=TransitionRoutingContext(
            reference_time_seconds=100.0,
            evidence=TransitionEvidenceState(
                local_discontinuity_250ms=0.21,
                local_discontinuity_500ms=0.43,
                local_discontinuity_1s=0.65,
                local_discontinuity_2s=0.87,
            ),
            hypotheses=(
                hypothesis := BoundaryHypothesis(
                    time_seconds=101.1,
                    proposals=(
                        basin := BoundaryProposal(
                            time_seconds=101.0,
                            source=BoundaryProposalSource.BASIN,
                            strength=0.82,
                        ),
                        BoundaryProposal(
                            time_seconds=101.2,
                            source=BoundaryProposalSource.STRUCTURAL,
                            strength=0.91,
                        ),
                    ),
                    evidence=(
                        BoundaryEvidence(
                            proposal=basin,
                            basin_depth=0.24,
                            basin_start_seconds=100.7,
                            basin_recovery_seconds=101.6,
                            local_change=0.51,
                            persistent_change=0.63,
                            structural_novelty=0.74,
                            metadata_support=False,
                        ),
                    ),
                    confidence=BoundaryConfidence.HIGH,
                ),
            ),
            fade=FadeRoutingEvidence(
                outgoing_fades=(
                    outgoing := OutgoingFadeEvidence(
                        start_time_seconds=99.0,
                        floor_time_seconds=100.0,
                        rise_time_seconds=101.0,
                        start_rms=1200.0,
                        floor_rms=120.0,
                        descent_ratio=10.0,
                        descent_consistency=0.9,
                    ),
                ),
                incoming_fades=(
                    incoming := IncomingFadeEvidence(
                        start_time_seconds=100.4,
                        established_time_seconds=101.0,
                        start_rms=120.0,
                        established_rms=1200.0,
                        ascent_ratio=10.0,
                        ascent_consistency=0.85,
                        persistence_ratio=9.5,
                    ),
                ),
                geometries=(
                    FadeGeometryEvidence(
                        outgoing=outgoing,
                        incoming=incoming,
                    ),
                ),
            ),
        ),
        kind=TransitionKind.HARD_CUT,
        temporal=TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            incoming_start_seconds=102.0,
            outgoing_end_seconds=100.0,
        ),
        outgoing_fade_selection=OutgoingFadeSelection(
            fade=outgoing,
            hypothesis=hypothesis,
            support_distance_seconds=1.1,
        ),
    )


def test_parser_accepts_routing_trace_path() -> None:
    args = cli._build_parser().parse_args(
        [
            "http://example.test/stream",
            "--output",
            "/tmp/tracks",
            "--routing-trace",
            "/tmp/routing.jsonl",
        ]
    )

    assert args.routing_trace == Path("/tmp/routing.jsonl")


def test_routing_trace_record_preserves_shadow_comparison() -> None:
    record = cli._routing_trace_record(_observation())

    assert record == {
        "reference_time_seconds": 100.0,
        "router_kind": "hard_cut",
        "acoustic_island_count": 0,
        "acoustic_island_detail": [],
        "semantic_outgoing_handoff_seconds": None,
        "semantic_incoming_handoff_seconds": None,
        "semantic_inserted_region_kind": None,
        "fade_routing_candidate": None,
        "fade_routing_two_sided_support": None,
        "fade_routing_outgoing_association": None,
        "fade_routing_incoming_association": None,
        "fade_routing_outgoing_support_seconds": None,
        "fade_routing_incoming_support_seconds": None,
        "interstitial_routing_candidate": None,
        "interstitial_routing_handoff_count": None,
        "interstitial_supported_handoff_count": None,
        "interstitial_absorbed_right_edge_fade": None,
        "interstitial_terminal_fade_support": None,
        "bounded_insert_ownership": None,
        "bounded_insert_symmetric_candidate": None,
        "bounded_insert_asymmetric_candidate": None,
        "hard_cut_fallback_eligible": None,
        "hard_cut_supported_handoff_count": None,
        "temporal_kind": "exclusion",
        "incoming_start_seconds": 102.0,
        "outgoing_end_seconds": 100.0,
        "pcm_boundary_scan_count": 0,
        "pcm_boundary_scan_detail": [],
        "fade_rise": False,
        "fade_rise_time_seconds": None,
        "fade_rise_jump_ratio": None,
        "fade_rise_persistence_ratio": None,
        "fade_rise_after_global_ratio": None,
        "fade_basins": 0,
        "fade_outgoing_count": 1,
        "fade_incoming_count": 1,
        "fade_geometry_count": 1,
        "fade_outgoing_selected": True,
        "fade_outgoing_selected_floor_seconds": 100.0,
        "fade_outgoing_selected_support_seconds": 101.1,
        "fade_outgoing_selected_distance_seconds": 1.1,
        "fade_incoming_selected": False,
        "fade_incoming_selected_start_seconds": None,
        "fade_incoming_selected_established_seconds": None,
        "fade_incoming_selected_support_seconds": None,
        "fade_incoming_selected_distance_seconds": None,
        "fade_outgoing_detail": [
            {
                "start_time_seconds": 99.0,
                "floor_time_seconds": 100.0,
                "rise_time_seconds": 101.0,
                "start_rms": 1200.0,
                "floor_rms": 120.0,
                "descent_ratio": 10.0,
                "descent_consistency": 0.9,
            }
        ],
        "fade_incoming_detail": [
            {
                "start_time_seconds": 100.4,
                "established_time_seconds": 101.0,
                "start_rms": 120.0,
                "established_rms": 1200.0,
                "ascent_ratio": 10.0,
                "ascent_consistency": 0.85,
                "persistence_ratio": 9.5,
            }
        ],
        "fade_geometry_detail": [
            {
                "outgoing_floor_seconds": 100.0,
                "incoming_start_seconds": 100.4,
                "separation_seconds": 0.4,
                "overlap_seconds": 0.0,
                "gap_seconds": 0.4,
                "outgoing_descent_consistency": 0.9,
                "incoming_ascent_consistency": 0.85,
            }
        ],
        "hypotheses": 1,
        "hypotheses_detail": [
            {
                "time_seconds": 101.1,
                "sources": ["basin", "structural"],
                "proposals_detail": [
                    {
                        "time_seconds": 101.0,
                        "source": "basin",
                        "strength": 0.82,
                    },
                    {
                        "time_seconds": 101.2,
                        "source": "structural",
                        "strength": 0.91,
                    },
                ],
                "evidence_detail": [
                    {
                        "proposal_time_seconds": 101.0,
                        "proposal_source": "basin",
                        "proposal_strength": 0.82,
                        "basin_depth": 0.24,
                        "basin_start_seconds": 100.7,
                        "basin_recovery_seconds": 101.6,
                        "local_change": 0.51,
                        "persistent_change": 0.63,
                        "structural_novelty": 0.74,
                        "metadata_support": False,
                    }
                ],
                "confidence": "high",
                "basin_start_seconds": 100.7,
                "basin_minimum_seconds": 101.0,
                "basin_recovery_seconds": 101.6,
            }
        ],
        "bounded_insert_count": 0,
        "bounded_insert_routing_candidate": None,
        "bounded_insert_routing_related_region_count": None,
        "bounded_insert_routing_unique_region": None,
        "bounded_insert_routing_left_structural_peak": None,
        "bounded_insert_routing_right_structural_peak": None,
        "bounded_insert_routing_reference_precedes_left": None,
        "bounded_insert_routing_structural_only_interior_count": None,
        "bounded_insert_detail": [],
        "local_complete": True,
        "local_discontinuity_250ms": 0.21,
        "local_discontinuity_500ms": 0.43,
        "local_discontinuity_1s": 0.65,
        "local_discontinuity_2s": 0.87,
    }


def test_routing_trace_record_exposes_exact_semantic_handoffs() -> None:
    from brynse.transition_router import (
        TransitionRoute,
        TransitionSemanticRoles,
    )

    base = _observation()
    hypothesis = base.context.hypotheses[0]

    route = TransitionRoute(
        kind=TransitionKind.HARD_CUT,
        semantic_roles=TransitionSemanticRoles(
            outgoing_handoff=hypothesis,
            incoming_handoff=hypothesis,
        ),
    )

    record = cli._routing_trace_record(
        replace(
            base,
            route=route,
        )
    )

    assert record["semantic_outgoing_handoff_seconds"] == 101.1
    assert record["semantic_incoming_handoff_seconds"] == 101.1
    assert record["semantic_inserted_region_kind"] is None


def test_routing_trace_record_exposes_exact_inserted_region_kind() -> None:
    from brynse.transition_router import (
        BoundedInsertOwnership,
        TransitionRoute,
        TransitionSemanticRoles,
    )

    base = _observation()

    route = TransitionRoute(
        kind=TransitionKind.BOUNDED_INSERT,
        bounded_insert_ownership=BoundedInsertOwnership.SYMMETRIC,
        semantic_roles=TransitionSemanticRoles(
            inserted_region=object(),
        ),
    )

    record = cli._routing_trace_record(
        replace(
            base,
            kind=TransitionKind.BOUNDED_INSERT,
            route=route,
        )
    )

    assert record["semantic_outgoing_handoff_seconds"] is None
    assert record["semantic_incoming_handoff_seconds"] is None
    assert record["semantic_inserted_region_kind"] == "symmetric"


def test_routing_trace_record_preserves_router_v2_assessments() -> None:
    from dataclasses import replace

    from brynse.transition_router import DefaultTransitionRouter

    base = _observation()
    router = DefaultTransitionRouter()
    context = base.context

    bounded_insert_routing = router.assess_bounded_insert_routing(
        context=context,
    )
    fade_routing = router.assess_fade(
        context=context,
    )
    interstitial_routing = router.assess_interstitial(
        context=context,
    )
    hard_cut_routing = router.assess_hard_cut(
        context=context,
    )

    observation = replace(
        base,
        bounded_insert_routing=bounded_insert_routing,
        fade_routing=fade_routing,
        interstitial_routing=interstitial_routing,
        hard_cut_routing=hard_cut_routing,
    )

    record = cli._routing_trace_record(observation)

    assert record["fade_routing_candidate"] is fade_routing.legacy_candidate
    assert (
        record["fade_routing_two_sided_support"]
        is fade_routing.has_two_sided_support
    )
    assert (
        record["fade_routing_outgoing_association"]
        is (fade_routing.outgoing_association is not None)
    )
    assert (
        record["fade_routing_incoming_association"]
        is (fade_routing.incoming_association is not None)
    )

    assert (
        record["interstitial_routing_candidate"]
        is interstitial_routing.legacy_candidate
    )
    assert record["interstitial_routing_handoff_count"] == len(
        interstitial_routing.routing_handoffs
    )
    assert record["interstitial_supported_handoff_count"] == len(
        interstitial_routing.supported_handoffs
    )
    assert (
        record["interstitial_absorbed_right_edge_fade"]
        is interstitial_routing.absorbed_right_edge_fade
    )
    assert (
        record["interstitial_terminal_fade_support"]
        is interstitial_routing.has_terminal_fade_support
    )

    expected_ownership = (
        None
        if bounded_insert_routing.ownership is None
        else bounded_insert_routing.ownership.value
    )

    assert record["bounded_insert_ownership"] == expected_ownership
    assert (
        record["bounded_insert_symmetric_candidate"]
        is bounded_insert_routing.symmetric_assessment.candidate
    )
    assert (
        record["bounded_insert_asymmetric_candidate"]
        is bounded_insert_routing.asymmetric_assessment.candidate
    )

    assert (
        record["hard_cut_fallback_eligible"]
        is hard_cut_routing.fallback_eligible
    )
    assert record["hard_cut_supported_handoff_count"] == len(
        hard_cut_routing.supported_handoffs
    )


def test_routing_trace_record_preserves_fade_rise_diagnostics() -> None:
    observation = _observation()

    rise = CrossfadeRiseEvidence(
        time_seconds=101.75,
        jump_ratio=3.25,
        persistence_ratio=4.5,
        after_global_ratio=2.75,
    )

    observation = replace(
        observation,
        context=replace(
            observation.context,
            fade=replace(
                observation.context.fade,
                rise=rise,
            ),
        ),
    )

    record = cli._routing_trace_record(observation)

    assert record["fade_rise"] is True
    assert record["fade_rise_time_seconds"] == 101.75
    assert record["fade_rise_jump_ratio"] == 3.25
    assert record["fade_rise_persistence_ratio"] == 4.5
    assert record["fade_rise_after_global_ratio"] == 2.75


def test_run_stream_writes_routing_trace(monkeypatch, tmp_path) -> None:
    observation = _observation()
    trace_path = tmp_path / "routing.jsonl"

    class FakeRunner:
        current_track_title = None

        def __init__(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            callback = kwargs["on_routing_observation"]
            assert callback is not None
            callback(observation)
            return SimpleNamespace(codec="mp3", metaint=32768)

        def stop(self):
            pass

    monkeypatch.setattr(cli, "RippingRunner", FakeRunner)

    args = Namespace(
        url="http://example.test/stream",
        output=tmp_path / "tracks",
        codec="mp3",
        ffmpeg="ffmpeg",
        metadata_threshold=8.0,
        metadata_order="preserve",
        transient_exclusion=True,
        autonomous_boundaries=False,
        routing_trace=trace_path,
        session_capture=None,
    )

    assert cli._run_stream(args) == 0

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    assert json.loads(lines[0]) == cli._routing_trace_record(observation)



def test_routing_trace_record_exposes_acoustic_island_evidence() -> None:
    from brynse.acoustic_island import AcousticIslandEvidence

    base = _observation()
    hypothesis = base.context.hypotheses[0]
    outgoing = base.context.fade.outgoing_fades[0]
    incoming = base.context.fade.incoming_fades[0]

    island = AcousticIslandEvidence(
        incoming_handoff=hypothesis,
        terminal_decay=outgoing,
        source_reset=incoming,
    )

    context = replace(
        base.context,
        acoustic_islands=(island,),
    )

    record = cli._routing_trace_record(
        replace(
            base,
            context=context,
        )
    )

    assert record["acoustic_island_count"] == 1
    assert record["acoustic_island_detail"] == [
        {
            "incoming_handoff_seconds": hypothesis.time_seconds,
            "terminal_decay_start_seconds": outgoing.start_time_seconds,
            "terminal_decay_floor_seconds": outgoing.floor_time_seconds,
            "source_reset_start_seconds": incoming.start_time_seconds,
            "source_reset_established_seconds": (
                incoming.established_time_seconds
            ),
            "observed_region_left_seconds": None,
            "observed_region_right_seconds": None,
        }
    ]
