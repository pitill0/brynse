import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
)
from brynse.integrations.radio import cli
from brynse.models import TemporalSplitDecision, TemporalSplitKind
from brynse.orchestrator import TransitionRoutingObservation
from brynse.transition_evidence import (
    FadeRoutingEvidence,
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
                BoundaryHypothesis(
                    time_seconds=101.1,
                    proposals=(
                        basin := BoundaryProposal(
                            time_seconds=101.0,
                            source=BoundaryProposalSource.BASIN,
                        ),
                        BoundaryProposal(
                            time_seconds=101.2,
                            source=BoundaryProposalSource.STRUCTURAL,
                        ),
                    ),
                    evidence=(
                        BoundaryEvidence(
                            proposal=basin,
                            basin_start_seconds=100.7,
                            basin_recovery_seconds=101.6,
                        ),
                    ),
                    confidence=BoundaryConfidence.HIGH,
                ),
            ),
            fade=FadeRoutingEvidence(),
        ),
        kind=TransitionKind.HARD_CUT,
        temporal=TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            incoming_start_seconds=102.0,
            outgoing_end_seconds=100.0,
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
        "temporal_kind": "exclusion",
        "incoming_start_seconds": 102.0,
        "outgoing_end_seconds": 100.0,
        "fade_rise": False,
        "fade_basins": 0,
        "hypotheses": 1,
        "hypotheses_detail": [
            {
                "time_seconds": 101.1,
                "sources": ["basin", "structural"],
                "confidence": "high",
                "basin_start_seconds": 100.7,
                "basin_minimum_seconds": 101.0,
                "basin_recovery_seconds": 101.6,
            }
        ],
        "local_complete": True,
        "local_discontinuity_250ms": 0.21,
        "local_discontinuity_500ms": 0.43,
        "local_discontinuity_1s": 0.65,
        "local_discontinuity_2s": 0.87,
    }


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
        transient_exclusion=True,
        autonomous_boundaries=False,
        routing_trace=trace_path,
    )

    assert cli._run_stream(args) == 0

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    assert json.loads(lines[0]) == cli._routing_trace_record(observation)
