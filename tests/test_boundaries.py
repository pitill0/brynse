from __future__ import annotations

import pytest

from fluxtuner_ripper.boundaries import (
    BoundaryConfidence,
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
    BoundaryReconciler,
)


def _proposal(
    time_seconds: float,
    *,
    source: BoundaryProposalSource = BoundaryProposalSource.BASIN,
    strength: float | None = None,
) -> BoundaryProposal:
    return BoundaryProposal(
        time_seconds=time_seconds,
        source=source,
        strength=strength,
    )


def test_boundary_proposal_keeps_audio_source_explicit() -> None:
    proposal = _proposal(
        120.5,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.75,
    )

    assert proposal.time_seconds == 120.5
    assert proposal.source is BoundaryProposalSource.STRUCTURAL
    assert proposal.strength == 0.75


@pytest.mark.parametrize("strength", [-0.01, 1.01])
def test_boundary_proposal_rejects_invalid_strength(strength: float) -> None:
    with pytest.raises(ValueError, match="strength"):
        _proposal(10.0, strength=strength)


def test_boundary_evidence_keeps_signals_descriptive() -> None:
    proposal = _proposal(100.0)
    evidence = BoundaryEvidence(
        proposal=proposal,
        basin_depth=0.04,
        local_change=0.55,
        persistent_change=0.32,
        structural_novelty=1.10,
        metadata_support=True,
    )

    assert evidence.proposal == proposal
    assert evidence.structural_novelty == 1.10
    assert evidence.metadata_support is True


def test_boundary_hypothesis_defaults_to_unresolved_confidence() -> None:
    proposal = _proposal(100.0)

    hypothesis = BoundaryHypothesis(
        time_seconds=100.0,
        proposals=(proposal,),
    )

    assert hypothesis.confidence is BoundaryConfidence.UNRESOLVED


def test_reconciler_splits_proposals_when_cluster_span_exceeds_radius() -> None:
    proposals = (
        _proposal(274.75),
        _proposal(276.95, source=BoundaryProposalSource.STRUCTURAL),
        _proposal(278.35),
    )

    hypotheses = BoundaryReconciler(cluster_radius_seconds=3.0).reconcile(proposals)

    assert len(hypotheses) == 2
    assert hypotheses[0].proposals == proposals[:2]
    assert hypotheses[0].time_seconds == pytest.approx(275.85)
    assert hypotheses[1].proposals == proposals[2:]
    assert hypotheses[1].time_seconds == pytest.approx(278.35)


def test_reconciler_separates_distant_proposal_clusters() -> None:
    proposals = (
        _proposal(100.0),
        _proposal(101.5, source=BoundaryProposalSource.STRUCTURAL),
        _proposal(120.0),
    )

    hypotheses = BoundaryReconciler(cluster_radius_seconds=3.0).reconcile(proposals)

    assert len(hypotheses) == 2
    assert hypotheses[0].time_seconds == pytest.approx(100.75)
    assert hypotheses[1].time_seconds == pytest.approx(120.0)


def test_reconciler_attaches_evidence_to_its_cluster() -> None:
    first = _proposal(100.0)
    nearby = _proposal(101.0, source=BoundaryProposalSource.STRUCTURAL)
    distant = _proposal(130.0)

    first_evidence = BoundaryEvidence(proposal=first, basin_depth=0.04)
    nearby_evidence = BoundaryEvidence(
        proposal=nearby,
        structural_novelty=0.7,
    )
    distant_evidence = BoundaryEvidence(proposal=distant, local_change=0.5)

    hypotheses = BoundaryReconciler().reconcile(
        (first, nearby, distant),
        evidence=(first_evidence, nearby_evidence, distant_evidence),
    )

    assert hypotheses[0].evidence == (first_evidence, nearby_evidence)
    assert hypotheses[1].evidence == (distant_evidence,)


def test_reconciler_rejects_evidence_for_unknown_proposal() -> None:
    known = _proposal(100.0)
    unknown = _proposal(200.0)

    with pytest.raises(ValueError, match="unknown proposal"):
        BoundaryReconciler().reconcile(
            (known,),
            evidence=(BoundaryEvidence(proposal=unknown),),
        )


def test_reconciler_rejects_non_positive_cluster_radius() -> None:
    with pytest.raises(ValueError, match="cluster_radius_seconds"):
        BoundaryReconciler(cluster_radius_seconds=0.0)


def test_reconciler_does_not_chain_proposals_beyond_cluster_span() -> None:
    proposals = (
        _proposal(0.0),
        _proposal(2.0, source=BoundaryProposalSource.STRUCTURAL),
        _proposal(4.0),
        _proposal(6.0, source=BoundaryProposalSource.STRUCTURAL),
        _proposal(8.0),
    )

    hypotheses = BoundaryReconciler(cluster_radius_seconds=3.0).reconcile(proposals)

    assert len(hypotheses) == 3
    assert hypotheses[0].proposals == proposals[:2]
    assert hypotheses[0].time_seconds == pytest.approx(1.0)
    assert hypotheses[1].proposals == proposals[2:4]
    assert hypotheses[1].time_seconds == pytest.approx(5.0)
    assert hypotheses[2].proposals == proposals[4:]


def test_reconciler_limits_cluster_total_span_to_radius() -> None:
    proposals = (
        _proposal(10.0),
        _proposal(12.5, source=BoundaryProposalSource.STRUCTURAL),
        _proposal(15.0),
    )

    hypotheses = BoundaryReconciler(cluster_radius_seconds=3.0).reconcile(proposals)

    assert len(hypotheses) == 2
    assert hypotheses[0].proposals == proposals[:2]
    assert hypotheses[0].time_seconds == pytest.approx(11.25)
    assert hypotheses[1].proposals == proposals[2:]
    assert hypotheses[1].time_seconds == pytest.approx(15.0)


def test_reconciler_does_not_merge_structural_events_six_seconds_apart() -> None:
    proposals = (
        _proposal(317.5, source=BoundaryProposalSource.STRUCTURAL),
        _proposal(323.5, source=BoundaryProposalSource.STRUCTURAL),
    )

    hypotheses = BoundaryReconciler(cluster_radius_seconds=3.0).reconcile(proposals)

    assert len(hypotheses) == 2
    assert hypotheses[0].time_seconds == pytest.approx(317.5)
    assert hypotheses[1].time_seconds == pytest.approx(323.5)
