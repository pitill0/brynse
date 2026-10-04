from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
)
from brynse.bounded_insert import BoundedInsertRegionDiscovery


def _edge(
    time_seconds: float,
    *,
    start_seconds: float,
    recovery_seconds: float,
    structural: bool = True,
) -> BoundaryHypothesis:
    basin = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.BASIN,
    )
    proposals = [basin]

    if structural:
        proposals.append(
            BoundaryProposal(
                time_seconds=time_seconds + 0.1,
                source=BoundaryProposalSource.STRUCTURAL,
            )
        )

    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=tuple(proposals),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_start_seconds=start_seconds,
                basin_recovery_seconds=recovery_seconds,
            ),
        ),
    )


def _structural(time_seconds: float) -> BoundaryHypothesis:
    proposal = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.STRUCTURAL,
    )

    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=(proposal,),
    )


def test_discovery_describes_geometry_between_supported_edges() -> None:
    left = _edge(
        100.0,
        start_seconds=99.5,
        recovery_seconds=100.5,
    )
    interior = _structural(102.0)
    right = _edge(
        105.0,
        start_seconds=104.5,
        recovery_seconds=105.5,
    )

    regions = BoundedInsertRegionDiscovery().discover(
        reference_time_seconds=101.0,
        hypotheses=(right, interior, left),
    )

    assert len(regions) == 1

    region = regions[0]

    assert region.left is left
    assert region.right is right
    assert region.interior_hypotheses == (interior,)
    assert region.support_span_seconds == 5.0
    assert region.acoustic_gap_seconds == 4.0
    assert region.reference_offset_from_left_support_seconds == 1.0
    assert region.reference_offset_to_right_support_seconds == 4.0
    assert region.structural_only_interior_count == 1


def test_discovery_preserves_reference_relationship_to_basin() -> None:
    left = _edge(
        100.0,
        start_seconds=99.0,
        recovery_seconds=103.0,
    )
    right = _edge(
        110.0,
        start_seconds=109.0,
        recovery_seconds=111.0,
    )

    region = BoundedInsertRegionDiscovery().discover(
        reference_time_seconds=102.0,
        hypotheses=(left, right),
    )[0]

    assert region.reference_inside_left_basin is True
    assert region.reference_inside_right_basin is False


def test_discovery_keeps_region_when_reference_precedes_support_inside_basin() -> None:
    left = _edge(
        100.0,
        start_seconds=98.0,
        recovery_seconds=104.0,
    )
    right = _edge(
        110.0,
        start_seconds=109.0,
        recovery_seconds=111.0,
    )

    regions = BoundedInsertRegionDiscovery().discover(
        reference_time_seconds=99.5,
        hypotheses=(left, right),
    )

    assert len(regions) == 1
    assert regions[0].reference_inside_left_basin is True
    assert regions[0].reference_offset_from_left_support_seconds == -0.5


def test_discovery_requires_basin_and_structural_support() -> None:
    unsupported = _edge(
        100.0,
        start_seconds=99.5,
        recovery_seconds=100.5,
        structural=False,
    )
    supported = _edge(
        105.0,
        start_seconds=104.5,
        recovery_seconds=105.5,
    )

    regions = BoundedInsertRegionDiscovery().discover(
        reference_time_seconds=102.0,
        hypotheses=(unsupported, supported),
    )

    assert regions == ()


def test_discovery_ignores_unrelated_region() -> None:
    left = _edge(
        100.0,
        start_seconds=99.5,
        recovery_seconds=100.5,
    )
    right = _edge(
        105.0,
        start_seconds=104.5,
        recovery_seconds=105.5,
    )

    regions = BoundedInsertRegionDiscovery().discover(
        reference_time_seconds=110.0,
        hypotheses=(left, right),
    )

    assert regions == ()


def test_discovery_preserves_adjacent_regions_at_shared_edge() -> None:
    left = _edge(
        100.0,
        start_seconds=99.5,
        recovery_seconds=100.5,
    )
    middle = _edge(
        105.0,
        start_seconds=104.5,
        recovery_seconds=105.5,
    )
    right = _edge(
        110.0,
        start_seconds=109.5,
        recovery_seconds=110.5,
    )

    regions = BoundedInsertRegionDiscovery().discover(
        reference_time_seconds=105.0,
        hypotheses=(left, middle, right),
    )

    assert tuple((region.left, region.right) for region in regions) == (
        (left, middle),
        (middle, right),
    )


def test_region_preserves_complete_neutral_edge_evidence() -> None:
    left_basin = BoundaryProposal(
        time_seconds=100.0,
        source=BoundaryProposalSource.BASIN,
        strength=0.82,
    )
    left_structural = BoundaryProposal(
        time_seconds=100.2,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.91,
    )
    left_basin_evidence = BoundaryEvidence(
        proposal=left_basin,
        basin_depth=0.24,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.7,
        local_change=0.51,
        persistent_change=0.63,
        structural_novelty=0.74,
        metadata_support=False,
    )
    left = BoundaryHypothesis(
        time_seconds=100.1,
        proposals=(left_basin, left_structural),
        evidence=(left_basin_evidence,),
        confidence=BoundaryConfidence.HIGH,
    )

    right_basin = BoundaryProposal(
        time_seconds=105.0,
        source=BoundaryProposalSource.BASIN,
        strength=None,
    )
    right_structural = BoundaryProposal(
        time_seconds=105.2,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.88,
    )
    right_basin_evidence = BoundaryEvidence(
        proposal=right_basin,
        basin_depth=0.08,
        basin_start_seconds=104.7,
        basin_recovery_seconds=105.5,
        local_change=None,
        persistent_change=0.70,
        structural_novelty=0.81,
        metadata_support=None,
    )
    right = BoundaryHypothesis(
        time_seconds=105.1,
        proposals=(right_basin, right_structural),
        evidence=(right_basin_evidence,),
        confidence=BoundaryConfidence.MEDIUM,
    )

    region = BoundedInsertRegionDiscovery().discover(
        reference_time_seconds=102.0,
        hypotheses=(left, right),
    )[0]

    assert region.left_edge.hypothesis is left
    assert region.left_edge.confidence is BoundaryConfidence.HIGH
    assert region.left_edge.proposals == (
        left_basin,
        left_structural,
    )
    assert region.left_edge.evidence == (left_basin_evidence,)
    assert region.left_edge.basin_geometry == left.basin_geometry

    assert region.right_edge.hypothesis is right
    assert region.right_edge.confidence is BoundaryConfidence.MEDIUM
    assert region.right_edge.proposals == (
        right_basin,
        right_structural,
    )
    assert region.right_edge.evidence == (right_basin_evidence,)

    # Missing measurements remain missing.  The observer must not
    # reinterpret absence as zero acoustic evidence.
    assert region.right_edge.proposals[0].strength is None
    assert region.right_edge.evidence[0].local_change is None
    assert region.right_edge.evidence[0].metadata_support is None
