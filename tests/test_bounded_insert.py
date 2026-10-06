import pytest

from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
)
from brynse.bounded_insert import (
    BoundedInsertRegionDiscovery,
    BoundedInsertRegionEvidence,
)


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


def test_region_represents_structural_only_edge_without_basin_geometry() -> None:
    left_basin = BoundaryProposal(
        time_seconds=100.0,
        source=BoundaryProposalSource.BASIN,
    )
    left_structural = BoundaryProposal(
        time_seconds=100.2,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.8,
    )
    left = BoundaryHypothesis(
        time_seconds=100.1,
        proposals=(left_basin, left_structural),
        evidence=(
            BoundaryEvidence(
                proposal=left_basin,
                basin_start_seconds=99.5,
                basin_recovery_seconds=100.7,
            ),
        ),
        confidence=BoundaryConfidence.HIGH,
    )

    right_structural = BoundaryProposal(
        time_seconds=105.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.4,
    )
    right = BoundaryHypothesis(
        time_seconds=105.0,
        proposals=(right_structural,),
        evidence=(),
        confidence=BoundaryConfidence.UNRESOLVED,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=104.5,
        left=left,
        right=right,
    )

    assert region.left_geometry is not None
    assert region.right_geometry is None
    assert region.acoustic_gap_seconds is None
    assert region.reference_inside_right_basin is False
    assert region.right_edge.basin_geometry is None
    assert region.right_edge.proposals == (right_structural,)


def test_region_represents_basin_only_edge_without_structural_support() -> None:
    left_basin = BoundaryProposal(
        time_seconds=100.0,
        source=BoundaryProposalSource.BASIN,
    )
    left_structural = BoundaryProposal(
        time_seconds=100.2,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.8,
    )
    left = BoundaryHypothesis(
        time_seconds=100.1,
        proposals=(left_basin, left_structural),
        evidence=(
            BoundaryEvidence(
                proposal=left_basin,
                basin_start_seconds=99.5,
                basin_recovery_seconds=100.7,
            ),
        ),
        confidence=BoundaryConfidence.HIGH,
    )

    right_basin = BoundaryProposal(
        time_seconds=105.0,
        source=BoundaryProposalSource.BASIN,
    )
    right = BoundaryHypothesis(
        time_seconds=105.0,
        proposals=(right_basin,),
        evidence=(
            BoundaryEvidence(
                proposal=right_basin,
                basin_start_seconds=104.7,
                basin_recovery_seconds=105.4,
            ),
        ),
        confidence=BoundaryConfidence.UNRESOLVED,
    )

    region = BoundedInsertRegionEvidence(
        reference_time_seconds=104.0,
        left=left,
        right=right,
    )

    assert region.left_geometry is not None
    assert region.right_geometry is not None
    assert region.acoustic_gap_seconds == 4.0
    assert region.reference_inside_right_basin is False
    assert region.right_edge.proposals == (right_basin,)


def _asymmetric_primary(
    *,
    time_seconds: float,
    basin_start_seconds: float,
    basin_recovery_seconds: float,
    structural_strength: float,
) -> BoundaryHypothesis:
    basin = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.BASIN,
    )
    structural = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=structural_strength,
    )
    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=(basin, structural),
        evidence=(
            BoundaryEvidence(
                proposal=basin,
                basin_start_seconds=basin_start_seconds,
                basin_recovery_seconds=basin_recovery_seconds,
            ),
        ),
        confidence=BoundaryConfidence.UNRESOLVED,
    )


def _structural_only_edge(
    *,
    time_seconds: float,
    strength: float,
) -> BoundaryHypothesis:
    proposal = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=strength,
    )
    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=(proposal,),
        evidence=(),
        confidence=BoundaryConfidence.UNRESOLVED,
    )


def _basin_only_edge(
    *,
    time_seconds: float,
    basin_start_seconds: float,
    basin_recovery_seconds: float,
) -> BoundaryHypothesis:
    proposal = BoundaryProposal(
        time_seconds=time_seconds,
        source=BoundaryProposalSource.BASIN,
    )
    return BoundaryHypothesis(
        time_seconds=time_seconds,
        proposals=(proposal,),
        evidence=(
            BoundaryEvidence(
                proposal=proposal,
                basin_start_seconds=basin_start_seconds,
                basin_recovery_seconds=basin_recovery_seconds,
            ),
        ),
        confidence=BoundaryConfidence.UNRESOLVED,
    )


def test_asymmetric_discovery_describes_structural_complement() -> None:
    primary = _asymmetric_primary(
        time_seconds=100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.8,
        structural_strength=0.8,
    )
    interior = _structural_only_edge(
        time_seconds=108.0,
        strength=0.2,
    )
    complementary = _structural_only_edge(
        time_seconds=119.0,
        strength=0.3,
    )

    regions = BoundedInsertRegionDiscovery().discover_asymmetric(
        reference_time_seconds=118.5,
        hypotheses=(primary, interior, complementary),
    )

    assert len(regions) == 1
    region = regions[0]

    assert region.primary is primary
    assert region.complementary is complementary
    assert region.interior_hypotheses == (interior,)
    assert region.primary_distance_from_reference_seconds == 18.5
    assert region.complementary_distance_from_reference_seconds == 0.5
    assert region.complementary_edge.basin_geometry is None


def test_asymmetric_discovery_describes_basin_complement() -> None:
    primary = _asymmetric_primary(
        time_seconds=100.0,
        basin_start_seconds=95.0,
        basin_recovery_seconds=100.5,
        structural_strength=0.8,
    )
    complementary = _basin_only_edge(
        time_seconds=103.0,
        basin_start_seconds=102.2,
        basin_recovery_seconds=103.2,
    )

    regions = BoundedInsertRegionDiscovery().discover_asymmetric(
        reference_time_seconds=101.6,
        hypotheses=(primary, complementary),
    )

    assert len(regions) == 1
    region = regions[0]

    assert region.primary is primary
    assert region.complementary is complementary
    assert region.complementary_edge.basin_geometry is not None


def test_asymmetric_discovery_abstains_when_primary_is_nearer() -> None:
    primary = _asymmetric_primary(
        time_seconds=100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.8,
    )
    complementary = _structural_only_edge(
        time_seconds=107.0,
        strength=0.3,
    )

    regions = BoundedInsertRegionDiscovery().discover_asymmetric(
        reference_time_seconds=101.0,
        hypotheses=(primary, complementary),
    )

    assert regions == ()


def test_asymmetric_discovery_abstains_with_intervening_primary() -> None:
    first = _asymmetric_primary(
        time_seconds=100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.5,
    )
    second = _asymmetric_primary(
        time_seconds=108.0,
        basin_start_seconds=107.5,
        basin_recovery_seconds=108.5,
        structural_strength=0.8,
    )
    complementary = _structural_only_edge(
        time_seconds=111.0,
        strength=0.3,
    )

    regions = BoundedInsertRegionDiscovery().discover_asymmetric(
        reference_time_seconds=109.6,
        hypotheses=(first, second, complementary),
    )

    assert len(regions) == 1
    assert regions[0].primary is second
    assert regions[0].complementary is complementary


def test_symmetric_discovery_remains_unchanged_by_asymmetric_support() -> None:
    primary = _asymmetric_primary(
        time_seconds=100.0,
        basin_start_seconds=99.5,
        basin_recovery_seconds=100.5,
        structural_strength=0.8,
    )
    complementary = _structural_only_edge(
        time_seconds=104.0,
        strength=0.3,
    )

    regions = BoundedInsertRegionDiscovery().discover(
        reference_time_seconds=102.0,
        hypotheses=(primary, complementary),
    )

    assert regions == ()


@pytest.mark.parametrize(
    (
        "reference_time_seconds",
        "primary_time_seconds",
        "primary_basin_start_seconds",
        "primary_basin_recovery_seconds",
        "complementary_time_seconds",
        "complementary_kind",
    ),
    (
        (
            1363.958,
            1344.876,
            1344.251,
            1348.051,
            1364.451,
            "structural",
        ),
        (
            1780.376,
            1778.519,
            1773.069,
            1778.769,
            1781.919,
            "basin",
        ),
    ),
)
def test_live_failures_are_discovered_as_asymmetric_regions(
    reference_time_seconds,
    primary_time_seconds,
    primary_basin_start_seconds,
    primary_basin_recovery_seconds,
    complementary_time_seconds,
    complementary_kind,
) -> None:
    primary = _asymmetric_primary(
        time_seconds=primary_time_seconds,
        basin_start_seconds=primary_basin_start_seconds,
        basin_recovery_seconds=primary_basin_recovery_seconds,
        structural_strength=0.8,
    )

    if complementary_kind == "structural":
        complementary = _structural_only_edge(
            time_seconds=complementary_time_seconds,
            strength=0.3,
        )
    else:
        complementary = _basin_only_edge(
            time_seconds=complementary_time_seconds,
            basin_start_seconds=1781.169,
            basin_recovery_seconds=1782.069,
        )

    regions = BoundedInsertRegionDiscovery().discover_asymmetric(
        reference_time_seconds=reference_time_seconds,
        hypotheses=(primary, complementary),
    )

    assert len(regions) == 1

    region = regions[0]

    assert region.primary is primary
    assert region.complementary is complementary
    assert (
        region.complementary_distance_from_reference_seconds
        < region.primary_distance_from_reference_seconds
    )


@pytest.mark.parametrize(
    (
        "reference_time_seconds",
        "primary_time_seconds",
        "complementary_time_seconds",
    ),
    (
        (1028.075, 1027.919, 1035.069),
        (4034.560, 4034.278, 4038.053),
        (4704.914, 4704.408, 4707.408),
    ),
)
def test_live_good_controls_do_not_become_asymmetric_regions(
    reference_time_seconds,
    primary_time_seconds,
    complementary_time_seconds,
) -> None:
    primary = _asymmetric_primary(
        time_seconds=primary_time_seconds,
        basin_start_seconds=primary_time_seconds - 0.5,
        basin_recovery_seconds=primary_time_seconds + 0.5,
        structural_strength=0.8,
    )
    complementary = _structural_only_edge(
        time_seconds=complementary_time_seconds,
        strength=0.3,
    )

    regions = BoundedInsertRegionDiscovery().discover_asymmetric(
        reference_time_seconds=reference_time_seconds,
        hypotheses=(primary, complementary),
    )

    assert regions == ()


@pytest.mark.parametrize(
    (
        "reference_time_seconds",
        "left_time_seconds",
        "right_time_seconds",
    ),
    (
        (363.154, 356.173, 374.123),
        (652.617, 651.861, 659.761),
        (836.937, 835.781, 843.156),
        (3493.878, 3493.321, 3505.396),
    ),
)
def test_live_symmetric_bounded_insert_pairs_remain_discoverable(
    reference_time_seconds,
    left_time_seconds,
    right_time_seconds,
) -> None:
    left = _asymmetric_primary(
        time_seconds=left_time_seconds,
        basin_start_seconds=left_time_seconds - 0.5,
        basin_recovery_seconds=left_time_seconds + 0.5,
        structural_strength=0.3,
    )
    right = _asymmetric_primary(
        time_seconds=right_time_seconds,
        basin_start_seconds=right_time_seconds - 0.5,
        basin_recovery_seconds=right_time_seconds + 0.5,
        structural_strength=0.2,
    )

    regions = BoundedInsertRegionDiscovery().discover(
        reference_time_seconds=reference_time_seconds,
        hypotheses=(left, right),
    )

    assert len(regions) == 1
    assert regions[0].left is left
    assert regions[0].right is right
