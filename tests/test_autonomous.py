from __future__ import annotations

from brynse.autonomous import AutonomousBoundaryDetector
from brynse.boundaries import BoundaryProposalSource, BoundaryReconciler
from brynse.models import AcousticLevel, AcousticProfile


def _profile(values: list[float], *, step: float = 0.5) -> AcousticProfile:
    return AcousticProfile(
        levels=tuple(
            AcousticLevel(
                start_time_seconds=index * step,
                end_time_seconds=(index + 1) * step,
                rms=value,
            )
            for index, value in enumerate(values)
        )
    )


def test_detects_clear_basin_followed_by_persistent_attack() -> None:
    values = (
        [1000.0] * 12 + [120.0, 100.0] + [900.0, 950.0, 1000.0, 980.0, 970.0, 960.0] + [1000.0] * 8
    )
    detector = AutonomousBoundaryDetector(min_track_seconds=1.0)

    candidates = detector.detect(profile=_profile(values))

    assert len(candidates) == 1
    assert candidates[0].time_seconds == 7.0
    assert candidates[0].basin_start_seconds == 6.0


def test_rejects_shallow_internal_dip() -> None:
    values = [1000.0] * 12 + [700.0, 650.0] + [1000.0] * 14
    detector = AutonomousBoundaryDetector(min_track_seconds=1.0)

    assert detector.detect(profile=_profile(values)) == ()


def test_rejects_non_persistent_attack() -> None:
    values = (
        [1000.0] * 12
        + [100.0, 100.0]
        + [900.0, 900.0]
        + [120.0, 110.0, 100.0, 100.0]
        + [1000.0] * 8
    )
    detector = AutonomousBoundaryDetector(min_track_seconds=1.0)

    assert detector.detect(profile=_profile(values)) == ()


def test_applies_minimum_track_spacing() -> None:
    first = [1000.0] * 12 + [100.0, 100.0] + [1000.0] * 8
    second = [1000.0] * 4 + [100.0, 100.0] + [1000.0] * 8
    detector = AutonomousBoundaryDetector(min_track_seconds=10.0)

    candidates = detector.detect(profile=_profile(first + second))

    assert len(candidates) == 1


def test_absolute_start_time_is_applied() -> None:
    values = [1000.0] * 12 + [100.0, 100.0] + [1000.0] * 10
    detector = AutonomousBoundaryDetector(min_track_seconds=1.0)

    candidate = detector.detect(
        profile=_profile(values),
        absolute_start_time_seconds=100.0,
    )[0]

    assert candidate.time_seconds == 107.0


def test_candidate_converts_to_basin_boundary_proposal() -> None:
    values = [1000.0] * 12 + [100.0, 100.0] + [1000.0] * 10
    detector = AutonomousBoundaryDetector(min_track_seconds=1.0)

    candidate = detector.detect(profile=_profile(values))[0]
    proposal = candidate.as_proposal()

    assert proposal.time_seconds == candidate.time_seconds
    assert proposal.source.value == "basin"
    assert proposal.strength is None


def test_detector_exposes_boundary_proposals_without_losing_absolute_time() -> None:
    values = [1000.0] * 12 + [100.0, 100.0] + [1000.0] * 10
    detector = AutonomousBoundaryDetector(min_track_seconds=1.0)

    proposals = detector.detect_proposals(
        profile=_profile(values),
        absolute_start_time_seconds=100.0,
    )

    assert len(proposals) == 1
    assert proposals[0].time_seconds == 107.0
    assert proposals[0].source is BoundaryProposalSource.BASIN
    assert proposals[0].strength is None


def test_autonomous_proposals_flow_into_boundary_reconciler() -> None:
    first = [1000.0] * 12 + [100.0, 100.0] + [1000.0] * 8
    second = [1000.0] * 4 + [100.0, 100.0] + [1000.0] * 8
    detector = AutonomousBoundaryDetector(min_track_seconds=1.0)

    proposals = detector.detect_proposals(profile=_profile(first + second))
    hypotheses = BoundaryReconciler(cluster_radius_seconds=3.0).reconcile(proposals)

    assert len(proposals) == 2
    assert len(hypotheses) == 2
    assert hypotheses[0].proposals == (proposals[0],)
    assert hypotheses[1].proposals == (proposals[1],)
