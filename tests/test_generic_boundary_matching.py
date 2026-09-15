from fluxtuner_ripper.matching import NearestBoundaryMatcher
from fluxtuner_ripper.models import AcousticBoundaryCandidate, BoundaryCandidate


def test_generic_boundary_candidate_matches_nearest_acoustic_candidate() -> None:
    matcher = NearestBoundaryMatcher(search_radius_seconds=2.0)

    boundary = BoundaryCandidate(
        time_seconds=10.0,
        source="external",
    )

    earlier = AcousticBoundaryCandidate(
        time_seconds=9.8,
        rms=0.2,
        relative_time_seconds=1.0,
    )
    later = AcousticBoundaryCandidate(
        time_seconds=10.4,
        rms=0.1,
        relative_time_seconds=1.6,
    )

    selected = matcher.match_candidate(
        candidate=boundary,
        acoustic_candidates=(earlier, later),
    )

    assert selected == earlier


def test_generic_boundary_candidate_returns_none_outside_radius() -> None:
    matcher = NearestBoundaryMatcher(search_radius_seconds=1.0)

    boundary = BoundaryCandidate(
        time_seconds=10.0,
        source="agent",
    )

    selected = matcher.match_candidate(
        candidate=boundary,
        acoustic_candidates=(
            AcousticBoundaryCandidate(
                time_seconds=12.0,
                rms=0.1,
                relative_time_seconds=2.0,
            ),
        ),
    )

    assert selected is None
