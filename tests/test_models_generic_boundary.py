import pytest

from fluxtuner_ripper.models import BoundaryCandidate, TrackCandidate


def test_boundary_candidate_accepts_generic_source_and_optional_offset() -> None:
    candidate = BoundaryCandidate(
        time_seconds=123.456,
        source="external",
        reference_offset=9876,
    )

    assert candidate.time_seconds == 123.456
    assert candidate.source == "external"
    assert candidate.reference_offset == 9876


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"time_seconds": -1.0, "source": "external"},
            "time_seconds must be non-negative",
        ),
        (
            {"time_seconds": 1.0, "source": "   "},
            "source must not be empty",
        ),
        (
            {
                "time_seconds": 1.0,
                "source": "external",
                "reference_offset": -1,
            },
            "reference_offset must be non-negative",
        ),
    ],
)
def test_boundary_candidate_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BoundaryCandidate(**kwargs)


def test_track_candidate_projects_to_generic_metadata_boundary() -> None:
    track = TrackCandidate(
        title="Artist - Track",
        start_offset=12345,
        start_time_seconds=42.5,
        confirmed_at_offset=23456,
        confirmed_at_time_seconds=50.5,
    )

    candidate = track.as_boundary_candidate()

    assert candidate == BoundaryCandidate(
        time_seconds=42.5,
        source="metadata",
        reference_offset=12345,
    )
