import pytest

from fluxtuner_ripper.models import BoundaryCandidate
from fluxtuner_ripper.radio_models import TrackCandidate


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


def test_segment_accepts_stream_position_and_optional_label() -> None:
    from fluxtuner_ripper.models import Segment

    segment = Segment(
        start_offset=1234,
        start_time_seconds=12.5,
        label="logical unit",
    )

    assert segment.start_offset == 1234
    assert segment.start_time_seconds == 12.5
    assert segment.label == "logical unit"


def test_track_candidate_projects_to_generic_segment() -> None:
    from fluxtuner_ripper.models import Segment

    track = TrackCandidate(
        title="Artist - Track",
        start_offset=12345,
        start_time_seconds=42.5,
        confirmed_at_offset=23456,
        confirmed_at_time_seconds=50.5,
    )

    assert track.as_segment() == Segment(
        start_offset=12345,
        start_time_seconds=42.5,
        label="Artist - Track",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"start_offset": -1, "start_time_seconds": 1.0},
            "start_offset must be non-negative",
        ),
        (
            {"start_offset": 0, "start_time_seconds": -1.0},
            "start_time_seconds must be non-negative",
        ),
    ],
)
def test_segment_rejects_invalid_stream_positions(
    kwargs: dict[str, object],
    message: str,
) -> None:
    from fluxtuner_ripper.models import Segment

    with pytest.raises(ValueError, match=message):
        Segment(**kwargs)
