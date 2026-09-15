import pytest

from fluxtuner_ripper.models import BoundaryCandidate
from fluxtuner_ripper.providers import FixedIntervalBoundaryProvider


def test_fixed_interval_provider_proposes_boundaries_in_requested_interval() -> None:
    provider = FixedIntervalBoundaryProvider(interval_seconds=10.0)

    assert provider.propose(
        start_time_seconds=0.0,
        end_time_seconds=35.0,
    ) == (
        BoundaryCandidate(time_seconds=10.0, source="fixed_interval"),
        BoundaryCandidate(time_seconds=20.0, source="fixed_interval"),
        BoundaryCandidate(time_seconds=30.0, source="fixed_interval"),
    )


def test_fixed_interval_provider_uses_open_start_and_closed_end() -> None:
    provider = FixedIntervalBoundaryProvider(interval_seconds=10.0)

    assert provider.propose(
        start_time_seconds=10.0,
        end_time_seconds=20.0,
    ) == (BoundaryCandidate(time_seconds=20.0, source="fixed_interval"),)


def test_fixed_interval_provider_can_return_no_candidates() -> None:
    provider = FixedIntervalBoundaryProvider(interval_seconds=10.0)

    assert (
        provider.propose(
            start_time_seconds=1.0,
            end_time_seconds=9.0,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"interval_seconds": 0.0},
            "interval_seconds must be greater than zero",
        ),
        (
            {"interval_seconds": -1.0},
            "interval_seconds must be greater than zero",
        ),
    ],
)
def test_fixed_interval_provider_rejects_invalid_interval(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FixedIntervalBoundaryProvider(**kwargs)


def test_fixed_interval_provider_rejects_invalid_requested_range() -> None:
    provider = FixedIntervalBoundaryProvider(interval_seconds=10.0)

    with pytest.raises(ValueError, match="start_time_seconds must be non-negative"):
        provider.propose(
            start_time_seconds=-1.0,
            end_time_seconds=10.0,
        )

    with pytest.raises(
        ValueError,
        match="end_time_seconds must be greater than or equal to start_time_seconds",
    ):
        provider.propose(
            start_time_seconds=20.0,
            end_time_seconds=10.0,
        )
