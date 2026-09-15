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


def test_manual_boundary_provider_returns_boundaries_in_requested_window() -> None:
    from fluxtuner_ripper.providers import ManualBoundaryProvider

    provider = ManualBoundaryProvider(
        boundary_times_seconds=(30.0, 10.0, 20.0),
    )

    candidates = provider.propose(
        start_time_seconds=10.0,
        end_time_seconds=30.0,
    )

    assert [candidate.time_seconds for candidate in candidates] == [
        20.0,
        30.0,
    ]
    assert [candidate.source for candidate in candidates] == [
        "manual",
        "manual",
    ]


def test_manual_boundary_provider_deduplicates_times() -> None:
    from fluxtuner_ripper.providers import ManualBoundaryProvider

    provider = ManualBoundaryProvider(
        boundary_times_seconds=(10.0, 10.0, 20.0),
    )

    candidates = provider.propose(
        start_time_seconds=0.0,
        end_time_seconds=30.0,
    )

    assert [candidate.time_seconds for candidate in candidates] == [
        10.0,
        20.0,
    ]


def test_manual_boundary_provider_rejects_negative_time() -> None:
    import pytest

    from fluxtuner_ripper.providers import ManualBoundaryProvider

    with pytest.raises(
        ValueError,
        match="boundary times must be non-negative",
    ):
        ManualBoundaryProvider(
            boundary_times_seconds=(-1.0,),
        )


def test_external_boundary_provider_preserves_external_identity() -> None:
    from fluxtuner_ripper.providers import (
        ExternalBoundary,
        ExternalBoundaryProvider,
    )

    provider = ExternalBoundaryProvider(
        boundaries=(
            ExternalBoundary(
                time_seconds=30.0,
                source="agent",
                reference_offset=1234,
            ),
            ExternalBoundary(
                time_seconds=10.0,
                source="semantic_model",
            ),
        )
    )

    candidates = provider.propose(
        start_time_seconds=0.0,
        end_time_seconds=40.0,
    )

    assert [candidate.time_seconds for candidate in candidates] == [
        10.0,
        30.0,
    ]
    assert [candidate.source for candidate in candidates] == [
        "semantic_model",
        "agent",
    ]
    assert candidates[0].reference_offset is None
    assert candidates[1].reference_offset == 1234


def test_external_boundary_provider_filters_requested_window() -> None:
    from fluxtuner_ripper.providers import (
        ExternalBoundary,
        ExternalBoundaryProvider,
    )

    provider = ExternalBoundaryProvider(
        boundaries=(
            ExternalBoundary(time_seconds=10.0),
            ExternalBoundary(time_seconds=20.0),
            ExternalBoundary(time_seconds=30.0),
        )
    )

    candidates = provider.propose(
        start_time_seconds=10.0,
        end_time_seconds=20.0,
    )

    assert [candidate.time_seconds for candidate in candidates] == [20.0]


def test_external_boundary_rejects_invalid_values() -> None:
    import pytest

    from fluxtuner_ripper.providers import ExternalBoundary

    with pytest.raises(ValueError, match="time_seconds must be non-negative"):
        ExternalBoundary(time_seconds=-1.0)

    with pytest.raises(ValueError, match="source must not be empty"):
        ExternalBoundary(
            time_seconds=1.0,
            source=" ",
        )

    with pytest.raises(ValueError, match="reference_offset must be non-negative"):
        ExternalBoundary(
            time_seconds=1.0,
            reference_offset=-1,
        )
