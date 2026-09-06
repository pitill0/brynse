from __future__ import annotations

import pytest

from fluxtuner_ripper.models import MetadataSemanticDecision, SplitKind
from fluxtuner_ripper.transient import ConservativeTransientExclusionPolicy


def _decision(
    title: str,
    *,
    lifetime: float = 2.0,
    kind: SplitKind = SplitKind.NO_BOUNDARY,
) -> MetadataSemanticDecision:
    return MetadataSemanticDecision(
        title=title,
        kind=kind,
        start_offset=1000,
        start_time_seconds=10.0,
        lifetime_seconds=lifetime,
    )


@pytest.mark.parametrize(
    "title",
    [
        "Kurze Werbepause - Wirklich kurz",
        "Advertisement",
        "AD BREAK",
        "Station ID",
        "Jingle",
        "Intervalo publicitario",
        "Pubblicità",
    ],
)
def test_conservative_transient_policy_accepts_explicit_non_track_markers(
    title: str,
) -> None:
    assert ConservativeTransientExclusionPolicy()(_decision(title))


@pytest.mark.parametrize(
    "title",
    [
        "Commercial-free",
        "Ad Free Radio",
        "Werbefrei",
        "Ohne Werbung",
        "Sin publicidad",
        "Artist - Song",
    ],
)
def test_conservative_transient_policy_rejects_negative_or_ordinary_titles(
    title: str,
) -> None:
    assert not ConservativeTransientExclusionPolicy()(_decision(title))


def test_conservative_transient_policy_rejects_long_interval() -> None:
    assert not ConservativeTransientExclusionPolicy()(_decision("Advertisement", lifetime=9.0))


def test_conservative_transient_policy_rejects_non_transient_decision() -> None:
    assert not ConservativeTransientExclusionPolicy()(
        _decision("Advertisement", kind=SplitKind.HARD_CUT)
    )


def test_conservative_transient_policy_rejects_invalid_max_duration() -> None:
    with pytest.raises(ValueError, match="max_duration_seconds"):
        ConservativeTransientExclusionPolicy(max_duration_seconds=0.0)
