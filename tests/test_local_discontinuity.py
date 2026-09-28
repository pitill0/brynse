from __future__ import annotations

from typing import Any

import pytest

from brynse.local_discontinuity import (
    LocalDiscontinuityAnalyzer,
    LocalDiscontinuityEvidence,
)
from brynse.models import DecodedPcm
from brynse.spectral import SpectralDistributionEvidence


def _pcm(*, sample_rate: int = 16000, seconds: float = 6.0) -> DecodedPcm:
    sample_count = round(sample_rate * seconds)
    return DecodedPcm(
        sample_rate=sample_rate,
        channels=1,
        sample_width_bytes=2,
        data=b"\x00\x00" * sample_count,
    )


def _spectral(*, between: float) -> SpectralDistributionEvidence:
    return SpectralDistributionEvidence(
        between=between,
        before_sequential_variability=0.0,
        after_sequential_variability=0.0,
        before_distribution_spread=0.0,
        after_distribution_spread=0.0,
    )


@pytest.mark.parametrize(
    "field",
    ["change_250ms", "change_500ms", "change_1s", "change_2s"],
)
def test_local_discontinuity_evidence_rejects_negative_values(field: str) -> None:
    values = {
        "change_250ms": 0.1,
        "change_500ms": 0.2,
        "change_1s": 0.3,
        "change_2s": 0.4,
    }
    values[field] = -0.01

    with pytest.raises(ValueError, match=field):
        LocalDiscontinuityEvidence(**values)


def test_local_discontinuity_evidence_exposes_flat_mapping() -> None:
    evidence = LocalDiscontinuityEvidence(
        change_250ms=0.1,
        change_500ms=0.2,
        change_1s=0.3,
        change_2s=0.4,
    )

    assert evidence.as_dict() == {
        "local_discontinuity_250ms": 0.1,
        "local_discontinuity_500ms": 0.2,
        "local_discontinuity_1s": 0.3,
        "local_discontinuity_2s": 0.4,
    }


def test_local_discontinuity_analyzer_uses_validated_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[float, float, float]] = []

    class FakeAnalyzer:
        def __init__(
            self,
            *,
            ffmpeg_binary: str = "ffmpeg",
            side_seconds: float = 12.0,
            guard_seconds: float = 2.0,
            block_seconds: float = 2.0,
        ) -> None:
            del ffmpeg_binary
            created.append((side_seconds, guard_seconds, block_seconds))
            self.side_seconds = side_seconds

        def analyze(self, **kwargs: Any) -> SpectralDistributionEvidence:
            del kwargs
            return _spectral(between=self.side_seconds)

    monkeypatch.setattr(
        "brynse.local_discontinuity.FfmpegSpectralDistributionAnalyzer",
        FakeAnalyzer,
    )

    analyzer = LocalDiscontinuityAnalyzer()
    result = analyzer.analyze(
        pcm=_pcm(),
        boundary_time_seconds=3.0,
    )

    assert created == [
        (0.25, 0.0, 0.25),
        (0.5, 0.0, 0.5),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 2.0),
    ]
    assert result == LocalDiscontinuityEvidence(
        change_250ms=0.25,
        change_500ms=0.5,
        change_1s=1.0,
        change_2s=2.0,
    )
    assert analyzer.required_preroll_seconds == 2.0
    assert analyzer.required_postroll_seconds == 2.0


def test_local_discontinuity_analyzer_requires_validated_sample_rate() -> None:
    analyzer = LocalDiscontinuityAnalyzer()

    with pytest.raises(ValueError, match="16000 Hz"):
        analyzer.analyze(
            pcm=_pcm(sample_rate=8000),
            boundary_time_seconds=3.0,
        )


def test_local_discontinuity_analyzer_returns_none_when_a_horizon_lacks_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAnalyzer:
        def __init__(
            self,
            *,
            ffmpeg_binary: str = "ffmpeg",
            side_seconds: float = 12.0,
            guard_seconds: float = 2.0,
            block_seconds: float = 2.0,
        ) -> None:
            del ffmpeg_binary, guard_seconds, block_seconds
            self.side_seconds = side_seconds

        def analyze(
            self,
            **kwargs: Any,
        ) -> SpectralDistributionEvidence | None:
            del kwargs
            if self.side_seconds == 1.0:
                return None
            return _spectral(between=self.side_seconds)

    monkeypatch.setattr(
        "brynse.local_discontinuity.FfmpegSpectralDistributionAnalyzer",
        FakeAnalyzer,
    )

    result = LocalDiscontinuityAnalyzer().analyze(
        pcm=_pcm(),
        boundary_time_seconds=3.0,
    )

    assert result is None
