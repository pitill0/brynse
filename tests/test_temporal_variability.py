from __future__ import annotations

from array import array

import pytest

from brynse.models import DecodedPcm
from brynse.temporal_variability import (
    TemporalVariabilityAnalyzer,
    TemporalVariabilityEvidence,
)


def _pcm(seconds: float, *, sample_rate: int = 16000) -> DecodedPcm:
    sample_count = round(seconds * sample_rate)
    samples = array(
        "h",
        (
            int(9000 * __import__("math").sin(2.0 * __import__("math").pi * 440 * i / sample_rate))
            for i in range(sample_count)
        ),
    )
    return DecodedPcm(
        sample_rate=sample_rate,
        channels=1,
        sample_width_bytes=2,
        data=samples.tobytes(),
    )


def test_temporal_variability_analyzer_reports_validated_context() -> None:
    analyzer = TemporalVariabilityAnalyzer()

    assert analyzer.required_preroll_seconds == 42.0
    assert analyzer.required_postroll_seconds == 42.0
    assert analyzer.required_sample_rate == 16000


def test_temporal_variability_analyzer_rejects_wrong_sample_rate() -> None:
    with pytest.raises(ValueError, match="16000 Hz"):
        TemporalVariabilityAnalyzer().analyze(
            pcm=_pcm(90.0, sample_rate=8000),
            boundary_time_seconds=45.0,
        )


def test_temporal_variability_analyzer_returns_none_without_complete_context() -> None:
    pytest.importorskip("numpy")
    analyzer = TemporalVariabilityAnalyzer()

    assert (
        analyzer.analyze(
            pcm=_pcm(90.0),
            boundary_time_seconds=40.0,
        )
        is None
    )


def test_temporal_variability_analyzer_produces_five_bin_multihorizon_evidence() -> None:
    pytest.importorskip("numpy")
    analyzer = TemporalVariabilityAnalyzer()
    evidence = analyzer.analyze(
        pcm=_pcm(90.0),
        boundary_time_seconds=45.0,
    )

    assert isinstance(evidence, TemporalVariabilityEvidence)
    assert len(evidence.pre_profile_pairwise) == 5
    assert len(evidence.post_profile_pairwise) == 5
    assert evidence.variability_ratio_8s >= 0.0
    assert evidence.variability_ratio_16s >= 0.0
    assert evidence.variability_ratio_24s >= 0.0
    assert evidence.variability_ratio_40s >= 0.0


def test_temporal_variability_contraction_ratio_matches_b7_3e_definition() -> None:
    analyzer = TemporalVariabilityAnalyzer()
    pre = (1.0, 2.0, 3.0, 4.0, 5.0)
    post = (2.0, 4.0, 6.0, 8.0, 10.0)

    assert analyzer._contraction_ratio(pre, post, 1) == pytest.approx(2.0 / 5.0)
    assert analyzer._contraction_ratio(pre, post, 2) == pytest.approx(3.0 / 4.5)
    assert analyzer._contraction_ratio(pre, post, 3) == pytest.approx(4.0 / 4.0)
    assert analyzer._contraction_ratio(pre, post, 5) == pytest.approx(6.0 / 3.0)
