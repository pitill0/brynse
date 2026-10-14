from __future__ import annotations

import math

import numpy as np
import pytest

from brynse.models import DecodedPcm
from brynse.pcm_features import PcmFeatureTimeline

SAMPLE_RATE = 16000


def _pcm(samples: np.ndarray) -> DecodedPcm:
    clipped = np.clip(samples, -1.0, 1.0)
    encoded = (clipped * 32767.0).astype("<i2")

    return DecodedPcm(
        sample_rate=SAMPLE_RATE,
        channels=1,
        sample_width_bytes=2,
        data=encoded.tobytes(),
    )


def _tone(
    *,
    frequency_hz: float,
    seconds: float,
    amplitude: float = 0.5,
) -> np.ndarray:
    t = np.arange(
        round(seconds * SAMPLE_RATE),
        dtype=np.float64,
    ) / SAMPLE_RATE

    return amplitude * np.sin(
        2.0 * math.pi * frequency_hz * t
    )


def _horizon(features, seconds: float):
    return next(
        item
        for item in features.horizons
        if item.horizon_seconds == seconds
    )


def test_pcm_feature_timeline_stable_tone_has_small_changes() -> None:
    samples = _tone(
        frequency_hz=440.0,
        seconds=10.0,
    )

    timeline = PcmFeatureTimeline.from_pcm(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
    )

    features = timeline.measure_boundary(
        time_seconds=5.0,
    )

    assert features is not None

    one_second = _horizon(features, 1.0)

    assert one_second.energy_change < 0.01
    assert one_second.spectral_change < 0.01


def test_pcm_feature_timeline_detects_spectral_source_change() -> None:
    samples = np.concatenate(
        (
            _tone(
                frequency_hz=220.0,
                seconds=5.0,
            ),
            _tone(
                frequency_hz=1760.0,
                seconds=5.0,
            ),
        )
    )

    timeline = PcmFeatureTimeline.from_pcm(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
    )

    features = timeline.measure_boundary(
        time_seconds=5.0,
    )

    assert features is not None

    one_second = _horizon(features, 1.0)

    assert one_second.energy_change < 0.1
    assert one_second.spectral_change > 0.5


def test_pcm_feature_timeline_detects_energy_change_without_large_spectral_change() -> None:
    samples = np.concatenate(
        (
            _tone(
                frequency_hz=440.0,
                seconds=5.0,
                amplitude=0.1,
            ),
            _tone(
                frequency_hz=440.0,
                seconds=5.0,
                amplitude=0.8,
            ),
        )
    )

    timeline = PcmFeatureTimeline.from_pcm(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
    )

    features = timeline.measure_boundary(
        time_seconds=5.0,
    )

    assert features is not None

    one_second = _horizon(features, 1.0)

    assert one_second.energy_change > 0.7
    assert one_second.spectral_change < 0.05


def test_pcm_feature_timeline_measure_many_reuses_precomputed_features() -> None:
    samples = np.concatenate(
        (
            _tone(
                frequency_hz=220.0,
                seconds=5.0,
            ),
            _tone(
                frequency_hz=1760.0,
                seconds=5.0,
            ),
        )
    )

    timeline = PcmFeatureTimeline.from_pcm(
        pcm=_pcm(samples),
        absolute_start_time_seconds=100.0,
    )

    times = tuple(
        102.0 + index * 0.01
        for index in range(600)
    )

    measured = timeline.measure_many(
        times_seconds=times,
    )

    assert len(measured) == len(times)
    assert timeline.frame_count > 0


def test_pcm_feature_timeline_requires_signed_16_bit_pcm() -> None:
    class InvalidPcm:
        sample_rate = SAMPLE_RATE
        channels = 1
        sample_width_bytes = 1
        data = b"\x00" * 100

    with pytest.raises(ValueError):
        PcmFeatureTimeline.from_pcm(
            pcm=InvalidPcm(),
            absolute_start_time_seconds=0.0,
        )


def test_pcm_feature_module_does_not_depend_on_subprocesses() -> None:
    import brynse.pcm_features as pcm_features

    source = (
        __import__("inspect")
        .getsource(pcm_features)
    )

    assert "subprocess" not in source


def test_pcm_directional_candidates_detect_falling_edge() -> None:
    samples = np.concatenate(
        (
            _tone(
                frequency_hz=440.0,
                seconds=5.0,
                amplitude=0.5,
            ),
            np.zeros(
                5 * SAMPLE_RATE,
                dtype=np.float64,
            ),
        )
    )

    timeline = PcmFeatureTimeline.from_pcm(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
    )

    candidates = timeline.directional_candidates(
        start_seconds=5.0,
        end_seconds=5.0,
    )

    assert candidates
    assert all(
        item.direction.value == "falling"
        for item in candidates
    )
    assert all(
        item.energy_ratio < 1.0
        for item in candidates
    )


def test_pcm_directional_candidates_detect_rising_edge() -> None:
    samples = np.concatenate(
        (
            np.zeros(
                5 * SAMPLE_RATE,
                dtype=np.float64,
            ),
            _tone(
                frequency_hz=440.0,
                seconds=5.0,
                amplitude=0.5,
            ),
        )
    )

    timeline = PcmFeatureTimeline.from_pcm(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
    )

    candidates = timeline.directional_candidates(
        start_seconds=5.0,
        end_seconds=5.0,
    )

    assert candidates
    assert all(
        item.direction.value == "rising"
        for item in candidates
    )
    assert all(
        item.energy_ratio > 1.0
        for item in candidates
    )


def test_pcm_directional_candidates_keep_spectral_change_separate() -> None:
    samples = np.concatenate(
        (
            _tone(
                frequency_hz=220.0,
                seconds=5.0,
                amplitude=0.5,
            ),
            _tone(
                frequency_hz=1760.0,
                seconds=5.0,
                amplitude=0.5,
            ),
        )
    )

    timeline = PcmFeatureTimeline.from_pcm(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
    )

    candidates = timeline.directional_candidates(
        start_seconds=5.0,
        end_seconds=5.0,
    )

    assert candidates

    one_second = timeline.measure_boundary(
        time_seconds=5.0,
        horizons_seconds=(0.5,),
    )

    assert one_second is not None
    horizon = one_second.horizons[0]

    assert horizon.energy_change < 0.1
    assert horizon.spectral_change > 0.5


def test_pcm_directional_candidates_do_not_select_a_winner() -> None:
    samples = _tone(
        frequency_hz=440.0,
        seconds=10.0,
    )

    timeline = PcmFeatureTimeline.from_pcm(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
    )

    candidates = timeline.directional_candidates(
        start_seconds=4.8,
        end_seconds=5.2,
        step_seconds=0.1,
    )

    assert len(candidates) == 10
    assert not hasattr(timeline, "rank")
    assert not hasattr(timeline, "winner")
