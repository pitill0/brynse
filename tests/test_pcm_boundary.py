from __future__ import annotations

import math

import numpy as np
import pytest

from brynse.models import DecodedPcm
from brynse.pcm_boundary import PcmBoundaryVerifier

SAMPLE_RATE = 16000


def _tone(
    *,
    frequency_hz: float,
    seconds: float,
    amplitude: float = 0.5,
) -> np.ndarray:
    time = np.arange(
        round(seconds * SAMPLE_RATE),
        dtype=np.float64,
    ) / SAMPLE_RATE

    return amplitude * np.sin(
        2.0 * math.pi * frequency_hz * time
    )


def _pcm(samples: np.ndarray) -> DecodedPcm:
    encoded = (
        np.clip(samples, -1.0, 1.0)
        * 32767.0
    ).astype("<i2")

    return DecodedPcm(
        sample_rate=SAMPLE_RATE,
        channels=1,
        sample_width_bytes=2,
        data=encoded.tobytes(),
    )


def test_pcm_boundary_observe_scans_local_window() -> None:
    verifier = PcmBoundaryVerifier(
        scan_radius_seconds=0.2,
        scan_step_seconds=0.1,
    )

    observations = verifier.observe(
        pcm=_pcm(
            _tone(
                frequency_hz=440.0,
                seconds=30.0,
            )
        ),
        absolute_start_time_seconds=10.0,
        anchor_time_seconds=20.0,
    )

    assert [
        round(item.time_seconds, 3)
        for item in observations
    ] == [
        19.8,
        19.9,
        20.0,
        20.1,
        20.2,
    ]


def test_pcm_boundary_observation_exposes_multihorizon_features() -> None:
    samples = np.concatenate(
        (
            _tone(
                frequency_hz=220.0,
                seconds=10.0,
            ),
            _tone(
                frequency_hz=1760.0,
                seconds=10.0,
            ),
        )
    )

    verifier = PcmBoundaryVerifier(
        scan_radius_seconds=0.1,
        scan_step_seconds=0.1,
    )

    observations = verifier.observe(
        pcm=_pcm(samples),
        absolute_start_time_seconds=10.0,
        anchor_time_seconds=20.0,
    )

    center = next(
        item
        for item in observations
        if item.time_seconds == pytest.approx(20.0)
    )

    assert [
        horizon.horizon_seconds
        for horizon in center.features.horizons
    ] == [
        0.25,
        0.5,
        1.0,
        2.0,
    ]

    one_second = next(
        horizon
        for horizon in center.features.horizons
        if horizon.horizon_seconds == 1.0
    )

    assert one_second.spectral_change > 0.5
    assert one_second.energy_change < 0.1


def test_pcm_boundary_verifier_rejects_invalid_scan_geometry() -> None:
    with pytest.raises(ValueError):
        PcmBoundaryVerifier(
            scan_radius_seconds=0.0
        )

    with pytest.raises(ValueError):
        PcmBoundaryVerifier(
            scan_step_seconds=0.0
        )


def test_pcm_boundary_observe_many_deduplicates_overlapping_windows() -> None:
    verifier = PcmBoundaryVerifier(
        scan_radius_seconds=0.2,
        scan_step_seconds=0.1,
    )

    observations = verifier.observe_many(
        pcm=_pcm(
            _tone(
                frequency_hz=440.0,
                seconds=30.0,
            )
        ),
        absolute_start_time_seconds=10.0,
        anchor_times_seconds=(20.0, 20.1),
    )

    assert [
        round(item.time_seconds, 3)
        for item in observations
    ] == [
        19.8,
        19.9,
        20.0,
        20.1,
        20.2,
        20.3,
    ]


def test_pcm_boundary_observer_has_no_ranking_policy() -> None:
    verifier = PcmBoundaryVerifier()

    assert not hasattr(verifier, "rank")
