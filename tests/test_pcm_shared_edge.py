from __future__ import annotations

import math

import numpy as np

from brynse.models import DecodedPcm
from brynse.pcm_shared_edge import PcmSharedEdgeVerifier

SAMPLE_RATE = 16000


def _tone(
    *,
    frequency_hz: float,
    seconds: float,
    amplitude: float = 0.5,
) -> np.ndarray:
    time = (
        np.arange(
            round(seconds * SAMPLE_RATE),
            dtype=np.float64,
        )
        / SAMPLE_RATE
    )

    return amplitude * np.sin(2.0 * math.pi * frequency_hz * time)


def _pcm(samples: np.ndarray) -> DecodedPcm:
    encoded = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")

    return DecodedPcm(
        sample_rate=SAMPLE_RATE,
        channels=1,
        sample_width_bytes=2,
        data=encoded.tobytes(),
    )


def test_shared_edge_retracts_late_boundary_to_rising_onset() -> None:
    samples = np.concatenate(
        (
            _tone(
                frequency_hz=220.0,
                seconds=4.8,
                amplitude=0.02,
            ),
            _tone(
                frequency_hz=880.0,
                seconds=5.2,
                amplitude=0.5,
            ),
        )
    )

    result = PcmSharedEdgeVerifier(
        search_before_seconds=0.5,
        step_seconds=0.025,
    ).verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=5.0,
    )

    assert result is not None
    assert abs(result.verified_time_seconds - 4.8) <= 0.05
    assert result.delta_seconds < 0


def test_shared_edge_abstains_without_rising_event() -> None:
    samples = _tone(
        frequency_hz=440.0,
        seconds=10.0,
        amplitude=0.5,
    )

    result = PcmSharedEdgeVerifier().verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=5.0,
    )

    assert result is None


def test_shared_edge_rejects_near_silent_rising_noise() -> None:
    samples = np.concatenate(
        (
            np.zeros(
                round(4.8 * SAMPLE_RATE),
                dtype=np.float64,
            ),
            _tone(
                frequency_hz=1000.0,
                seconds=5.2,
                amplitude=1e-5,
            ),
        )
    )

    result = PcmSharedEdgeVerifier().verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=5.0,
    )

    assert result is None


def test_shared_edge_does_not_move_boundary_forward() -> None:
    samples = np.concatenate(
        (
            _tone(
                frequency_hz=220.0,
                seconds=5.2,
                amplitude=0.02,
            ),
            _tone(
                frequency_hz=880.0,
                seconds=4.8,
                amplitude=0.5,
            ),
        )
    )

    result = PcmSharedEdgeVerifier(
        search_before_seconds=0.5,
    ).verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=5.0,
    )

    assert result is None


def test_shared_edge_preserves_pcm_evidence() -> None:
    samples = np.concatenate(
        (
            _tone(
                frequency_hz=220.0,
                seconds=4.8,
                amplitude=0.02,
            ),
            _tone(
                frequency_hz=880.0,
                seconds=5.2,
                amplitude=0.5,
            ),
        )
    )

    result = PcmSharedEdgeVerifier().verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=5.0,
    )

    assert result is not None
    assert result.evidence_250ms.horizon_seconds == 0.25
    assert result.evidence_500ms.horizon_seconds == 0.5
    assert result.evidence_250ms.energy_change >= 0.25
    assert result.evidence_500ms.energy_change >= 0.25
