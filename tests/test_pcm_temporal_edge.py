from __future__ import annotations

import math

import numpy as np

from brynse.models import DecodedPcm
from brynse.pcm_temporal_edge import PcmTemporalEdgeVerifier

SAMPLE_RATE = 16000


def _tone(
    *,
    frequency_hz: float,
    seconds: float,
    amplitude: float,
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


def test_temporal_edge_preserves_strong_provisional_onset() -> None:
    samples = np.concatenate(
        (
            np.zeros(
                5 * SAMPLE_RATE,
                dtype=np.float64,
            ),
            _tone(
                frequency_hz=880.0,
                seconds=5.0,
                amplitude=0.5,
            ),
        )
    )

    result = PcmTemporalEdgeVerifier().verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=5.0,
    )

    assert result is not None
    assert result.verified_time_seconds == 5.0


def test_temporal_edge_can_move_late_provisional_to_earlier_onset() -> None:
    samples = np.concatenate(
        (
            np.zeros(
                round(4.0 * SAMPLE_RATE),
                dtype=np.float64,
            ),
            _tone(
                frequency_hz=880.0,
                seconds=6.0,
                amplitude=0.5,
            ),
        )
    )

    result = PcmTemporalEdgeVerifier().verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=5.5,
    )

    assert result is not None
    assert abs(result.verified_time_seconds - 4.0) <= 0.05
    assert result.delta_seconds < 0


def test_temporal_edge_can_move_early_provisional_to_later_decay() -> None:
    samples = np.concatenate(
        (
            _tone(
                frequency_hz=440.0,
                seconds=5.5,
                amplitude=0.5,
            ),
            np.zeros(
                round(4.5 * SAMPLE_RATE),
                dtype=np.float64,
            ),
        )
    )

    result = PcmTemporalEdgeVerifier().verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=4.8,
    )

    assert result is not None
    assert abs(result.verified_time_seconds - 5.5) <= 0.05
    assert result.delta_seconds > 0


def test_temporal_edge_does_not_require_rising_or_falling_direction() -> None:
    rising = np.concatenate(
        (
            np.zeros(
                5 * SAMPLE_RATE,
                dtype=np.float64,
            ),
            _tone(
                frequency_hz=880.0,
                seconds=5.0,
                amplitude=0.5,
            ),
        )
    )

    falling = np.concatenate(
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

    verifier = PcmTemporalEdgeVerifier()

    rising_result = verifier.verify(
        pcm=_pcm(rising),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=5.5,
    )
    falling_result = verifier.verify(
        pcm=_pcm(falling),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=4.5,
    )

    assert rising_result is not None
    assert falling_result is not None

    assert rising_result.evidence_250ms.direction.value == "rising"
    assert falling_result.evidence_250ms.direction.value == "falling"


def test_temporal_edge_rejects_near_silent_activity() -> None:
    samples = np.concatenate(
        (
            np.zeros(
                5 * SAMPLE_RATE,
                dtype=np.float64,
            ),
            _tone(
                frequency_hz=1000.0,
                seconds=5.0,
                amplitude=1e-5,
            ),
        )
    )

    result = PcmTemporalEdgeVerifier().verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=5.5,
    )

    assert result is None


def test_temporal_edge_abstains_without_material_change() -> None:
    samples = _tone(
        frequency_hz=440.0,
        seconds=10.0,
        amplitude=0.5,
    )

    result = PcmTemporalEdgeVerifier().verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=5.0,
    )

    assert result is None


def test_temporal_edge_explicit_bounds_can_reach_distant_edge() -> None:
    from brynse.pcm_temporal_edge import PcmTemporalSearchBounds

    samples = np.concatenate(
        (
            np.zeros(
                8 * SAMPLE_RATE,
                dtype=np.float64,
            ),
            _tone(
                frequency_hz=880.0,
                seconds=4.0,
                amplitude=0.5,
            ),
        )
    )

    result = PcmTemporalEdgeVerifier().verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=3.0,
        search_bounds=PcmTemporalSearchBounds(
            start_time_seconds=2.0,
            end_time_seconds=9.0,
        ),
    )

    assert result is not None

    # The verifier works on 250/500 ms windows, so the strongest coherent
    # boundary may precede the mathematical sample discontinuity slightly.
    # The contract under test is that explicit Specialist bounds allow PCM
    # to recover an edge far outside the legacy +/-2 s radius.
    assert abs(result.verified_time_seconds - 8.0) <= 0.25
    assert result.delta_seconds > 4.0


def test_temporal_edge_explicit_bounds_do_not_preserve_strong_provisional() -> None:
    from brynse.pcm_temporal_edge import PcmTemporalSearchBounds

    samples = np.concatenate(
        (
            _tone(
                frequency_hz=440.0,
                seconds=3.0,
                amplitude=0.08,
            ),
            _tone(
                frequency_hz=440.0,
                seconds=3.0,
                amplitude=0.5,
            ),
            _tone(
                frequency_hz=440.0,
                seconds=4.0,
                amplitude=0.02,
            ),
        )
    )

    bounded = PcmTemporalEdgeVerifier().verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=3.0,
        search_bounds=PcmTemporalSearchBounds(
            start_time_seconds=2.5,
            end_time_seconds=6.5,
        ),
    )

    assert bounded is not None
    assert abs(bounded.verified_time_seconds - 6.0) <= 0.05


def test_temporal_edge_explicit_bounds_reject_inverted_interval() -> None:
    import pytest

    from brynse.pcm_temporal_edge import PcmTemporalSearchBounds

    with pytest.raises(
        ValueError,
        match="search start must not exceed end",
    ):
        PcmTemporalSearchBounds(
            start_time_seconds=9.0,
            end_time_seconds=2.0,
        )
