from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from brynse.fade_specialist import FadeSpecialist, FadeSpecialistDiagnostics
from brynse.models import DecodedPcm, TemporalSplitDecision, TemporalSplitKind
from brynse.pcm_shared_edge import PcmSharedEdgeVerifier
from brynse.transition_specialists import TransitionSpecialistResult

SAMPLE_RATE = 16000


def _fade_guard_result(pcm: DecodedPcm) -> TransitionSpecialistResult | None:
    observation = SimpleNamespace(
        pcm16=pcm,
        window=SimpleNamespace(start_time_seconds=0.0),
        routing=SimpleNamespace(reference_time_seconds=18.0),
    )
    return FadeSpecialist._reject_false_fade_exclusion(
        observation=observation,
        result=TransitionSpecialistResult(
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=9.0,
                incoming_start_seconds=21.0,
            ),
            diagnostics=FadeSpecialistDiagnostics(None, None),
        ),
    )


def test_false_fade_exclusion_recovered_by_sustained_hard_cut() -> None:
    pcm = _pcm(
        np.concatenate((_tone(frequency_hz=220, seconds=15), _tone(frequency_hz=880, seconds=15)))
    )
    result = _fade_guard_result(pcm)
    assert result is not None
    assert result.temporal is not None
    assert result.temporal.kind is TemporalSplitKind.HARD_CUT
    assert abs(result.temporal.outgoing_end_seconds - 15.0) < 0.2


def test_fade_guard_abstains_on_silence_gap() -> None:
    pcm = _pcm(
        np.concatenate(
            (
                _tone(frequency_hz=220, seconds=12),
                np.zeros(round(3 * SAMPLE_RATE)),
                _tone(frequency_hz=880, seconds=15),
            )
        )
    )
    assert _fade_guard_result(pcm) is None


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


def test_sustained_timbre_change_can_correct_early_or_late_metadata() -> None:
    samples = np.concatenate(
        (
            _tone(frequency_hz=220.0, seconds=12.0),
            _tone(frequency_hz=880.0, seconds=12.0),
        )
    )
    verifier = PcmSharedEdgeVerifier()
    for reference in (6.0, 18.0):
        result = verifier.verify(
            pcm=_pcm(samples),
            absolute_start_time_seconds=0.0,
            provisional_time_seconds=reference,
        )
        assert result is not None
        assert abs(result.verified_time_seconds - 12.0) <= 0.1


def test_bidirectional_spectral_search_abstains_on_steady_music() -> None:
    samples = _tone(frequency_hz=440.0, seconds=24.0)
    result = PcmSharedEdgeVerifier().verify(
        pcm=_pcm(samples),
        absolute_start_time_seconds=0.0,
        provisional_time_seconds=12.0,
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
