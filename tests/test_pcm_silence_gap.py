"""F-1/F-5: establish both silence boundaries from PCM, not ICY time."""

import numpy as np

from brynse.models import DecodedPcm, TemporalSplitKind
from brynse.pcm_silence_gap import PcmSilenceGapVerifier


def _pcm(samples: np.ndarray) -> DecodedPcm:
    return DecodedPcm(
        sample_rate=16000,
        channels=1,
        sample_width_bytes=2,
        data=(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes(),
    )


def _tone(seconds: float, frequency: float = 220.0) -> np.ndarray:
    t = np.arange(round(seconds * 16000)) / 16000
    return 0.2 * np.sin(2 * np.pi * frequency * t)


def test_silence_gap_verified_with_early_and_late_icy() -> None:
    pcm = _pcm(np.concatenate((_tone(9.0), np.zeros(24000), _tone(9.0, 550.0))))
    for icy in (4.5, 10.5, 16.5):
        verified = PcmSilenceGapVerifier().verify(
            pcm=pcm,
            absolute_start_time_seconds=0.0,
            reference_time_seconds=icy,
        )
        assert verified is not None
        assert verified.kind is TemporalSplitKind.EXCLUSION
        assert abs(verified.outgoing_end_seconds - 9.0) < 0.03
        assert abs(verified.incoming_start_seconds - 10.5) < 0.03


def test_silence_verifier_abstains_without_both_audible_flanks() -> None:
    pcm = _pcm(np.concatenate((np.zeros(24000), _tone(9.0))))
    assert (
        PcmSilenceGapVerifier().verify(
            pcm=pcm,
            absolute_start_time_seconds=0.0,
            reference_time_seconds=2.0,
        )
        is None
    )


def test_silence_verifier_abstains_on_short_quiet_dip() -> None:
    pcm = _pcm(np.concatenate((_tone(4.0), np.zeros(round(0.3 * 16000)), _tone(4.0))))
    assert (
        PcmSilenceGapVerifier().verify(
            pcm=pcm,
            absolute_start_time_seconds=0.0,
            reference_time_seconds=4.0,
        )
        is None
    )


def test_silence_verifier_does_not_label_a_continuous_crossfade_as_a_gap() -> None:
    """Mixed audio is not an excluded silence, even with a metadata change."""
    overlap_seconds = 6.0
    overlap_samples = round(overlap_seconds * 16000)
    first = _tone(8.0, 220.0)
    second = _tone(8.0, 660.0)
    ramp = np.linspace(0.0, 1.0, overlap_samples)
    mixture = first[:overlap_samples] * (1.0 - ramp) + second[:overlap_samples] * ramp
    pcm = _pcm(np.concatenate((first, mixture, second)))

    for reference in (8.0, 11.0, 14.0):
        assert (
            PcmSilenceGapVerifier().verify(
                pcm=pcm,
                absolute_start_time_seconds=0.0,
                reference_time_seconds=reference,
            )
            is None
        )
