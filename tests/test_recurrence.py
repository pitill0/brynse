from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from brynse.models import DecodedPcm
from brynse.recurrence import RecurrenceDetector, RecurrenceSettings

RATE = 8000


def _music(seconds: float, seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    out = np.zeros(n)
    base = rng.uniform(100, 400)
    for index in range(int(seconds * 4)):
        start = index * RATE // 4
        frequency = base * 2 ** (rng.integers(0, 24) / 12)
        t = np.arange(min(RATE // 4, n - start)) / RATE
        out[start : start + len(t)] += np.sin(2 * np.pi * frequency * t) * np.exp(-t * 6)
    return 0.3 * out + 0.02 * rng.standard_normal(n)


def _jingle(seconds: float = 3.2) -> npt.NDArray[np.float64]:
    t = np.arange(int(seconds * RATE)) / RATE
    noise = np.random.default_rng(99).standard_normal(len(t))
    return 0.5 * np.sin(2 * np.pi * (400 + 600 * t) * t) + 0.1 * noise


def _pcm(signal: npt.NDArray[np.float64]) -> DecodedPcm:
    noisy = signal + 0.003 * np.random.default_rng(5).standard_normal(len(signal))
    data = (np.clip(noisy, -1, 1) * 32767).astype("<i2").tobytes()
    return DecodedPcm(sample_rate=RATE, channels=1, sample_width_bytes=2, data=data)


def test_finds_every_occurrence_of_a_repeated_jingle_with_tight_edges() -> None:
    jingle = _jingle()
    parts = []
    starts = []
    position = 0
    for index in range(4):
        track = _music(90, index)
        parts.append(track)
        position += len(track)
        if index < 3:
            starts.append(position / RATE)
            parts.append(jingle)
            position += len(jingle)

    clips = RecurrenceDetector().detect(_pcm(np.concatenate(parts)))

    assert len(clips) == 1
    clip = clips[0]
    assert clip.duration_seconds == pytest.approx(len(jingle) / RATE, abs=0.05)
    assert len(clip.occurrences) == 3
    for occurrence, expected in zip(clip.occurrences, starts, strict=True):
        assert occurrence.start_seconds == pytest.approx(expected, abs=0.03)
        assert occurrence.end_seconds == pytest.approx(expected + len(jingle) / RATE, abs=0.03)
        assert occurrence.similarity > 0.95


def test_unique_audio_produces_no_recurrence() -> None:
    signal = np.concatenate([_music(90, seed) for seed in range(4)])

    assert RecurrenceDetector().detect(_pcm(signal)) == ()


def test_repeats_closer_than_min_lag_are_ignored() -> None:
    jingle = _jingle()
    signal = np.concatenate([_music(20, 1), jingle, _music(20, 2), jingle, _music(20, 3)])

    assert RecurrenceDetector().detect(_pcm(signal)) == ()
    relaxed = RecurrenceDetector(RecurrenceSettings(min_lag_seconds=10.0))
    assert len(relaxed.detect(_pcm(signal))) == 1


def test_long_repeats_are_not_reported_as_clips() -> None:
    chorus = _music(60, 42)
    signal = np.concatenate([_music(70, 1), chorus, _music(70, 2), chorus, _music(30, 3)])

    # 60 s exceeds max_duration_seconds (45 s) by default.
    assert RecurrenceDetector().detect(_pcm(signal)) == ()


def test_rejects_inconsistent_settings() -> None:
    with pytest.raises(ValueError):
        RecurrenceDetector(RecurrenceSettings(min_duration_seconds=50.0))


def test_short_input_is_handled() -> None:
    pcm = DecodedPcm(sample_rate=RATE, channels=1, sample_width_bytes=2, data=b"\x00\x01" * 100)

    assert RecurrenceDetector().detect(pcm) == ()


def test_rejects_non_mono_and_malformed_pcm() -> None:
    with pytest.raises(ValueError, match="mono"):
        DecodedPcm(sample_rate=RATE, channels=2, sample_width_bytes=2, data=b"\x00" * 128)
    with pytest.raises(ValueError, match="align"):
        DecodedPcm(sample_rate=RATE, channels=1, sample_width_bytes=2, data=b"\x00")


@pytest.mark.parametrize(
    "settings",
    [
        RecurrenceSettings(peaks_per_frame=0),
        RecurrenceSettings(fan_out=0),
        RecurrenceSettings(peak_time_radius=-1),
        RecurrenceSettings(target_zone_frames=64),
        RecurrenceSettings(edge_similarity=1.1),
        RecurrenceSettings(min_lag_seconds=-1),
    ],
)
def test_rejects_invalid_detector_settings(settings: RecurrenceSettings) -> None:
    with pytest.raises(ValueError):
        RecurrenceDetector(settings)
