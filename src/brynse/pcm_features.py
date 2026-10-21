"""Reusable in-memory PCM feature timeline for temporal boundary analysis.

The timeline performs one feature extraction pass over decoded PCM and then
supports cheap PRE/POST measurements at arbitrary boundary timestamps.

It deliberately knows nothing about transition ownership, metadata, Router
semantics, or temporal split policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from brynse.models import DecodedPcm

_DEFAULT_FRAME_SECONDS = 0.050
_DEFAULT_HOP_SECONDS = 0.025
_DEFAULT_BAND_EDGES_HZ = (
    40.0,
    120.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    6000.0,
    8000.0,
)


class PcmEnergyDirection(Enum):
    """Observed direction of local PCM energy across a boundary."""

    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"


@dataclass(frozen=True)
class PcmDirectionalBoundaryEvidence:
    """Descriptive directional PCM evidence at one timestamp."""

    time_seconds: float
    horizon_seconds: float
    pre_rms: float
    post_rms: float
    energy_change: float
    spectral_change: float
    direction: PcmEnergyDirection
    energy_ratio: float


@dataclass(frozen=True)
class PcmBoundaryHorizonEvidence:
    """Descriptive PRE/POST PCM evidence for one temporal horizon."""

    horizon_seconds: float
    pre_rms: float
    post_rms: float
    energy_change: float
    spectral_change: float

    def __post_init__(self) -> None:
        if self.horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be greater than zero")

        for name, value in (
            ("pre_rms", self.pre_rms),
            ("post_rms", self.post_rms),
            ("energy_change", self.energy_change),
            ("spectral_change", self.spectral_change),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.pre_rms < 0 or self.post_rms < 0:
            raise ValueError("RMS values must be non-negative")

        if not 0.0 <= self.energy_change <= 1.0:
            raise ValueError("energy_change must be between zero and one")

        if not 0.0 <= self.spectral_change <= 1.0:
            raise ValueError("spectral_change must be between zero and one")


@dataclass(frozen=True)
class PcmBoundaryFeatures:
    """Multi-horizon descriptive evidence at one possible boundary."""

    time_seconds: float
    horizons: tuple[PcmBoundaryHorizonEvidence, ...]

    def __post_init__(self) -> None:
        if self.time_seconds < 0:
            raise ValueError("time_seconds must be non-negative")


class PcmFeatureTimeline:
    """Precompute compact PCM features once and query them in memory."""

    def __init__(
        self,
        *,
        absolute_start_time_seconds: float,
        frame_centers_seconds: np.ndarray,
        rms_prefix: np.ndarray,
        spectral_prefix: np.ndarray,
    ) -> None:
        if absolute_start_time_seconds < 0:
            raise ValueError("absolute_start_time_seconds must be non-negative")

        self._absolute_start_time_seconds = absolute_start_time_seconds
        self._frame_centers_seconds = frame_centers_seconds
        self._rms_prefix = rms_prefix
        self._spectral_prefix = spectral_prefix

    @classmethod
    def from_pcm(
        cls,
        *,
        pcm: DecodedPcm,
        absolute_start_time_seconds: float,
        frame_seconds: float = _DEFAULT_FRAME_SECONDS,
        hop_seconds: float = _DEFAULT_HOP_SECONDS,
    ) -> PcmFeatureTimeline:
        """Build one reusable feature timeline from mono signed-16-bit PCM."""

        if absolute_start_time_seconds < 0:
            raise ValueError("absolute_start_time_seconds must be non-negative")

        if pcm.sample_rate <= 0:
            raise ValueError("PCM sample rate must be greater than zero")

        if pcm.channels != 1:
            raise ValueError("PCM feature timeline requires mono PCM")

        if pcm.sample_width_bytes != 2:
            raise ValueError("PCM feature timeline requires signed 16-bit PCM")

        if frame_seconds <= 0:
            raise ValueError("frame_seconds must be greater than zero")

        if hop_seconds <= 0:
            raise ValueError("hop_seconds must be greater than zero")

        frame_samples = round(frame_seconds * pcm.sample_rate)
        hop_samples = round(hop_seconds * pcm.sample_rate)

        if frame_samples < 2:
            raise ValueError("frame duration produces too few samples")

        if hop_samples < 1:
            raise ValueError("hop duration produces too few samples")

        raw = np.frombuffer(pcm.data, dtype="<i2").astype(
            np.float64,
            copy=False,
        )
        samples = raw / 32768.0

        if samples.size < frame_samples:
            empty_centers = np.empty(0, dtype=np.float64)
            empty_rms_prefix = np.zeros(1, dtype=np.float64)
            empty_spectral_prefix = np.zeros(
                (1, len(_DEFAULT_BAND_EDGES_HZ) - 1),
                dtype=np.float64,
            )
            return cls(
                absolute_start_time_seconds=absolute_start_time_seconds,
                frame_centers_seconds=empty_centers,
                rms_prefix=empty_rms_prefix,
                spectral_prefix=empty_spectral_prefix,
            )

        frame_count = 1 + (samples.size - frame_samples) // hop_samples

        shape = (frame_count, frame_samples)
        strides = (
            samples.strides[0] * hop_samples,
            samples.strides[0],
        )

        frames = np.lib.stride_tricks.as_strided(
            samples,
            shape=shape,
            strides=strides,
            writeable=False,
        )

        frame_power = np.mean(frames * frames, axis=1)
        frame_rms = np.sqrt(frame_power)

        window = np.hanning(frame_samples)
        spectra = np.fft.rfft(frames * window, axis=1)
        power = np.abs(spectra) ** 2
        frequencies = np.fft.rfftfreq(
            frame_samples,
            d=1.0 / pcm.sample_rate,
        )

        band_edges = tuple(edge for edge in _DEFAULT_BAND_EDGES_HZ if edge < pcm.sample_rate / 2.0)

        nyquist = pcm.sample_rate / 2.0

        if not band_edges:
            band_edges = (0.0, nyquist)
        elif band_edges[0] > 0.0:
            band_edges = (0.0, *band_edges)

        if band_edges[-1] < nyquist:
            band_edges = (*band_edges, nyquist)

        band_count = len(band_edges) - 1
        band_power = np.zeros(
            (frame_count, band_count),
            dtype=np.float64,
        )

        for index in range(band_count):
            left = band_edges[index]
            right = band_edges[index + 1]

            if index == band_count - 1:
                mask = (frequencies >= left) & (frequencies <= right)
            else:
                mask = (frequencies >= left) & (frequencies < right)

            if np.any(mask):
                band_power[:, index] = np.sum(
                    power[:, mask],
                    axis=1,
                )

        totals = np.sum(band_power, axis=1, keepdims=True)

        spectral_distribution = np.divide(
            band_power,
            totals,
            out=np.zeros_like(band_power),
            where=totals > 1e-18,
        )

        frame_starts = np.arange(frame_count, dtype=np.float64) * hop_samples / pcm.sample_rate

        frame_centers = (
            absolute_start_time_seconds + frame_starts + frame_samples / pcm.sample_rate / 2.0
        )

        rms_prefix = np.concatenate(
            (
                np.zeros(1, dtype=np.float64),
                np.cumsum(frame_rms, dtype=np.float64),
            )
        )

        spectral_prefix = np.vstack(
            (
                np.zeros(
                    (1, spectral_distribution.shape[1]),
                    dtype=np.float64,
                ),
                np.cumsum(
                    spectral_distribution,
                    axis=0,
                    dtype=np.float64,
                ),
            )
        )

        return cls(
            absolute_start_time_seconds=absolute_start_time_seconds,
            frame_centers_seconds=frame_centers,
            rms_prefix=rms_prefix,
            spectral_prefix=spectral_prefix,
        )

    @property
    def frame_count(self) -> int:
        """Return the number of precomputed PCM frames."""

        return int(self._frame_centers_seconds.size)

    def measure_boundary(
        self,
        *,
        time_seconds: float,
        horizons_seconds: tuple[float, ...] = (
            0.25,
            0.5,
            1.0,
            2.0,
        ),
    ) -> PcmBoundaryFeatures | None:
        """Measure descriptive PRE/POST changes around one timestamp."""

        if time_seconds < self._absolute_start_time_seconds:
            raise ValueError("time_seconds precedes PCM feature timeline")

        evidence: list[PcmBoundaryHorizonEvidence] = []

        for horizon_seconds in horizons_seconds:
            measured = self._measure_horizon(
                time_seconds=time_seconds,
                horizon_seconds=horizon_seconds,
            )

            if measured is None:
                return None

            evidence.append(measured)

        return PcmBoundaryFeatures(
            time_seconds=time_seconds,
            horizons=tuple(evidence),
        )

    def measure_many(
        self,
        *,
        times_seconds: tuple[float, ...],
        horizons_seconds: tuple[float, ...] = (
            0.25,
            0.5,
            1.0,
            2.0,
        ),
    ) -> tuple[PcmBoundaryFeatures, ...]:
        """Measure many timestamps without repeating feature extraction."""

        measured: list[PcmBoundaryFeatures] = []

        for time_seconds in times_seconds:
            item = self.measure_boundary(
                time_seconds=time_seconds,
                horizons_seconds=horizons_seconds,
            )

            if item is not None:
                measured.append(item)

        return tuple(measured)

    def directional_candidates(
        self,
        *,
        start_seconds: float,
        end_seconds: float,
        step_seconds: float = 0.05,
        horizons_seconds: tuple[float, ...] = (
            0.25,
            0.5,
        ),
    ) -> tuple[PcmDirectionalBoundaryEvidence, ...]:
        """Return descriptive directional evidence across one local interval."""

        if end_seconds < start_seconds:
            raise ValueError("end_seconds must not precede start_seconds")

        if step_seconds <= 0:
            raise ValueError("step_seconds must be greater than zero")

        evidence: list[PcmDirectionalBoundaryEvidence] = []

        index = 0

        while True:
            time_seconds = start_seconds + index * step_seconds

            if time_seconds > end_seconds + 1e-9:
                break

            measured = self.measure_boundary(
                time_seconds=time_seconds,
                horizons_seconds=horizons_seconds,
            )

            if measured is not None:
                for horizon in measured.horizons:
                    epsilon = 1e-12
                    delta = horizon.post_rms - horizon.pre_rms

                    if abs(delta) <= epsilon:
                        direction = PcmEnergyDirection.STABLE
                    elif delta > 0:
                        direction = PcmEnergyDirection.RISING
                    else:
                        direction = PcmEnergyDirection.FALLING

                    evidence.append(
                        PcmDirectionalBoundaryEvidence(
                            time_seconds=time_seconds,
                            horizon_seconds=horizon.horizon_seconds,
                            pre_rms=horizon.pre_rms,
                            post_rms=horizon.post_rms,
                            energy_change=horizon.energy_change,
                            spectral_change=horizon.spectral_change,
                            direction=direction,
                            energy_ratio=(horizon.post_rms + epsilon) / (horizon.pre_rms + epsilon),
                        )
                    )

            index += 1

        return tuple(evidence)

    def _measure_horizon(
        self,
        *,
        time_seconds: float,
        horizon_seconds: float,
    ) -> PcmBoundaryHorizonEvidence | None:
        if horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be greater than zero")

        pre_start = time_seconds - horizon_seconds
        pre_end = time_seconds
        post_start = time_seconds
        post_end = time_seconds + horizon_seconds

        pre = self._mean_features(
            start_seconds=pre_start,
            end_seconds=pre_end,
        )
        post = self._mean_features(
            start_seconds=post_start,
            end_seconds=post_end,
        )

        if pre is None or post is None:
            return None

        pre_rms, pre_spectral = pre
        post_rms, post_spectral = post

        energy_change = abs(pre_rms - post_rms) / max(
            pre_rms + post_rms,
            1e-12,
        )

        spectral_change = self._jensen_shannon_divergence(
            pre_spectral,
            post_spectral,
        )

        return PcmBoundaryHorizonEvidence(
            horizon_seconds=horizon_seconds,
            pre_rms=pre_rms,
            post_rms=post_rms,
            energy_change=float(min(max(energy_change, 0.0), 1.0)),
            spectral_change=float(min(max(spectral_change, 0.0), 1.0)),
        )

    def _mean_features(
        self,
        *,
        start_seconds: float,
        end_seconds: float,
    ) -> tuple[float, np.ndarray] | None:
        if end_seconds <= start_seconds:
            return None

        left = int(
            np.searchsorted(
                self._frame_centers_seconds,
                start_seconds,
                side="left",
            )
        )
        right = int(
            np.searchsorted(
                self._frame_centers_seconds,
                end_seconds,
                side="left",
            )
        )

        if right <= left:
            return None

        rms_sum = self._rms_prefix[right] - self._rms_prefix[left]
        spectral_sum = self._spectral_prefix[right] - self._spectral_prefix[left]

        count = right - left

        return (
            float(rms_sum / count),
            spectral_sum / count,
        )

    @staticmethod
    def _jensen_shannon_divergence(
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        """Return bounded base-2 Jensen-Shannon divergence in [0, 1]."""

        left_total = float(np.sum(left))
        right_total = float(np.sum(right))

        if left_total <= 1e-18 and right_total <= 1e-18:
            return 0.0

        if left_total <= 1e-18 or right_total <= 1e-18:
            return 1.0

        p = left / left_total
        q = right / right_total
        midpoint = 0.5 * (p + q)

        def kl_divergence(
            source: np.ndarray,
            target: np.ndarray,
        ) -> float:
            mask = source > 0.0
            return float(
                np.sum(source[mask] * np.log2(source[mask] / np.maximum(target[mask], 1e-18)))
            )

        value = 0.5 * (kl_divergence(p, midpoint) + kl_divergence(q, midpoint))

        return min(max(value, 0.0), 1.0)
