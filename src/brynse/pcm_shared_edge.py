"""PCM verification of one provisional shared temporal edge.

Router and specialist geometry decide that a transition owns one shared edge.
This module only verifies the exact local timestamp against decoded PCM.

It does not infer transition kind or ownership.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from brynse.models import DecodedPcm
from brynse.pcm_features import (
    PcmDirectionalBoundaryEvidence,
    PcmEnergyDirection,
    PcmFeatureTimeline,
)


@dataclass(frozen=True)
class PcmVerifiedSharedEdge:
    """PCM-verified replacement for one provisional shared edge."""

    provisional_time_seconds: float
    verified_time_seconds: float
    evidence_250ms: PcmDirectionalBoundaryEvidence
    evidence_500ms: PcmDirectionalBoundaryEvidence

    @property
    def delta_seconds(self) -> float:
        return self.verified_time_seconds - self.provisional_time_seconds


class PcmSharedEdgeVerifier:
    """Verify provisional shared edges against local PCM.

    A unique sustained spectral change can correct a reference in either
    direction. The original conservative rising-edge rule remains available
    for short windows where the stronger evidence is unavailable.
    Neither rule infers transition kind or ownership.
    """

    def __init__(
        self,
        *,
        search_before_seconds: float = 0.5,
        step_seconds: float = 0.025,
        minimum_energy_change: float = 0.25,
        minimum_post_rms: float = 1e-4,
    ) -> None:
        if search_before_seconds <= 0:
            raise ValueError("search_before_seconds must be greater than zero")

        if step_seconds <= 0:
            raise ValueError("step_seconds must be greater than zero")

        if not 0.0 <= minimum_energy_change <= 1.0:
            raise ValueError("minimum_energy_change must be between zero and one")

        if minimum_post_rms < 0:
            raise ValueError("minimum_post_rms must be non-negative")

        self._search_before_seconds = search_before_seconds
        self._step_seconds = step_seconds
        self._minimum_energy_change = minimum_energy_change
        self._minimum_post_rms = minimum_post_rms

    def verify(
        self,
        *,
        pcm: DecodedPcm,
        absolute_start_time_seconds: float,
        provisional_time_seconds: float,
    ) -> PcmVerifiedSharedEdge | None:
        """Return a supported PCM edge, or abstain."""

        if provisional_time_seconds < absolute_start_time_seconds:
            raise ValueError("provisional edge precedes decoded PCM")

        timeline = PcmFeatureTimeline.from_pcm(
            pcm=pcm,
            absolute_start_time_seconds=absolute_start_time_seconds,
        )

        spectral_time = self._spectral_handoff(
            pcm=pcm,
            start_time=absolute_start_time_seconds,
            reference_time=provisional_time_seconds,
        )
        if spectral_time is not None:
            observations = timeline.directional_candidates(
                start_seconds=spectral_time,
                end_seconds=spectral_time,
                horizons_seconds=(0.25, 0.5),
            )
            if len(observations) == 2:
                return PcmVerifiedSharedEdge(
                    provisional_time_seconds=provisional_time_seconds,
                    verified_time_seconds=spectral_time,
                    evidence_250ms=observations[0],
                    evidence_500ms=observations[1],
                )

        start_seconds = max(
            absolute_start_time_seconds,
            provisional_time_seconds - self._search_before_seconds,
        )

        candidates = timeline.directional_candidates(
            start_seconds=start_seconds,
            end_seconds=provisional_time_seconds,
            step_seconds=self._step_seconds,
            horizons_seconds=(0.25, 0.5),
        )

        by_time: dict[
            float,
            dict[float, PcmDirectionalBoundaryEvidence],
        ] = {}

        for candidate in candidates:
            key = round(candidate.time_seconds, 9)
            by_time.setdefault(key, {})[candidate.horizon_seconds] = candidate

        viable: list[
            tuple[
                float,
                float,
                float,
                PcmDirectionalBoundaryEvidence,
                PcmDirectionalBoundaryEvidence,
            ]
        ] = []

        for key, horizons in by_time.items():
            short = horizons.get(0.25)
            medium = horizons.get(0.5)

            if short is None or medium is None:
                continue

            if (
                short.direction is not PcmEnergyDirection.RISING
                or medium.direction is not PcmEnergyDirection.RISING
            ):
                continue

            if (
                short.energy_change < self._minimum_energy_change
                or medium.energy_change < self._minimum_energy_change
            ):
                continue

            if short.post_rms < self._minimum_post_rms or medium.post_rms < self._minimum_post_rms:
                continue

            energy_support = min(
                short.energy_change,
                medium.energy_change,
            )
            spectral_support = min(
                short.spectral_change,
                medium.spectral_change,
            )

            viable.append(
                (
                    energy_support,
                    spectral_support,
                    key,
                    short,
                    medium,
                )
            )

        if not viable:
            return None

        (
            _energy_support,
            _spectral_support,
            verified_time,
            short,
            medium,
        ) = max(
            viable,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            ),
        )

        # The current policy proves only retraction.  The provisional
        # timestamp itself is not a refinement.
        if verified_time >= provisional_time_seconds - 1e-9:
            return None

        return PcmVerifiedSharedEdge(
            provisional_time_seconds=provisional_time_seconds,
            verified_time_seconds=verified_time,
            evidence_250ms=short,
            evidence_500ms=medium,
        )

    @staticmethod
    def _spectral_handoff(
        *, pcm: DecodedPcm, start_time: float, reference_time: float
    ) -> float | None:
        """Find a unique, sustained timbral change on either side of metadata.

        Long PRE/POST windows suppress ordinary chord and drum changes.
        A weak, broad or competing peak is not enough to overrule ICY time.
        """
        rate = float(pcm.sample_rate)
        samples: NDArray[np.int16] = np.frombuffer(pcm.data, dtype="<i2")
        if rate > 8000:
            stride = max(1, pcm.sample_rate // 8000)
            samples = samples[::stride]
            rate /= stride
        if len(samples) < 16 * rate:
            return None

        frame_size = round(rate * 0.19)
        hop_size = round(rate * 0.025)
        frames = np.lib.stride_tricks.sliding_window_view(samples, frame_size)[::hop_size]
        frequencies = np.fft.rfftfreq(frame_size, d=1.0 / rate)
        power = np.abs(np.fft.rfft(frames * np.hanning(frame_size), axis=1)) ** 2
        edges = np.geomspace(50.0, min(7500.0, rate * 0.48), 96)
        bands = np.stack(
            [
                power[:, (frequencies >= low) & (frequencies < high)].sum(axis=1)
                for low, high in zip(edges[:-1], edges[1:], strict=True)
            ],
            axis=1,
        )
        bands /= np.maximum(bands.sum(axis=1, keepdims=True), 1e-12)
        times = start_time + (np.arange(len(bands)) * hop_size + frame_size / 2) / rate
        prefix = np.vstack((np.zeros((1, bands.shape[1])), bands.cumsum(axis=0)))

        def mean(left: float, right: float) -> np.ndarray | None:
            i = int(np.searchsorted(times, left))
            j = int(np.searchsorted(times, right))
            if j - i < 16:
                return None
            return (prefix[j] - prefix[i]) / (j - i)

        scores: list[tuple[float, float]] = []
        for sample_time in np.arange(reference_time - 8, reference_time + 8.001, 0.05):
            time = float(sample_time)
            evidence = []
            for horizon in (6.0, 8.0):
                before = mean(time - horizon, time)
                after = mean(time, time + horizon)
                if before is None or after is None:
                    break
                norm = np.linalg.norm(before) * np.linalg.norm(after)
                evidence.append(1.0 - float(np.dot(before, after) / norm) if norm > 1e-12 else 0.0)
            if len(evidence) == 2:
                scores.append((min(evidence), float(time)))

        if not scores:
            return None
        best_score, best_time = max(scores)
        if best_score < 0.45:
            return None
        competing = max(
            (score for score, time in scores if abs(time - best_time) >= 1.0),
            default=0.0,
        )
        if best_score - competing < 0.06:
            return None
        return round(best_time, 6)
