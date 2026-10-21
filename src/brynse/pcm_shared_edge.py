"""PCM verification of one provisional shared temporal edge.

Router and specialist geometry decide that a transition owns one shared edge.
This module only verifies the exact local timestamp against decoded PCM.

It does not infer transition kind or ownership.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """Verify a late provisional shared edge against local PCM.

    This first conservative policy only retracts a provisional edge.  It
    requires a rising acoustic event to agree across 250 ms and 500 ms
    horizons.  No transition kind or ownership is inferred here.
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
        """Return an earlier PCM-verified edge, or abstain."""

        if provisional_time_seconds < absolute_start_time_seconds:
            raise ValueError("provisional edge precedes decoded PCM")

        timeline = PcmFeatureTimeline.from_pcm(
            pcm=pcm,
            absolute_start_time_seconds=absolute_start_time_seconds,
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
