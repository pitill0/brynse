"""Local PCM verification for one specialist-proposed temporal edge.

The specialist owns transition semantics and supplies the provisional edge.
This module only verifies the exact local timestamp against decoded PCM.

It does not know whether an edge is outgoing or incoming, and it does not
infer transition kind or ownership.
"""

from __future__ import annotations

from dataclasses import dataclass

from brynse.models import DecodedPcm
from brynse.pcm_features import (
    PcmDirectionalBoundaryEvidence,
    PcmFeatureTimeline,
)


@dataclass(frozen=True)
class PcmTemporalSearchBounds:
    """Absolute PCM interval supplied by the owning Specialist."""

    start_time_seconds: float
    end_time_seconds: float

    def __post_init__(self) -> None:
        if self.start_time_seconds > self.end_time_seconds:
            raise ValueError(
                "PCM temporal search start must not exceed end"
            )


@dataclass(frozen=True)
class PcmVerifiedTemporalEdge:
    """PCM verification of one specialist-proposed temporal edge."""

    provisional_time_seconds: float
    verified_time_seconds: float
    evidence_250ms: PcmDirectionalBoundaryEvidence
    evidence_500ms: PcmDirectionalBoundaryEvidence

    @property
    def delta_seconds(self) -> float:
        return (
            self.verified_time_seconds
            - self.provisional_time_seconds
        )


class PcmTemporalEdgeVerifier:
    """Verify one provisional temporal edge inside a constrained PCM window."""

    def __init__(
        self,
        *,
        search_radius_seconds: float = 2.0,
        step_seconds: float = 0.025,
        preserve_energy_change: float = 0.8,
        minimum_energy_change: float = 0.25,
        minimum_signal_rms: float = 0.005,
    ) -> None:
        if search_radius_seconds <= 0:
            raise ValueError(
                "search_radius_seconds must be greater than zero"
            )

        if step_seconds <= 0:
            raise ValueError(
                "step_seconds must be greater than zero"
            )

        for name, value in (
            ("preserve_energy_change", preserve_energy_change),
            ("minimum_energy_change", minimum_energy_change),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between zero and one"
                )

        if minimum_signal_rms < 0:
            raise ValueError(
                "minimum_signal_rms must be non-negative"
            )

        self._search_radius_seconds = search_radius_seconds
        self._step_seconds = step_seconds
        self._preserve_energy_change = preserve_energy_change
        self._minimum_energy_change = minimum_energy_change
        self._minimum_signal_rms = minimum_signal_rms

    @staticmethod
    def _by_time(
        candidates: tuple[PcmDirectionalBoundaryEvidence, ...],
    ) -> dict[
        float,
        dict[float, PcmDirectionalBoundaryEvidence],
    ]:
        grouped: dict[
            float,
            dict[float, PcmDirectionalBoundaryEvidence],
        ] = {}

        for candidate in candidates:
            key = round(candidate.time_seconds, 9)
            grouped.setdefault(key, {})[
                candidate.horizon_seconds
            ] = candidate

        return grouped

    def _coherent_pair(
        self,
        horizons: dict[
            float,
            PcmDirectionalBoundaryEvidence,
        ],
    ) -> tuple[
        PcmDirectionalBoundaryEvidence,
        PcmDirectionalBoundaryEvidence,
    ] | None:
        short = horizons.get(0.25)
        medium = horizons.get(0.5)

        if short is None or medium is None:
            return None

        if short.direction is not medium.direction:
            return None

        if (
            max(
                short.pre_rms,
                short.post_rms,
                medium.pre_rms,
                medium.post_rms,
            )
            < self._minimum_signal_rms
        ):
            return None

        return short, medium

    @staticmethod
    def _energy_support(
        pair: tuple[
            PcmDirectionalBoundaryEvidence,
            PcmDirectionalBoundaryEvidence,
        ],
    ) -> float:
        short, medium = pair
        return min(
            short.energy_change,
            medium.energy_change,
        )

    @staticmethod
    def _spectral_support(
        pair: tuple[
            PcmDirectionalBoundaryEvidence,
            PcmDirectionalBoundaryEvidence,
        ],
    ) -> float:
        short, medium = pair
        return min(
            short.spectral_change,
            medium.spectral_change,
        )

    def verify(
        self,
        *,
        pcm: DecodedPcm,
        absolute_start_time_seconds: float,
        provisional_time_seconds: float,
        search_bounds: PcmTemporalSearchBounds | None = None,
    ) -> PcmVerifiedTemporalEdge | None:
        """Verify one edge inside Specialist-owned or legacy local bounds."""

        if provisional_time_seconds < absolute_start_time_seconds:
            raise ValueError(
                "provisional edge precedes decoded PCM"
            )

        timeline = PcmFeatureTimeline.from_pcm(
            pcm=pcm,
            absolute_start_time_seconds=absolute_start_time_seconds,
        )

        provisional = timeline.directional_candidates(
            start_seconds=provisional_time_seconds,
            end_seconds=provisional_time_seconds,
            horizons_seconds=(0.25, 0.5),
        )

        provisional_grouped = self._by_time(provisional)

        provisional_pair = next(
            (
                pair
                for horizons in provisional_grouped.values()
                if (
                    pair := self._coherent_pair(horizons)
                )
                is not None
            ),
            None,
        )

        if (
            search_bounds is None
            and provisional_pair is not None
            and self._energy_support(provisional_pair)
            >= self._preserve_energy_change
        ):
            short, medium = provisional_pair

            return PcmVerifiedTemporalEdge(
                provisional_time_seconds=provisional_time_seconds,
                verified_time_seconds=provisional_time_seconds,
                evidence_250ms=short,
                evidence_500ms=medium,
            )

        if search_bounds is None:
            start_seconds = max(
                absolute_start_time_seconds,
                provisional_time_seconds
                - self._search_radius_seconds,
            )
            end_seconds = (
                provisional_time_seconds
                + self._search_radius_seconds
            )
        else:
            bytes_per_frame = (
                pcm.channels
                * pcm.sample_width_bytes
            )
            duration_seconds = (
                len(pcm.data)
                / bytes_per_frame
                / pcm.sample_rate
            )
            absolute_end_time_seconds = (
                absolute_start_time_seconds
                + duration_seconds
            )

            start_seconds = max(
                absolute_start_time_seconds,
                search_bounds.start_time_seconds,
            )
            end_seconds = min(
                absolute_end_time_seconds,
                search_bounds.end_time_seconds,
            )

            if start_seconds > end_seconds:
                return None

        candidates = timeline.directional_candidates(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            step_seconds=self._step_seconds,
            horizons_seconds=(0.25, 0.5),
        )

        viable: list[
            tuple[
                float,
                float,
                float,
                float,
                PcmDirectionalBoundaryEvidence,
                PcmDirectionalBoundaryEvidence,
            ]
        ] = []

        for time_seconds, horizons in self._by_time(
            candidates
        ).items():
            pair = self._coherent_pair(horizons)

            if pair is None:
                continue

            energy_support = self._energy_support(pair)

            if energy_support < self._minimum_energy_change:
                continue

            spectral_support = self._spectral_support(pair)

            viable.append(
                (
                    energy_support,
                    spectral_support,
                    -abs(
                        time_seconds
                        - provisional_time_seconds
                    ),
                    time_seconds,
                    pair[0],
                    pair[1],
                )
            )

        if not viable:
            return None

        (
            _energy_support,
            _spectral_support,
            _distance,
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

        return PcmVerifiedTemporalEdge(
            provisional_time_seconds=provisional_time_seconds,
            verified_time_seconds=verified_time,
            evidence_250ms=short,
            evidence_500ms=medium,
        )
