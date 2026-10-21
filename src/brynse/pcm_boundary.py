"""PCM-local boundary verification primitives.

Transition evidence and specialist geometry may propose where a temporal edge
lies. This module deliberately knows nothing about transition ownership or
semantic kind: it measures decoded PCM around candidate instants so that a
specialist can ultimately verify its temporal decision against the signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from brynse.models import DecodedPcm
from brynse.pcm_features import PcmBoundaryFeatures, PcmFeatureTimeline


@dataclass(frozen=True)
class PcmBoundaryObservation:
    """Descriptive in-memory PCM evidence at one possible boundary."""

    time_seconds: float
    features: PcmBoundaryFeatures


class PcmBoundaryVerifier:
    """Observe PCM around specialist-proposed temporal landmarks.

    This verifier is intentionally observational. It does not select a winning
    timestamp and therefore cannot modify production cuts.

    Candidate landmarks delimit places worth examining; the PCM feature
    timeline performs the actual signal measurement entirely in memory.
    """

    def __init__(
        self,
        *,
        scan_radius_seconds: float = 2.0,
        scan_step_seconds: float = 0.1,
    ) -> None:
        if scan_radius_seconds <= 0:
            raise ValueError("scan_radius_seconds must be greater than zero")

        if scan_step_seconds <= 0:
            raise ValueError("scan_step_seconds must be greater than zero")

        self._scan_radius_seconds = scan_radius_seconds
        self._scan_step_seconds = scan_step_seconds

    def observe(
        self,
        *,
        pcm: DecodedPcm,
        absolute_start_time_seconds: float,
        anchor_time_seconds: float,
    ) -> tuple[PcmBoundaryObservation, ...]:
        """Measure PCM features around one approximate boundary."""

        return self.observe_many(
            pcm=pcm,
            absolute_start_time_seconds=absolute_start_time_seconds,
            anchor_times_seconds=(anchor_time_seconds,),
        )

    def observe_many(
        self,
        *,
        pcm: DecodedPcm,
        absolute_start_time_seconds: float,
        anchor_times_seconds: tuple[float, ...],
    ) -> tuple[PcmBoundaryObservation, ...]:
        """Measure one deduplicated in-memory scan around several landmarks.

        Anchors only delimit places worth observing. They do not participate
        in ranking and never become temporal boundaries by authority.
        """

        if absolute_start_time_seconds < 0:
            raise ValueError("absolute_start_time_seconds must be non-negative")

        if any(anchor < absolute_start_time_seconds for anchor in anchor_times_seconds):
            raise ValueError("anchor time precedes decoded PCM")

        scan_times: dict[float, float] = {}

        for anchor_time_seconds in anchor_times_seconds:
            start = max(
                absolute_start_time_seconds,
                anchor_time_seconds - self._scan_radius_seconds,
            )
            stop = anchor_time_seconds + self._scan_radius_seconds

            index = 0

            while True:
                time_seconds = start + index * self._scan_step_seconds

                if time_seconds > stop + 1e-9:
                    break

                key = round(time_seconds, 9)
                scan_times[key] = time_seconds
                index += 1

        timeline = PcmFeatureTimeline.from_pcm(
            pcm=pcm,
            absolute_start_time_seconds=absolute_start_time_seconds,
        )

        measured = timeline.measure_many(
            times_seconds=tuple(scan_times[key] for key in sorted(scan_times)),
        )

        return tuple(
            PcmBoundaryObservation(
                time_seconds=features.time_seconds,
                features=features,
            )
            for features in measured
        )
