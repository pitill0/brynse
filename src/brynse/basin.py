"""Adaptive RMS-basin boundary proposal generation."""

from __future__ import annotations

import math
import statistics
from array import array
from dataclasses import dataclass

from brynse.boundaries import (
    BoundaryEvidence,
    BoundaryProposal,
    BoundaryProposalSource,
)
from brynse.models import DecodedPcm


@dataclass(frozen=True)
class Basin:
    """One adaptive quiet basin recovered from an RMS envelope."""

    start_seconds: float
    minimum_seconds: float
    recovery_seconds: float
    baseline_rms: float
    minimum_rms: float

    @property
    def depth(self) -> float:
        return self.minimum_rms / max(self.baseline_rms, 1e-12)

    @property
    def duration_seconds(self) -> float:
        return self.recovery_seconds - self.start_seconds


class AdaptiveBasinBoundaryDetector:
    """Generate basin proposals using the A13 adaptive RMS geometry.

    This intentionally stops at proposal generation. It does not apply the
    former A13 depth/local/far acceptance gate.
    """

    def __init__(
        self,
        *,
        rms_step_seconds: float = 0.10,
        baseline_seconds: float = 3.0,
        recovery_span_seconds: float = 0.5,
        max_recovery_seconds: float = 4.0,
        start_quiet_ratio: float = 0.30,
        basin_continue_ratio: float = 0.45,
        recovery_baseline_ratio: float = 0.35,
        recovery_jump_ratio: float = 3.0,
    ) -> None:
        if rms_step_seconds <= 0:
            raise ValueError("rms_step_seconds must be greater than zero")
        if baseline_seconds <= 0:
            raise ValueError("baseline_seconds must be greater than zero")
        if recovery_span_seconds <= 0:
            raise ValueError("recovery_span_seconds must be greater than zero")
        if max_recovery_seconds <= 0:
            raise ValueError("max_recovery_seconds must be greater than zero")
        if not 0 < start_quiet_ratio < 1:
            raise ValueError("start_quiet_ratio must be between zero and one")
        if not 0 < basin_continue_ratio < 1:
            raise ValueError("basin_continue_ratio must be between zero and one")
        if not 0 < recovery_baseline_ratio < 1:
            raise ValueError("recovery_baseline_ratio must be between zero and one")
        if recovery_jump_ratio <= 1:
            raise ValueError("recovery_jump_ratio must be greater than one")

        self._rms_step = rms_step_seconds
        self._baseline_seconds = baseline_seconds
        self._recovery_span_seconds = recovery_span_seconds
        self._max_recovery_seconds = max_recovery_seconds
        self._start_quiet_ratio = start_quiet_ratio
        self._basin_continue_ratio = basin_continue_ratio
        self._recovery_baseline_ratio = recovery_baseline_ratio
        self._recovery_jump_ratio = recovery_jump_ratio

    def detect_basins(self, *, pcm: DecodedPcm) -> tuple[Basin, ...]:
        """Return adaptive basins from mono 16-bit PCM."""
        levels = self._rms_levels(pcm)

        baseline_n = round(self._baseline_seconds / self._rms_step)
        recovery_n = round(self._recovery_span_seconds / self._rms_step)
        max_recovery_n = round(self._max_recovery_seconds / self._rms_step)

        if baseline_n <= 0 or recovery_n <= 0 or max_recovery_n <= 0:
            return ()

        out: list[Basin] = []
        index = baseline_n

        while index < len(levels) - recovery_n:
            baseline = statistics.median(levels[index - baseline_n : index])
            if baseline <= 0 or levels[index] > baseline * self._start_quiet_ratio:
                index += 1
                continue

            start = index
            members = [index]
            probe = index + 1
            local_n = max(1, round(0.3 / self._rms_step))

            while probe < len(levels):
                local = levels[max(start, probe - local_n + 1) : probe + 1]
                if statistics.median(local) > baseline * self._basin_continue_ratio:
                    break
                members.append(probe)
                probe += 1

            minimum = min(members, key=lambda item: levels[item])
            minimum_rms = levels[minimum]

            recovery: int | None = None
            end = min(len(levels), probe + max_recovery_n)

            for recovery_probe in range(
                max(probe, minimum + 1),
                end - recovery_n + 1,
            ):
                recovered = statistics.median(levels[recovery_probe : recovery_probe + recovery_n])
                if (
                    recovered >= baseline * self._recovery_baseline_ratio
                    and recovered >= max(minimum_rms, 1e-12) * self._recovery_jump_ratio
                ):
                    recovery = recovery_probe
                    break

            if recovery is None:
                index += 1
                continue

            out.append(
                Basin(
                    start_seconds=start * self._rms_step,
                    minimum_seconds=(minimum + 0.5) * self._rms_step,
                    recovery_seconds=recovery * self._rms_step,
                    baseline_rms=baseline,
                    minimum_rms=minimum_rms,
                )
            )
            index = max(index + 1, recovery + recovery_n)

        return tuple(out)

    def detect_proposals(
        self,
        *,
        pcm: DecodedPcm,
        absolute_start_time_seconds: float = 0.0,
    ) -> tuple[BoundaryProposal, ...]:
        """Expose adaptive basin minima as unresolved BASIN proposals."""
        return tuple(
            item.proposal
            for item in self.detect_evidence(
                pcm=pcm,
                absolute_start_time_seconds=absolute_start_time_seconds,
            )
        )

    def detect_evidence(
        self,
        *,
        pcm: DecodedPcm,
        absolute_start_time_seconds: float = 0.0,
    ) -> tuple[BoundaryEvidence, ...]:
        """Return basin proposals with descriptive basin-depth evidence."""
        if absolute_start_time_seconds < 0:
            raise ValueError("absolute_start_time_seconds must be non-negative")

        evidence: list[BoundaryEvidence] = []
        for basin in self.detect_basins(pcm=pcm):
            proposal = BoundaryProposal(
                time_seconds=absolute_start_time_seconds + basin.minimum_seconds,
                source=BoundaryProposalSource.BASIN,
                strength=None,
            )
            evidence.append(
                BoundaryEvidence(
                    proposal=proposal,
                    basin_depth=basin.depth,
                    basin_start_seconds=(
                        absolute_start_time_seconds + basin.start_seconds
                    ),
                    basin_recovery_seconds=(
                        absolute_start_time_seconds + basin.recovery_seconds
                    ),
                )
            )

        return tuple(evidence)

    def _rms_levels(self, pcm: DecodedPcm) -> tuple[float, ...]:
        samples = array("h")
        samples.frombytes(pcm.data)

        size = round(self._rms_step * pcm.sample_rate)
        if size <= 0:
            return ()

        levels: list[float] = []
        for start in range(0, len(samples) - size + 1, size):
            chunk = samples[start : start + size]
            square_sum = sum(sample * sample for sample in chunk)
            levels.append(math.sqrt(square_sum / len(chunk)))

        return tuple(levels)
